"""Entrypoint for the Reachy Mini conversation app."""

import time
import asyncio
import logging
import argparse
import threading
from pathlib import Path
from collections.abc import Callable, Awaitable

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini_conversation_app import app_lifecycle
from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.utils import parse_args, setup_logger, log_connection_troubleshooting
from reachy_mini_conversation_app.config import (
    REALTIME_MODEL,
    get_default_voice,
    set_instance_path,
    resolve_app_timeout_minutes,
    refresh_runtime_config_from_env,
)
from reachy_mini_conversation_app.memory import MemoryState
from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.prompts import get_session_voice
from reachy_mini_conversation_app.realtime import RealtimeConversation
from reachy_mini_conversation_app.tools.types import ToolResult, ToolDependencies
from reachy_mini_conversation_app.startup_settings import (
    StartupSettings,
    load_startup_settings_into_runtime,
)


def _start_inactivity_timeout_thread(
    timeout_minutes: float,
    stream: LocalStream,
    logger: logging.Logger,
    app_stop_event: threading.Event | None,
    go_to_sleep: Callable[[], ToolResult] | None = None,
) -> threading.Thread:
    """Start a daemon that puts the app to sleep after inactivity."""
    timeout_seconds = timeout_minutes * 60.0

    def poll_inactivity_timeout() -> None:
        logger.info("App inactivity timeout enabled: %.1f minutes", timeout_minutes)
        while app_stop_event is None or not app_stop_event.is_set():
            elapsed = stream.seconds_since_activity()
            if elapsed >= timeout_seconds:
                logger.info("No activity for %.1f minutes; going to sleep", elapsed / 60.0)
                try:
                    if go_to_sleep is not None:
                        go_to_sleep()
                    else:
                        stream.close()
                except Exception as error:
                    logger.error("Failed to stop after inactivity timeout: %s", error)
                    stream.close()
                return
            time.sleep(1.0)

    thread = threading.Thread(target=poll_inactivity_timeout, daemon=True, name="inactivity-timeout")
    thread.start()
    return thread


def main() -> None:
    """Run the command-line entrypoint."""
    args, _ = parse_args()
    run(args)


def run(
    args: argparse.Namespace,
    robot: ReachyMini | None = None,
    app_stop_event: threading.Event | None = None,
    settings_app: FastAPI | None = None,
    instance_path: str | Path | None = None,
) -> None:
    """Run the OpenAI Realtime conversation app."""
    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation App with %s", REALTIME_MODEL)
    set_instance_path(instance_path)
    startup_settings = StartupSettings()
    if instance_path is not None:
        env_path = Path(instance_path) / ".env"
        try:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=True)
                refresh_runtime_config_from_env()
                logger.info("Loaded instance configuration from %s", env_path)
            startup_settings = load_startup_settings_into_runtime(instance_path)
        except (OSError, ValueError) as error:
            logger.warning("Failed to load instance settings: %s", error)

    if robot is None:
        try:
            robot = ReachyMini(robot_name=args.robot_name) if args.robot_name is not None else ReachyMini()
        except (TimeoutError, ConnectionError) as error:
            logger.error("Failed to connect to Reachy Mini: %s", error)
            log_connection_troubleshooting(logger, args.robot_name)
            return
        except Exception as error:
            logger.error("Unexpected robot initialization error: %s", error)
            return

    app_lifecycle.wake_up_if_sleeping(robot, logger)
    movement_manager = MovementManager(current_robot=robot)
    dependencies = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        memory=MemoryState.load(instance_path),
        instance_path=instance_path,
        camera_enabled=not args.no_camera,
    )
    output_sample_rate = robot.media.get_output_audio_samplerate()
    selected_voice = startup_settings.voice or get_session_voice(default=get_default_voice())

    def build_conversation(voice: str) -> RealtimeConversation:
        return RealtimeConversation(
            dependencies,
            voice=voice,
            output_sample_rate=output_sample_rate,
        )

    effective_settings_app = settings_app
    if args.ui and effective_settings_app is None:
        effective_settings_app = FastAPI()

        @effective_settings_app.middleware("http")
        async def no_cache(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
            """Prevent browsers from retaining stale UI assets."""
            response = await call_next(request)
            response.headers["Cache-Control"] = "no-store"
            return response

    stream = LocalStream(
        robot,
        conversation_factory=build_conversation,
        settings_app=effective_settings_app,
        instance_path=instance_path,
        startup_voice=selected_voice,
    )
    if effective_settings_app is not None:
        stream.init_settings_ui()

    sleep_lock = threading.Lock()
    sleep_requested = threading.Event()

    def go_to_sleep_and_stop_app() -> ToolResult:
        if not sleep_lock.acquire(blocking=False):
            return {"status": "already_requested"}
        try:
            if sleep_requested.is_set():
                return {"status": "already_requested"}
            sleep_requested.set()
            movement_manager.stop(reset_to_neutral=False)
            sleep_error: str | None = None
            try:
                robot.disable_wobbling()
                robot.goto_sleep()
            except Exception as error:
                sleep_error = f"{type(error).__name__}: {error}"
                logger.error("Failed to move Reachy Mini to sleep pose: %s", error)
            stop_requested = False
            if app_stop_event is None or not app_stop_event.is_set():
                stop_requested = app_lifecycle.request_stop_current_app(robot, logger)
            if app_stop_event is not None:
                app_stop_event.set()
            else:
                stream.close()
            result: ToolResult = {
                "status": "sleeping" if sleep_error is None else "stop_requested",
                "stop_current_app_requested": stop_requested,
            }
            if sleep_error is not None:
                result["error"] = f"go_to_sleep movement failed: {sleep_error}"
            return result
        finally:
            sleep_lock.release()

    dependencies.go_to_sleep = go_to_sleep_and_stop_app

    ui_server: uvicorn.Server | None = None
    if args.ui and settings_app is None and effective_settings_app is not None:
        ui_server = uvicorn.Server(
            uvicorn.Config(effective_settings_app, host="0.0.0.0", port=7860, log_level="warning")
        )
        threading.Thread(target=ui_server.run, daemon=True, name="ui-server").start()
        logger.info("Web UI available at http://localhost:7860")

    movement_manager.start()
    robot.enable_wobbling()
    timeout_minutes = resolve_app_timeout_minutes()
    if timeout_minutes is not None:
        _start_inactivity_timeout_thread(
            timeout_minutes,
            stream,
            logger,
            app_stop_event,
            lambda: app_lifecycle.run_go_to_sleep_tool(dependencies, logger),
        )

    if app_stop_event is not None:

        def poll_stop_event() -> None:
            app_stop_event.wait()
            logger.info("App stop requested")
            stream.close()

        threading.Thread(target=poll_stop_event, daemon=True, name="app-stop").start()

    if not args.no_camera:
        try:
            robot.media.get_frame_jpeg()
            logger.info("Camera JPEG warm-up call completed")
        except Exception as error:
            logger.warning("Failed to prewarm camera JPEG pipeline: %s", error)

    try:
        stream.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption; shutting down")
    finally:
        if ui_server is not None:
            ui_server.should_exit = True
        movement_manager.stop(reset_to_neutral=False)
        try:
            robot.disable_wobbling()
            robot.media.close()
        except Exception as error:
            logger.debug("Media shutdown warning: %s", error)
        robot.client.disconnect()
        logger.info("Shutdown complete")


class ReachyMiniConversationApp(ReachyMiniApp):
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the app inside the Reachy Mini Apps runtime."""
        asyncio.set_event_loop(asyncio.new_event_loop())
        args, _ = parse_args()
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=self._get_instance_path().parent,
        )


if __name__ == "__main__":
    app = ReachyMiniConversationApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
