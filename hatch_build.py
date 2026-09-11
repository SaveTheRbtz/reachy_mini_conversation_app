import shutil
import subprocess
from pathlib import Path

from hatchling.builders.config import BuilderConfig
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface[BuilderConfig]):
    """Build generated API code and browser assets before packaging source checkouts."""

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        """Reuse complete distribution artifacts or build them from the authored sources."""
        root = Path(self.root)
        if version == "editable" or not (root / "PKG-INFO").is_file():
            npm = shutil.which("npm")
            if npm is None:
                raise RuntimeError(
                    "Building from source requires Node.js 22.12 or newer; install a wheel or sdist instead"
                )
            subprocess.run([npm, "ci", "--ignore-scripts", "--include=dev"], cwd=root, check=True)
            subprocess.run([npm, "run", "build"], cwd=root, check=True)

        package = root / "src" / "reachy_mini_conversation_app"
        for target in (
            "gen/reachy/conversation/v1/api_pb.py",
            "gen/reachy/conversation/v1/api_connect.py",
            "static/index.html",
        ):
            if not (package / target).is_file():
                raise RuntimeError(f"Missing build artifact: {target}")
