import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { resolve } from "node:path";
import { createInterface } from "node:readline";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";

export interface TestBackend {
  origin: string;
  stop(): Promise<void>;
}

export async function startBackend(args: string[] = []): Promise<TestBackend> {
  const root = fileURLToPath(new URL("../../", import.meta.url));
  const python =
    process.env.PYTHON ??
    resolve(root, process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python");
  const server = spawn(python, ["-u", "tests/fixtures/connect_server.py", ...args], {
    cwd: root,
    env: {
      ...process.env,
      REACHY_MINI_SKIP_DOTENV: "1",
      REACHY_MINI_CUSTOM_PROFILE: "",
      OPENAI_VOICE: "",
      OPENAI_API_KEY: "",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let logs = "";
  let origin: string | undefined;
  server.stderr.on("data", (chunk: Buffer) => {
    logs += chunk.toString();
  });
  const lines = createInterface({ input: server.stdout });
  async function stop(): Promise<void> {
    lines.close();
    if (!server.pid || server.exitCode !== null || server.signalCode !== null) return;
    if (origin) {
      try {
        await fetch(`${origin}/__test/faults`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: "{}",
          signal: AbortSignal.timeout(1000),
        });
      } catch (error) {
        console.warn("Could not reset backend faults before shutdown", error);
      }
    }
    const exited = once(server, "exit");
    server.kill();
    const timer = setTimeout(() => server.kill("SIGKILL"), 5000);
    try {
      await exited;
    } finally {
      clearTimeout(timer);
    }
  }
  try {
    const signal = AbortSignal.timeout(15000);
    const startup = await Promise.race([
      once(lines, "line", { signal }),
      once(server, "exit", { signal }).then(() => {
        throw new Error(`Python API exited before startup: ${logs}`);
      }),
    ]);
    const address: unknown = JSON.parse(String(startup[0]));
    assert.ok(
      address && typeof address === "object" && "port" in address && typeof address.port === "number",
    );
    origin = `http://127.0.0.1:${address.port}`;
    for (;;) {
      signal.throwIfAborted();
      try {
        const response = await fetch(`${origin}/__test/state`, { signal });
        const state: unknown = await response.json();
        if (response.ok && state && typeof state === "object" && "ready" in state && state.ready === true) {
          return { origin, stop };
        }
      } catch (error) {
        if (!(error instanceof TypeError)) throw error;
      }
      await delay(50, undefined, { signal });
    }
  } catch (error) {
    await stop();
    throw new Error(`Python API startup failed: ${logs}`, { cause: error });
  }
}
