import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { createInterface } from "node:readline";
import { setTimeout as delay } from "node:timers/promises";
import { test } from "node:test";
import { Code, ConnectError } from "@connectrpc/connect";
import { Conversation_ConnectionState as ConnectionState } from "../src/gen/reachy/conversation/v1/api_pb.ts";

test("generated browser client interoperates with the Python API and releases stalled work", { timeout: 20000 }, async () => {
  const python = process.env.PYTHON ?? (process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python");
  const server = spawn(python, ["-u", "tests/fixtures/connect_server.py"], {
    env: { ...process.env, REACHY_MINI_SKIP_DOTENV: "1", REACHY_MINI_CUSTOM_PROFILE: "", OPENAI_VOICE: "" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let logs = "";
  server.stderr.on("data", (chunk: Buffer) => { logs += chunk.toString(); });
  const lines = createInterface({ input: server.stdout });
  try {
    const startup = await Promise.race([
      once(lines, "line"),
      once(server, "exit").then(() => { throw new Error(`Python API exited before startup: ${logs}`); }),
      once(server, "error").then(([error]) => { throw error; }),
    ]);
    const address: unknown = JSON.parse(String(startup[0]));
    assert.ok(address && typeof address === "object" && "port" in address && typeof address.port === "number");
    const origin = `http://127.0.0.1:${address.port}`;
    Object.defineProperty(globalThis, "location", { value: { origin }, configurable: true });
    const { api, untilReady } = await import("../src/api.ts");
    const conversation = await untilReady(() => api.getConversation({ name: "conversation" }), AbortSignal.timeout(5000));
    assert.equal(conversation.connectionState, ConnectionState.CONNECTED);
    assert.equal(conversation.muted, false);
    await assert.rejects(api.getConversation({ name: "missing" }),
      (error: unknown) => error instanceof ConnectError && error.code === Code.InvalidArgument);

    const profile = await api.createProfile({
      profileId: "user-browser-test",
      profile: { instructions: "Test personality", greeting: "Hello" },
    });
    const disabled = await api.updateProfile({
      profile: { name: profile.name, toolOverride: { toolIds: [] } },
      updateMask: { paths: ["tool_override"] },
    });
    assert.ok(disabled.toolOverride);
    assert.deepEqual(disabled.effectiveToolIds, []);
    const edited = await api.updateProfile({
      profile: { name: profile.name, greeting: "Updated" },
      updateMask: { paths: ["greeting"] },
    });
    assert.equal(edited.instructions, "Test personality");
    assert.ok(edited.toolOverride);
    const reset = await api.updateProfile({
      profile: { name: profile.name }, updateMask: { paths: ["tool_override"] },
    });
    assert.equal(reset.toolOverride, undefined);
    await api.deleteProfile({ name: profile.name });

    const secret = "sk-synthetic-browser-test";
    const settings = await api.setSettingsApiKey({ name: "settings", apiKey: secret });
    assert.equal(settings.apiKeyConfigured, true);
    assert.ok(!JSON.stringify(settings).includes(secret));
    await api.restartConversation({ name: "conversation" });

    const controller = new AbortController();
    const snapshots = api.watchConversation({ name: "conversation" }, { signal: controller.signal, timeoutMs: 10000 })[Symbol.asyncIterator]();
    assert.equal((await snapshots.next()).value?.muted, false);
    const stalled = assert.rejects(api.sendConversationText({ name: "conversation", text: "stall" }, { timeoutMs: 100 }),
      (error: unknown) => error instanceof ConnectError && error.code === Code.DeadlineExceeded);
    const muted = await api.updateConversation({
      conversation: { name: "conversation", muted: true }, updateMask: { paths: ["muted"] },
    });
    assert.equal(muted.muted, true);
    assert.equal((await snapshots.next()).value?.muted, true);
    await stalled;
    controller.abort();
    await assert.rejects(snapshots.next(), (error: unknown) => error instanceof ConnectError && error.code === Code.Canceled);
    await api.sendConversationText({ name: "conversation", text: "hello" });
    await delay(5500);
    const stats: unknown = await fetch(`${origin}/stats`).then((response) => response.json());
    assert.deepEqual(stats, { active_watches: 0, active_says: 0, completed_says: 1, restarts: 1 });
  } finally {
    lines.close();
    if (server.exitCode === null) {
      server.kill();
      await once(server, "exit");
    }
  }
});
