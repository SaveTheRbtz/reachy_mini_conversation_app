import assert from "node:assert/strict";
import { setTimeout as delay } from "node:timers/promises";
import { test } from "node:test";
import { Code, ConnectError } from "@connectrpc/connect";
import { Conversation_ConnectionState as ConnectionState } from "../src/gen/reachy/conversation/v1/api_pb.ts";
import { startBackend } from "./backend.ts";

test(
  "generated browser client interoperates with the Python API and releases stalled work",
  { timeout: 30000 },
  async () => {
    const server = await startBackend();
    try {
      const { origin } = server;
      Object.defineProperty(globalThis, "location", { value: { origin }, configurable: true });
      const { api } = await import("../src/api.ts");
      let conversation = await api.getConversation({ name: "conversation" });
      const readiness = AbortSignal.timeout(5000);
      while (conversation.connectionState !== ConnectionState.CONNECTED) {
        await delay(50, undefined, { signal: readiness });
        conversation = await api.getConversation({ name: "conversation" });
      }
      assert.equal(conversation.connectionState, ConnectionState.CONNECTED);
      assert.equal(conversation.muted, false);
      await assert.rejects(
        api.getConversation({ name: "missing" }),
        (error: unknown) => error instanceof ConnectError && error.code === Code.InvalidArgument,
      );

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
        profile: { name: profile.name },
        updateMask: { paths: ["tool_override"] },
      });
      assert.equal(reset.toolOverride, undefined);
      await api.deleteProfile({ name: profile.name });

      const secret = "sk-synthetic-browser-test";
      const settings = await api.setSettingsApiKey({ name: "settings", apiKey: secret });
      assert.equal(settings.apiKeyConfigured, true);
      assert.ok(!JSON.stringify(settings).includes(secret));
      await api.restartConversation({ name: "conversation" });

      const controller = new AbortController();
      const snapshots = api
        .watchConversation({ name: "conversation" }, { signal: controller.signal, timeoutMs: 10000 })
        [Symbol.asyncIterator]();
      assert.equal((await snapshots.next()).value?.muted, false);
      await fetch(`${origin}/__test/faults`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ send_stalled: true }),
      });
      const stalled = assert.rejects(
        api.sendConversationText({ name: "conversation", text: "stall" }, { timeoutMs: 100 }),
        (error: unknown) => error instanceof ConnectError && error.code === Code.DeadlineExceeded,
      );
      const muted = await api.updateConversation({
        conversation: { name: "conversation", muted: true },
        updateMask: { paths: ["muted"] },
      });
      assert.equal(muted.muted, true);
      assert.equal((await snapshots.next()).value?.muted, true);
      await stalled;
      // Keep reading when canceled, as the production for-await loop does.
      const canceledSnapshot = snapshots.next();
      controller.abort();
      await assert.rejects(
        canceledSnapshot,
        (error: unknown) => error instanceof ConnectError && error.code === Code.Canceled,
      );
      await fetch(`${origin}/__test/faults`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ send_unavailable: true }),
      });
      await assert.rejects(
        api.sendConversationText({ name: "conversation", text: "unavailable" }),
        (error: unknown) => error instanceof ConnectError && error.code === Code.Unavailable,
      );
      await fetch(`${origin}/__test/faults`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      await api.sendConversationText({ name: "conversation", text: "hello" });
      await delay(5500);
      const state: unknown = await fetch(`${origin}/__test/state`).then((response) => response.json());
      assert.ok(state && typeof state === "object");
      assert.equal("active_watches" in state && state.active_watches, 0);
      assert.equal("active_says" in state && state.active_says, 0);
      assert.equal("completed_says" in state && state.completed_says, 1);
      assert.equal("worker_alive" in state && state.worker_alive, true);
    } finally {
      await server.stop();
    }
  },
);
