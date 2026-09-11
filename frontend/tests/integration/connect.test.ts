import { expect, test as base } from "vitest";
import { createClient, Code, type Client } from "@connectrpc/connect";
import { createConnectTransport } from "@connectrpc/connect-web";
import { ConversationService } from "../../src/gen/reachy/conversation/v1/api_pb.ts";
import { startBackend, type TestBackend } from "../support/backend.ts";

const test = base.extend<{ backend: TestBackend; client: Client<typeof ConversationService> }>({
  backend: async ({}, use) => {
    const server = await startBackend();
    try {
      await use(server);
    } finally {
      await server.stop();
    }
  },
  client: async ({ backend }, use) => {
    await use(
      createClient(
        ConversationService,
        createConnectTransport({ baseUrl: `${backend.origin}/rpc`, defaultTimeoutMs: 8000 }),
      ),
    );
  },
});

test("generated TypeScript and Python preserve field presence and canonical errors", async ({ client }) => {
  await expect(client.getConversation({ name: "missing" })).rejects.toMatchObject({
    code: Code.InvalidArgument,
  });
  const profile = await client.createProfile({
    profileId: "user-contract-test",
    profile: { instructions: "Test personality" },
  });
  const disabled = await client.updateProfile({
    profile: { name: profile.name, toolOverride: { toolIds: [] } },
    updateMask: { paths: ["tool_override"] },
  });
  expect(disabled.toolOverride?.toolIds).toEqual([]);
  expect(disabled.effectiveToolIds).toEqual([]);
  const reset = await client.updateProfile({
    profile: { name: profile.name },
    updateMask: { paths: ["tool_override"] },
  });
  expect(reset.toolOverride).toBeUndefined();
  for (const muted of [true, false]) {
    expect(
      await client.updateConversation({
        conversation: { name: "conversation", muted },
        updateMask: { paths: ["muted"] },
      }),
    ).toMatchObject({ muted });
  }
});

test("deadlines and canceled watches release server work while other commands remain responsive", async ({
  backend,
  client,
}) => {
  const controller = new AbortController();
  const snapshots = client
    .watchConversation({ name: "conversation" }, { signal: controller.signal, timeoutMs: 10000 })
    [Symbol.asyncIterator]();
  expect((await snapshots.next()).value?.muted).toBe(false);
  await fetch(`${backend.origin}/__test/faults`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ send_stalled: true }),
  });
  const stalled = expect(
    client.sendConversationText({ name: "conversation", text: "stall" }, { timeoutMs: 100 }),
  ).rejects.toMatchObject({ code: Code.DeadlineExceeded });
  expect(
    await client.updateConversation({
      conversation: { name: "conversation", muted: true },
      updateMask: { paths: ["muted"] },
    }),
  ).toMatchObject({ muted: true });
  expect((await snapshots.next()).value?.muted).toBe(true);
  await stalled;
  const canceledSnapshot = snapshots.next();
  controller.abort();
  await expect(canceledSnapshot).rejects.toMatchObject({ code: Code.Canceled });
  await expect
    .poll(() => fetch(`${backend.origin}/__test/state`).then((response) => response.json()), {
      timeout: 8000,
    })
    .toMatchObject({ active_watches: 0, active_says: 0, completed_says: 0, worker_alive: true });

  await fetch(`${backend.origin}/__test/faults`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ send_unavailable: true }),
  });
  await expect(
    client.sendConversationText({ name: "conversation", text: "unavailable" }),
  ).rejects.toMatchObject({
    code: Code.Unavailable,
  });
  await fetch(`${backend.origin}/__test/faults`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  await client.sendConversationText({ name: "conversation", text: "hello" });
  await expect
    .poll(() => fetch(`${backend.origin}/__test/state`).then((response) => response.json()))
    .toMatchObject({ completed_says: 1, sent_texts: ["hello"] });
});
