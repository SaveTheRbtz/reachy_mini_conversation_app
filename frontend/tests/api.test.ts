import assert from "node:assert/strict";
import { test } from "node:test";
import { create } from "@bufbuild/protobuf";
import { Code, ConnectError } from "@connectrpc/connect";
import { ConversationSchema, ListProfilesResponseSchema } from "../src/gen/reachy/conversation/v1/api_pb.ts";

Object.defineProperty(globalThis, "location", {
  value: { origin: "http://reachy.local" },
  configurable: true,
});
const { api, describeError, listProfiles, watchConversation } = await import("../src/api.ts");

test("profile selection includes every page", async (context) => {
  const requestedTokens: string[] = [];
  context.mock.method(api, "listProfiles", async (request: { pageToken?: string }) => {
    requestedTokens.push(request.pageToken ?? "");
    return create(ListProfilesResponseSchema, {
      profiles: [{ name: request.pageToken ? "profiles/user-second" : "profiles/user-first" }],
      nextPageToken: request.pageToken ? "" : "next",
    });
  });
  assert.deepEqual(
    (await listProfiles()).map((profile) => profile.name),
    ["profiles/user-first", "profiles/user-second"],
  );
  assert.deepEqual(requestedTokens, ["", "next"]);
});

for (const ending of ["failure", "clean EOF"]) {
  test(`watch reconnects after ${ending}, takes a fresh snapshot, and stops on abort`, async (context) => {
    context.mock.timers.enable({ apis: ["setTimeout"] });
    context.mock.method(console, "warn", () => {});
    const controller = new AbortController();
    let streams = 0;
    let errors = 0;
    const muted: boolean[] = [];
    context.mock.method(api, "watchConversation", async function* () {
      streams++;
      yield create(ConversationSchema, { name: "conversation", muted: streams > 1 });
      if (streams === 1) {
        if (ending === "failure") throw new ConnectError("Disconnected", Code.Unavailable);
        return;
      }
      controller.abort();
    });
    const watched = watchConversation(
      controller.signal,
      (snapshot) => muted.push(snapshot.muted),
      () => errors++,
    );
    for (let turn = 0; turn < 8; turn++) await Promise.resolve();
    context.mock.timers.tick(1000);
    await watched;
    assert.equal(errors, 1);
    assert.equal(streams, 2);
    assert.deepEqual(muted, [false, true]);
  });
}

test("healthy watch renewal stays quiet and server errors retain their readable messages", async (context) => {
  const controller = new AbortController();
  let streams = 0;
  context.mock.method(api, "watchConversation", async function* () {
    streams++;
    yield create(ConversationSchema);
    if (streams === 1) throw new ConnectError("Renew watch", Code.DeadlineExceeded);
    controller.abort();
  });
  await watchConversation(
    controller.signal,
    () => {},
    (error) => assert.fail(String(error)),
  );
  assert.equal(streams, 2);
  assert.equal(
    describeError(new ConnectError("That profile is in use", Code.FailedPrecondition)),
    "That profile is in use",
  );
});
