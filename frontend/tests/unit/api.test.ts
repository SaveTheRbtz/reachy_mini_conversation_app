import { afterAll, expect, test, vi } from "vitest";
import { create } from "@bufbuild/protobuf";
import { Code, ConnectError } from "@connectrpc/connect";
import {
  type Conversation,
  ConversationSchema,
  ListProfilesResponseSchema,
} from "../../src/gen/reachy/conversation/v1/api_pb.ts";

vi.stubGlobal("location", { origin: "http://reachy.local" });
const { api, describeError, listProfiles, watchConversation } = await import("../../src/api.ts");
afterAll(() => vi.unstubAllGlobals());

async function stopWatch(controller: AbortController, watched: Promise<void>): Promise<void> {
  controller.abort();
  try {
    if (vi.isFakeTimers()) await vi.runAllTimersAsync();
    await watched;
  } finally {
    vi.useRealTimers();
  }
}

test("profile selection includes every page", async () => {
  const listing = vi.spyOn(api, "listProfiles").mockImplementation(async (request) =>
    create(ListProfilesResponseSchema, {
      profiles: [{ name: request.pageToken ? "profiles/user-second" : "profiles/user-first" }],
      nextPageToken: request.pageToken ? "" : "next",
    }),
  );
  expect((await listProfiles()).map((profile) => profile.name)).toEqual([
    "profiles/user-first",
    "profiles/user-second",
  ]);
  expect(listing.mock.calls.map(([request]) => request.pageToken)).toEqual(["", "next"]);
});

for (const ending of ["failure", "clean EOF"]) {
  test(`watch recovers after ${ending} with a fresh snapshot`, async ({ onTestFinished }) => {
    vi.useFakeTimers();
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const controller = new AbortController();
    const onSnapshot = vi.fn<(snapshot: Conversation) => void>();
    const onError = vi.fn();
    let streams = 0;
    const watching = vi.spyOn(api, "watchConversation").mockImplementation(async function* () {
      streams++;
      yield create(ConversationSchema, { name: "conversation", muted: streams > 1 });
      if (streams === 1) {
        if (ending === "failure") throw new ConnectError("Disconnected", Code.Unavailable);
        return;
      }
      controller.abort();
    });
    const watched = watchConversation(controller.signal, onSnapshot, onError);
    onTestFinished(() => stopWatch(controller, watched));
    await vi.advanceTimersByTimeAsync(0);
    expect(onError).toHaveBeenCalledOnce();
    await vi.advanceTimersByTimeAsync(999);
    expect(watching).toHaveBeenCalledOnce();
    await vi.advanceTimersByTimeAsync(1);
    await watched;
    expect(watching).toHaveBeenCalledTimes(2);
    expect(onSnapshot.mock.calls.map(([snapshot]) => snapshot.muted)).toEqual([false, true]);
  });
}

test("aborting during retry stops the watch without opening another stream", async ({ onTestFinished }) => {
  vi.useFakeTimers();
  vi.spyOn(console, "warn").mockImplementation(() => {});
  const controller = new AbortController();
  const onError = vi.fn();
  const watching = vi.spyOn(api, "watchConversation").mockImplementation(async function* () {
    throw new ConnectError("Unavailable", Code.Unavailable);
  });
  const watched = watchConversation(controller.signal, vi.fn(), onError);
  onTestFinished(() => stopWatch(controller, watched));
  await vi.advanceTimersByTimeAsync(0);
  expect(onError).toHaveBeenCalledOnce();
  controller.abort();
  await watched;
  await vi.advanceTimersByTimeAsync(1000);
  expect(watching).toHaveBeenCalledOnce();
});

test("a deadline after a stale snapshot reports failure before retrying", async ({ onTestFinished }) => {
  vi.useFakeTimers();
  vi.spyOn(console, "warn").mockImplementation(() => {});
  const controller = new AbortController();
  const onError = vi.fn();
  vi.spyOn(api, "watchConversation").mockImplementation(async function* () {
    yield create(ConversationSchema);
    await new Promise((resolve) => setTimeout(resolve, 11000));
    throw new ConnectError("Deadline exceeded", Code.DeadlineExceeded);
  });
  const watched = watchConversation(controller.signal, vi.fn(), onError);
  onTestFinished(() => stopWatch(controller, watched));
  await vi.advanceTimersByTimeAsync(11000);
  expect(onError).toHaveBeenCalledOnce();
  controller.abort();
  await watched;
});

test("a deadline on a healthy watch renews silently", async ({ onTestFinished }) => {
  const controller = new AbortController();
  let streams = 0;
  vi.spyOn(api, "watchConversation").mockImplementation(async function* () {
    streams++;
    yield create(ConversationSchema);
    if (streams === 1) throw new ConnectError("Renew watch", Code.DeadlineExceeded);
    controller.abort();
  });
  const onError = vi.fn();
  const watched = watchConversation(controller.signal, vi.fn(), onError);
  onTestFinished(() => stopWatch(controller, watched));
  await watched;
  expect(streams).toBe(2);
  expect(onError).not.toHaveBeenCalled();
});

test("server errors retain their readable messages", () => {
  expect(describeError(new ConnectError("That profile is in use", Code.FailedPrecondition))).toBe(
    "That profile is in use",
  );
});
