import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";

interface RequestFrame {
  jsonrpc: string;
  id: string;
  method: string;
  params: Record<string, unknown>;
}

class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];
  readyState = 0;
  sent: RequestFrame[] = [];
  url: string;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }

  send(frame: string): void {
    assert.equal(this.readyState, FakeWebSocket.OPEN);
    this.sent.push(JSON.parse(frame) as RequestFrame);
  }

  receive(message: unknown): void {
    this.onmessage?.({ data: JSON.stringify(message) });
  }

  close() {
    if (this.readyState === 3) return;
    this.readyState = 3;
    this.onclose?.();
  }
}

let api: typeof import("../src/api.ts");
let importCount = 0;
const cleanup: (() => void)[] = [];

beforeEach(async () => {
  FakeWebSocket.instances = [];
  globalThis.WebSocket = FakeWebSocket as unknown as typeof WebSocket;
  Object.defineProperty(globalThis, "location", { value: { protocol: "https:", host: "reachy.local" }, configurable: true });
  api = await import(`../src/api.ts?test=${++importCount}`) as typeof import("../src/api.ts");
});

afterEach(() => {
  for (const unsubscribe of cleanup.splice(0)) unsubscribe();
  for (const socket of FakeWebSocket.instances) socket.close();
});

test("concurrent requests share a socket and resolve their own responses", async () => {
  const voice = api.rpcCall("voices.current");
  const microphone = api.rpcCall("conversation.mic", { muted: true });
  assert.equal(FakeWebSocket.instances.length, 1);
  const [socket] = FakeWebSocket.instances;
  assert.ok(socket);
  assert.equal(socket.url, "wss://reachy.local/rpc");
  socket.open();
  await Promise.resolve();
  const [voiceRequest, microphoneRequest] = socket.sent;
  assert.ok(voiceRequest);
  assert.ok(microphoneRequest);
  assert.deepEqual(voiceRequest.params, {});
  assert.deepEqual(microphoneRequest.params, { muted: true });
  assert.equal(voiceRequest.jsonrpc, "2.0");
  socket.receive({ id: microphoneRequest.id, result: { muted: true } });
  socket.receive({ id: voiceRequest.id, result: { voice: "marin" } });
  assert.deepEqual(await voice, { voice: "marin" });
  assert.deepEqual(await microphone, { muted: true });
});

test("server errors retain stable reasons and readable details", async () => {
  const request = api.rpcCall("personalities.load", { name: "missing" });
  const [socket] = FakeWebSocket.instances;
  assert.ok(socket);
  socket.open();
  await Promise.resolve();
  const [sentRequest] = socket.sent;
  assert.ok(sentRequest);
  socket.receive({
    id: sentRequest.id,
    error: { code: -32000, message: "unknown_profile", data: { reason: "unknown_profile" } },
  });
  await assert.rejects(request, (error: unknown) => {
    assert.ok(error instanceof api.RpcError);
    assert.equal(error.reason, "unknown_profile");
    assert.equal(api.describeError(error), "That personality is no longer available.");
    return true;
  });
  assert.equal(api.describeError(new api.RpcError("failed", "new_reason", "Useful detail")), "Useful detail");
  assert.equal(api.describeError(new Error("Network unavailable")), "Network unavailable");
  assert.equal(api.describeError("Unexpected error"), "Unexpected error");
});

test("timeouts release requests and late responses cannot settle a newer request", async (context) => {
  context.mock.timers.enable({ apis: ["setTimeout"] });
  const request = api.rpcCall("voices.current", {}, { timeoutMs: 20 });
  const [socket] = FakeWebSocket.instances;
  assert.ok(socket);
  socket.open();
  await Promise.resolve();
  const [sentRequest] = socket.sent;
  assert.ok(sentRequest);
  const timedOutId = sentRequest.id;
  const rejected = assert.rejects(request, { reason: "timeout" });
  context.mock.timers.tick(20);
  await rejected;
  const nextRequest = api.rpcCall("voices.current");
  await Promise.resolve();
  socket.receive({ id: timedOutId, result: { voice: "stale" } });
  const nextFrame = socket.sent[1];
  assert.ok(nextFrame);
  socket.receive({ id: nextFrame.id, result: { voice: "marin" } });
  assert.deepEqual(await nextRequest, { voice: "marin" });
});

test("stalled opening rejects all waiting calls and permits a fresh connection", async (context) => {
  context.mock.timers.enable({ apis: ["setTimeout"] });
  context.mock.method(console, "warn", () => {});
  const connections: boolean[] = [];
  cleanup.push(api.subscribe("rpc.connection", ({ connected }) => connections.push(connected)));
  const voice = api.rpcCall("voices.current");
  const microphone = api.rpcCall("conversation.mic");
  const rejectedVoice = assert.rejects(voice, { reason: "timeout" });
  const rejectedMicrophone = assert.rejects(microphone, { reason: "timeout" });
  const [stalled] = FakeWebSocket.instances;
  assert.ok(stalled);
  context.mock.timers.tick(8000);
  await Promise.all([rejectedVoice, rejectedMicrophone]);
  assert.equal(stalled.readyState, 3);
  const retry = api.rpcCall("voices.current");
  const replacement = FakeWebSocket.instances[1];
  assert.ok(replacement);
  replacement.open();
  await Promise.resolve();
  stalled.onopen?.();
  stalled.onclose?.();
  assert.deepEqual(connections, [false, true]);
  const [request] = replacement.sent;
  assert.ok(request);
  replacement.receive({ id: request.id, result: { voice: "marin" } });
  assert.deepEqual(await retry, { voice: "marin" });
});

test("disconnect rejects pending calls and subscriptions retry failed reconnects", async (context) => {
  context.mock.timers.enable({ apis: ["setTimeout"] });
  const warning = context.mock.method(console, "warn", () => {});
  const connections: boolean[] = [];
  const activity: string[] = [];
  const stopConnection = api.subscribe("rpc.connection", ({ connected }) => connections.push(connected));
  cleanup.push(stopConnection);
  const stopActivity = api.subscribe("conversation.activity", ({ reason }) => activity.push(reason));
  cleanup.push(stopActivity);
  const [socket] = FakeWebSocket.instances;
  assert.ok(socket);
  socket.open();
  const request = api.rpcCall("voices.current");
  await Promise.resolve();
  socket.receive({ method: "conversation.activity", params: { reason: "playback_started" } });
  assert.deepEqual(activity, ["playback_started"]);
  const rejected = assert.rejects(request, { reason: "disconnected" });
  socket.close();
  await rejected;
  context.mock.timers.tick(1000);
  const failedRetry = FakeWebSocket.instances[1];
  assert.ok(failedRetry);
  failedRetry.close();
  await Promise.resolve();
  assert.equal(warning.mock.callCount(), 1);
  context.mock.timers.tick(1000);
  const replacement = FakeWebSocket.instances[2];
  assert.ok(replacement);
  replacement.open();
  replacement.receive({ method: "conversation.activity", params: { reason: "playback_stopped" } });
  assert.deepEqual(connections, [true, false, false, true]);
  assert.deepEqual(activity, ["playback_started", "playback_stopped"]);
  stopActivity();
  replacement.receive({ method: "conversation.activity", params: { reason: "playback_started" } });
  assert.deepEqual(activity, ["playback_started", "playback_stopped"]);
  replacement.close();
  stopConnection();
  context.mock.timers.tick(1000);
  assert.equal(FakeWebSocket.instances.length, 3);
});

test("malformed frames are logged without losing a valid response", async (context) => {
  const warning = context.mock.method(console, "warn", () => {});
  const request = api.rpcCall("voices.current");
  const [socket] = FakeWebSocket.instances;
  assert.ok(socket);
  socket.open();
  await Promise.resolve();
  socket.onmessage?.({ data: "invalid json" });
  socket.receive(null);
  assert.equal(warning.mock.callCount(), 2);
  const [sentRequest] = socket.sent;
  assert.ok(sentRequest);
  socket.receive({ id: sentRequest.id, result: { voice: "marin" } });
  assert.deepEqual(await request, { voice: "marin" });
});

test("send failures reject immediately and the next request can succeed", async (context) => {
  context.mock.timers.enable({ apis: ["setTimeout"] });
  const request = api.rpcCall("voices.current");
  const [socket] = FakeWebSocket.instances;
  assert.ok(socket);
  const send = context.mock.method(socket, "send", () => { throw new Error("Send failed"); });
  socket.open();
  await assert.rejects(request, /Send failed/);
  send.mock.restore();
  const nextRequest = api.rpcCall("voices.current");
  await Promise.resolve();
  const [sentRequest] = socket.sent;
  assert.ok(sentRequest);
  socket.receive({ id: sentRequest.id, result: { voice: "marin" } });
  assert.deepEqual(await nextRequest, { voice: "marin" });
  context.mock.timers.tick(8000);
});

test("startup retries return the eventual value and stop after a view aborts", async (context) => {
  context.mock.timers.enable({ apis: ["setTimeout"] });
  const controller = new AbortController();
  let attempts = 0;
  let notifications = 0;
  const request = api.untilReady(async () => {
    if (++attempts === 1) throw new Error("Starting up");
    return { ready: true };
  }, controller.signal, () => { notifications++; });
  await Promise.resolve();
  context.mock.timers.tick(2000);
  assert.deepEqual(await request, { ready: true });
  assert.equal(notifications, 1);
  controller.abort();
  await assert.rejects(api.untilReady(async () => {
    throw new Error("Disconnected");
  }, controller.signal), /Disconnected/);
});
