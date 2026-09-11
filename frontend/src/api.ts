import type { RpcMethods, RpcNotifications } from "./contracts.ts";

const DEFAULT_TIMEOUT_MS = 8000;
const RPC_URL = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/rpc`;

export class RpcError extends Error {
  readonly reason: string;
  readonly detail: string | undefined;

  constructor(message: string, reason = "rpc_error", detail?: string) {
    super(message || reason);
    this.name = "RpcError";
    this.reason = reason;
    this.detail = detail;
  }
}

interface PendingRequest {
  resolve: (result: unknown) => void;
  reject: (error: RpcError) => void;
  timer: ReturnType<typeof setTimeout>;
}

type Notification = {
  [Method in keyof RpcNotifications]: { method: Method; params: RpcNotifications[Method] };
}[keyof RpcNotifications];

type RpcMessage = Notification | {
  id: string | null;
  result?: unknown;
  error?: { message: string; data?: { reason?: string; detail?: string } };
};

let socket: WebSocket | null = null;
let connecting: Promise<void> | null = null;
let rpcCounter = 0;
const pending = new Map<string, PendingRequest>();
const subscribers: {
  [Method in keyof RpcNotifications]: Set<(params: RpcNotifications[Method]) => void>;
} = {
  "rpc.connection": new Set(),
  "conversation.activity": new Set(),
};

function connect(): Promise<void> {
  if (socket?.readyState === WebSocket.OPEN) return Promise.resolve();
  if (connecting) return connecting;
  connecting = new Promise((resolve, reject) => {
    let opened = false;
    const ws = new WebSocket(RPC_URL);
    socket = ws;
    ws.onopen = () => {
      opened = true;
      connecting = null;
      resolve();
      notify("rpc.connection", { connected: true });
    };
    ws.onmessage = (event: MessageEvent<string>) => handleMessage(event.data);
    ws.onclose = () => {
      socket = null;
      connecting = null;
      for (const request of pending.values()) {
        clearTimeout(request.timer);
        request.reject(new RpcError("connection closed", "disconnected"));
      }
      pending.clear();
      notify("rpc.connection", { connected: false });
      if (!opened) reject(new RpcError("cannot reach /rpc", "disconnected"));
      else if (Object.values(subscribers).some((callbacks) => callbacks.size > 0)) {
        setTimeout(() => {
          connect().catch((error: unknown) => console.warn("RPC reconnect failed:", error));
        }, 1000);
      }
    };
  });
  return connecting;
}

function handleMessage(frame: string): void {
  const message = JSON.parse(frame) as RpcMessage;
  if ("method" in message) {
    notify(message.method, message.params);
    return;
  }
  if (message.id === null) return;
  const request = pending.get(message.id);
  if (!request) return;
  pending.delete(message.id);
  clearTimeout(request.timer);
  if (message.error) {
    request.reject(new RpcError(message.error.message, message.error.data?.reason, message.error.data?.detail));
  } else {
    request.resolve(message.result);
  }
}

function notify<Method extends keyof RpcNotifications>(method: Method, params: RpcNotifications[Method]): void {
  for (const callback of subscribers[method] ?? []) {
    try {
      callback(params);
    } catch (error) {
      console.error(`subscribe(${method}) callback threw:`, error);
    }
  }
}

type CallArguments<Method extends keyof RpcMethods> = {} extends RpcMethods[Method]["params"]
  ? [params?: RpcMethods[Method]["params"], options?: { timeoutMs?: number }]
  : [params: RpcMethods[Method]["params"], options?: { timeoutMs?: number }];

export async function rpcCall<Method extends keyof RpcMethods>(
  method: Method,
  ...args: CallArguments<Method>
): Promise<RpcMethods[Method]["result"]> {
  const [params = {}, { timeoutMs = DEFAULT_TIMEOUT_MS } = {}] = args;
  await connect();
  const ws = socket;
  if (ws?.readyState !== WebSocket.OPEN) throw new RpcError("not connected", "disconnected");
  const id = `ui-${++rpcCounter}`;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new RpcError(`timed out: ${method}`, "timeout"));
    }, timeoutMs);
    pending.set(id, {
      resolve: (result) => resolve(result as RpcMethods[Method]["result"]),
      reject,
      timer,
    });
    ws.send(JSON.stringify({ jsonrpc: "2.0", id, method, params }));
  });
}

export function subscribe<Method extends keyof RpcNotifications>(
  method: Method,
  callback: (params: RpcNotifications[Method]) => void,
): () => void {
  const callbacks = subscribers[method];
  callbacks.add(callback);
  connect().catch((error: unknown) => console.warn("RPC connection failed:", error));
  return () => { callbacks.delete(callback); };
}

const STARTUP_POLL_MS = 2000;
const STARTUP_DEADLINE_MS = 90000;

export async function untilReady<Result>(
  request: () => Promise<Result>,
  signal: AbortSignal,
  onRetry?: () => void,
): Promise<Result> {
  const deadline = Date.now() + STARTUP_DEADLINE_MS;
  let notified = false;
  for (;;) {
    try {
      return await request();
    } catch (error) {
      if (signal.aborted || Date.now() >= deadline) throw error;
      if (!notified) {
        notified = true;
        onRetry?.();
      }
    }
    await new Promise((resolve) => setTimeout(resolve, STARTUP_POLL_MS));
    if (signal.aborted) throw new Error("view unmounted");
  }
}

const ERROR_MESSAGES: Readonly<Record<string, string>> = {
  invalid_openai_key: "Enter an OpenAI API key.",
  invalid_name: "Enter a valid profile name.",
  invalid_instructions: "Enter personality instructions.",
  profile_exists: "A personality with this name already exists.",
  invalid_tool_selection: "One or more selected tools are no longer available.",
  unknown_profile: "That personality is no longer available.",
  missing_voice: "Choose a voice first.",
  profile_locked: "Profile switching is locked by the administrator.",
  profile_in_use: "This personality is active or set to load at startup. Switch to another one first.",
  not_deletable: "This personality can't be deleted.",
  loop_unavailable: "Reachy is still starting up. Try again in a moment.",
};

export function describeError(error: unknown): string {
  if (error instanceof RpcError) return ERROR_MESSAGES[error.reason] ?? error.detail ?? error.message;
  return error instanceof Error ? error.message : String(error);
}
