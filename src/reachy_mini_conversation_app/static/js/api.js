const DEFAULT_TIMEOUT_MS = 8000;
const RPC_URL = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/rpc`;
export class RpcError extends Error {
    reason;
    detail;
    constructor(message, reason = "rpc_error", detail) {
        super(message || reason);
        this.name = "RpcError";
        this.reason = reason;
        this.detail = detail;
    }
}
let socket = null;
let connecting = null;
let rpcCounter = 0;
const pending = new Map();
const subscribers = {
    "rpc.connection": new Set(),
    "conversation.activity": new Set(),
};
function connect() {
    if (socket?.readyState === WebSocket.OPEN)
        return Promise.resolve();
    if (connecting)
        return connecting;
    connecting = new Promise((resolve, reject) => {
        const ws = new WebSocket(RPC_URL);
        socket = ws;
        const openingTimer = setTimeout(() => {
            reject(new RpcError("timed out connecting to /rpc", "timeout"));
            disconnected();
            ws.close();
        }, DEFAULT_TIMEOUT_MS);
        ws.onopen = () => {
            if (socket !== ws)
                return;
            clearTimeout(openingTimer);
            connecting = null;
            resolve();
            notify("rpc.connection", { connected: true });
        };
        ws.onmessage = (event) => {
            if (socket !== ws)
                return;
            try {
                handleMessage(event.data);
            }
            catch (error) {
                console.warn("Invalid RPC message:", error);
            }
        };
        ws.onclose = disconnected;
        function disconnected() {
            clearTimeout(openingTimer);
            if (socket !== ws)
                return;
            socket = null;
            connecting = null;
            for (const request of pending.values()) {
                clearTimeout(request.timer);
                request.reject(new RpcError("connection closed", "disconnected"));
            }
            pending.clear();
            notify("rpc.connection", { connected: false });
            reject(new RpcError("cannot reach /rpc", "disconnected"));
            if (Object.values(subscribers).some((callbacks) => callbacks.size > 0)) {
                setTimeout(() => {
                    if (Object.values(subscribers).some((callbacks) => callbacks.size > 0)) {
                        connect().catch((error) => console.warn("RPC reconnect failed:", error));
                    }
                }, 1000);
            }
        }
    });
    return connecting;
}
function handleMessage(frame) {
    const message = JSON.parse(frame);
    if ("method" in message) {
        notify(message.method, message.params);
        return;
    }
    if (message.id === null)
        return;
    const request = pending.get(message.id);
    if (!request)
        return;
    pending.delete(message.id);
    clearTimeout(request.timer);
    if (message.error) {
        request.reject(new RpcError(message.error.message, message.error.data?.reason, message.error.data?.detail));
    }
    else {
        request.resolve(message.result);
    }
}
function notify(method, params) {
    for (const callback of subscribers[method] ?? []) {
        try {
            callback(params);
        }
        catch (error) {
            console.error(`subscribe(${method}) callback threw:`, error);
        }
    }
}
export async function rpcCall(method, ...args) {
    const [params = {}, { timeoutMs = DEFAULT_TIMEOUT_MS } = {}] = args;
    await connect();
    const ws = socket;
    if (ws?.readyState !== WebSocket.OPEN)
        throw new RpcError("not connected", "disconnected");
    const id = `ui-${++rpcCounter}`;
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
            pending.delete(id);
            reject(new RpcError(`timed out: ${method}`, "timeout"));
        }, timeoutMs);
        pending.set(id, {
            resolve: (result) => resolve(result),
            reject,
            timer,
        });
        try {
            ws.send(JSON.stringify({ jsonrpc: "2.0", id, method, params }));
        }
        catch (error) {
            pending.delete(id);
            clearTimeout(timer);
            reject(error);
        }
    });
}
export function subscribe(method, callback) {
    const callbacks = subscribers[method];
    callbacks.add(callback);
    connect().catch((error) => console.warn("RPC connection failed:", error));
    return () => { callbacks.delete(callback); };
}
const STARTUP_POLL_MS = 2000;
const STARTUP_DEADLINE_MS = 90000;
export async function untilReady(request, signal, onRetry) {
    const deadline = Date.now() + STARTUP_DEADLINE_MS;
    let notified = false;
    for (;;) {
        try {
            return await request();
        }
        catch (error) {
            if (signal.aborted || Date.now() >= deadline)
                throw error;
            if (!notified) {
                notified = true;
                onRetry?.();
            }
        }
        await new Promise((resolve) => setTimeout(resolve, STARTUP_POLL_MS));
        if (signal.aborted)
            throw new Error("view unmounted");
    }
}
const ERROR_MESSAGES = {
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
export function describeError(error) {
    if (error instanceof RpcError)
        return ERROR_MESSAGES[error.reason] ?? error.detail ?? error.message;
    return error instanceof Error ? error.message : String(error);
}
