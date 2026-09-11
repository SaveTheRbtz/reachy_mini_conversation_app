import { createClient, Code, ConnectError } from "@connectrpc/connect";
import { createConnectTransport } from "@connectrpc/connect-web";
import { ConversationService, type Conversation, type Profile } from "./gen/reachy/conversation/v1/api_pb.ts";

export const api = createClient(
  ConversationService,
  createConnectTransport({
    baseUrl: `${location.origin}/rpc`,
    defaultTimeoutMs: 8000,
  }),
);

export async function listProfiles(signal?: AbortSignal): Promise<Profile[]> {
  const profiles: Profile[] = [];
  let pageToken = "";
  do {
    const page = await api.listProfiles({ pageToken }, { signal });
    profiles.push(...page.profiles);
    pageToken = page.nextPageToken;
  } while (pageToken);
  return profiles;
}

export async function watchConversation(
  signal: AbortSignal,
  onSnapshot: (conversation: Conversation) => void,
  onError: (error: unknown) => void,
): Promise<void> {
  while (!signal.aborted) {
    let receivedAt = 0;
    try {
      for await (const conversation of api.watchConversation(
        { name: "conversation" },
        { signal, timeoutMs: 30000 },
      )) {
        receivedAt = Date.now();
        onSnapshot(conversation);
      }
      if (!signal.aborted) throw new ConnectError("Conversation stream ended", Code.Unavailable);
    } catch (error) {
      if (signal.aborted) return;
      // Renew a healthy watch silently; the server sends a snapshot every five seconds.
      if (ConnectError.from(error).code === Code.DeadlineExceeded && Date.now() - receivedAt < 10000)
        continue;
      console.warn("Conversation watch failed", error);
      onError(error);
    }
    await waitForRetry(signal);
  }
}

function waitForRetry(signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(done, 1000);
    signal.addEventListener("abort", done, { once: true });
    if (signal.aborted) done();
    function done() {
      clearTimeout(timer);
      signal.removeEventListener("abort", done);
      resolve();
    }
  });
}

export function describeError(error: unknown): string {
  if (error instanceof ConnectError) return error.rawMessage;
  return error instanceof Error ? error.message : String(error);
}
