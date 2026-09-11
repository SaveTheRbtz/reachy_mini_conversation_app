import type { RpcMethods } from "./contracts.ts";

type PendingApply = { name: string; promise: Promise<RpcMethods["personalities.apply"]["result"]> };
let pending: PendingApply | null = null;

export function setPendingApply(entry: PendingApply | null): void {
  pending = entry;
  // The request can fail before the talk view mounts to consume it.
  entry?.promise.catch((error: unknown) => console.warn("Personality apply failed", error));
}

export function consumePendingApply(): PendingApply | null {
  const entry = pending;
  pending = null;
  return entry;
}
