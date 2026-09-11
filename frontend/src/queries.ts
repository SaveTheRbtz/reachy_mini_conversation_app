import { MutationCache, QueryClient, queryOptions } from "@tanstack/vue-query";
import { Code, ConnectError } from "@connectrpc/connect";
import { api, listProfiles } from "./api.ts";

export const queryClient = new QueryClient({
  mutationCache: new MutationCache({ onError: (error) => console.warn("Command failed", error) }),
  defaultOptions: {
    queries: {
      networkMode: "always",
      staleTime: 30000,
      refetchOnWindowFocus: false,
      retry: (failures, error) =>
        failures < 2 && [Code.Unavailable, Code.DeadlineExceeded].includes(ConnectError.from(error).code),
    },
    mutations: {
      networkMode: "always",
      retry: false,
      gcTime: 0,
    },
  },
});

export const profilesQuery = queryOptions({
  queryKey: ["profiles"],
  queryFn: ({ signal }) => listProfiles(signal),
});
export const settingsQuery = queryOptions({
  queryKey: ["settings"],
  queryFn: ({ signal }) => api.getSettings({ name: "settings" }, { signal }),
});
export const capabilitiesQuery = queryOptions({
  queryKey: ["capabilities"],
  queryFn: ({ signal }) => api.getCapabilities({ name: "capabilities" }, { signal }),
});
export function profileQuery(name: string) {
  return queryOptions({
    queryKey: ["profiles", name],
    enabled: Boolean(name),
    queryFn: ({ signal }) => api.getProfile({ name }, { signal }),
  });
}
