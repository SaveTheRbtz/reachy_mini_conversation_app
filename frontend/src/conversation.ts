import { inject, onMounted, onUnmounted, provide, ref, shallowRef, type InjectionKey } from "vue";
import { describeError, watchConversation } from "./api.ts";
import type { Conversation } from "./gen/reachy/conversation/v1/api_pb.ts";

const conversationKey: InjectionKey<ReturnType<typeof provideConversation>> = Symbol("conversation");

export function provideConversation() {
  const snapshot = shallowRef<Conversation>();
  const error = shallowRef<string>();
  const connected = ref(false);
  const controller = new AbortController();
  const state = { snapshot, error, connected };
  provide(conversationKey, state);
  onMounted(() => {
    void watchConversation(
      controller.signal,
      (current) => {
        snapshot.value = current;
        error.value = undefined;
        connected.value = true;
      },
      (failure) => {
        connected.value = false;
        error.value = describeError(failure);
      },
    );
  });
  onUnmounted(() => controller.abort());
  return state;
}

export function useConversation() {
  const state = inject(conversationKey);
  if (!state) throw new Error("Conversation state must be provided by the app.");
  return state;
}
