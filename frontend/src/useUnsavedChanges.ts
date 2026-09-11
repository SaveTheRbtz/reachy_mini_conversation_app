import { computed, onBeforeUnmount, shallowRef, watch, type Ref } from "vue";
import { onBeforeRouteLeave, onBeforeRouteUpdate } from "vue-router";

export function useUnsavedChanges(dirty: Readonly<Ref<boolean>>, busy?: Readonly<Ref<boolean>>) {
  const pending = shallowRef<(leave: boolean) => void>();

  function confirm() {
    if (!dirty.value) return true;
    if (busy?.value || pending.value) return false;
    return new Promise<boolean>((resolve) => {
      pending.value = resolve;
    });
  }

  function decide(leave: boolean) {
    pending.value?.(leave);
    pending.value = undefined;
  }

  function beforeUnload(event: BeforeUnloadEvent) {
    event.preventDefault();
    event.returnValue = "";
  }

  onBeforeRouteLeave(confirm);
  onBeforeRouteUpdate(confirm);
  watch(
    dirty,
    (unsaved) => {
      if (unsaved) window.addEventListener("beforeunload", beforeUnload);
      else window.removeEventListener("beforeunload", beforeUnload);
    },
    { immediate: true },
  );
  onBeforeUnmount(() => {
    window.removeEventListener("beforeunload", beforeUnload);
    decide(false);
  });

  return { confirming: computed(() => Boolean(pending.value)), decide };
}
