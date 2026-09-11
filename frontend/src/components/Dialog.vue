<script setup lang="ts">
import { onBeforeUnmount, onMounted, useTemplateRef } from "vue";

withDefaults(defineProps<{ title: string; confirmLabel?: string; danger?: boolean; busy?: boolean }>(), {
  confirmLabel: "Continue",
  danger: false,
  busy: false,
});
const emit = defineEmits<{ confirm: []; cancel: [] }>();
const dialog = useTemplateRef<HTMLDialogElement>("dialog");
onMounted(() => dialog.value?.showModal());
onBeforeUnmount(() => dialog.value?.close());
</script>

<template>
  <dialog ref="dialog" aria-labelledby="dialog-title" @cancel.prevent="!busy && emit('cancel')">
    <h2 id="dialog-title">{{ title }}</h2>
    <div class="dialog-body"><slot /></div>
    <div class="actions">
      <button class="button quiet" type="button" :disabled="busy" autofocus @click="emit('cancel')">
        Cancel
      </button>
      <button
        class="button"
        :class="danger ? 'danger' : 'primary'"
        type="button"
        :disabled="busy"
        @click="emit('confirm')"
      >
        {{ busy ? "Working…" : confirmLabel }}
      </button>
    </div>
  </dialog>
</template>

<style scoped>
dialog {
  width: min(28rem, calc(100% - 2rem));
  border: 1px solid var(--border);
  border-radius: 1.5rem;
  padding: 1.75rem;
  background: var(--surface);
  color: var(--text);
  box-shadow: var(--shadow);
}
dialog::backdrop {
  background: #10251db3;
  backdrop-filter: blur(4px);
}
h2 {
  margin-top: 0;
}
.dialog-body {
  margin: 1rem 0 1.75rem;
  color: var(--muted);
}
.actions {
  justify-content: flex-end;
}
</style>
