<script setup lang="ts">
import { computed, ref } from "vue";
import { useMutation } from "@tanstack/vue-query";
import { api, describeError } from "../api.ts";
import { useConversation } from "../conversation.ts";
import { profileAvatar, profileLabel } from "../presentation.ts";
import { Conversation_ConnectionState as ConnectionState } from "../gen/reachy/conversation/v1/api_pb.ts";

const { snapshot, error, connected } = useConversation();
const message = ref("");
const sent = ref(false);
const ready = computed(
  () => connected.value && snapshot.value?.connectionState === ConnectionState.CONNECTED,
);
const heading = computed(() =>
  !connected.value
    ? "Connecting to your robot"
    : !ready.value
      ? "Conversation unavailable"
      : snapshot.value?.playing
        ? "Reachy is speaking"
        : snapshot.value?.muted
          ? "Microphone paused"
          : "Listening to you",
);
const status = computed(() =>
  !connected.value
    ? "Robot offline"
    : !ready.value
      ? "Conversation offline"
      : `${snapshot.value?.playing ? "Speaking · " : ""}${snapshot.value?.muted ? "Microphone muted" : "Microphone on"}`,
);
const microphone = useMutation({
  mutationFn: (muted: boolean) =>
    api.updateConversation({
      conversation: { name: "conversation", muted },
      updateMask: { paths: ["muted"] },
    }),
  onSuccess: (current) => {
    if (snapshot.value) snapshot.value = { ...snapshot.value, muted: current.muted };
  },
});
const interrupt = useMutation({ mutationFn: () => api.interruptConversation({ name: "conversation" }) });
const restart = useMutation({ mutationFn: () => api.restartConversation({ name: "conversation" }) });
const send = useMutation({
  mutationFn: () => api.sendConversationText({ name: "conversation", text: message.value.trim() }),
  onSuccess: () => {
    message.value = "";
    sent.value = true;
  },
});
const commandError = computed(
  () => microphone.error.value || interrupt.error.value || restart.error.value || send.error.value,
);
</script>

<template>
  <header class="page-heading">
    <div>
      <h1 tabindex="-1">Conversation</h1>
      <p class="muted">Talk with Reachy or send a message below.</p>
    </div>
    <button class="button quiet" :disabled="restart.isPending.value" @click="restart.mutate()">
      {{ restart.isPending.value ? "Reconnecting…" : "Reconnect" }}
    </button>
  </header>
  <div v-if="error || (snapshot && !ready)" class="notice error" role="alert">
    <p>{{ error || snapshot?.connectionError || "The conversation is reconnecting." }}</p>
    <RouterLink to="/settings">Open connection settings</RouterLink>
  </div>
  <section class="conversation-stage panel" :class="{ speaking: snapshot?.playing && ready }">
    <div class="personality-row">
      <img
        :src="profileAvatar(snapshot?.profile ?? 'profiles/builtin-default')"
        alt=""
        width="48"
        height="48"
      />
      <div>
        <RouterLink
          v-if="snapshot"
          class="current-profile"
          :to="`/profiles/${snapshot.profile.split('/')[1]}`"
          >{{ profileLabel(snapshot.profile) }}</RouterLink
        >
        <p class="muted voice-label">{{ snapshot ? `Voice: ${snapshot.voice}` : "Loading personality…" }}</p>
      </div>
      <RouterLink class="button quiet change-profile" to="/profiles">Change personality</RouterLink>
    </div>
    <div class="conversation-activity">
      <div
        class="audio-indicator"
        :class="{ paused: snapshot?.muted && !snapshot?.playing }"
        aria-hidden="true"
      >
        <span /><span /><span /><span /><span />
      </div>
      <h2>{{ heading }}</h2>
      <p class="live-status" role="status">{{ status }}</p>
      <div class="actions conversation-controls">
        <button
          class="button primary"
          :disabled="!snapshot || microphone.isPending.value"
          :aria-pressed="snapshot?.muted ?? false"
          @click="microphone.mutate(!snapshot?.muted)"
        >
          {{ snapshot?.muted ? "Unmute microphone" : "Mute microphone" }}
        </button>
        <button class="button" :disabled="!snapshot || interrupt.isPending.value" @click="interrupt.mutate()">
          Stop speaking
        </button>
      </div>
      <p class="stage-hint muted">
        {{
          snapshot?.muted
            ? "You can still type while the microphone is muted."
            : "Uses the microphone and speaker on your robot."
        }}
      </p>
    </div>
  </section>
  <p v-if="commandError" class="notice error" role="alert">{{ describeError(commandError) }}</p>
  <form class="message-composer panel" @submit.prevent="message.trim() && send.mutate()">
    <label class="field" for="message">Send a message</label>
    <textarea
      id="message"
      v-model="message"
      aria-label="Your message"
      rows="3"
      placeholder="Type a message to Reachy…"
      :disabled="send.isPending.value"
      @input="sent = false"
      @keydown.ctrl.enter.prevent="ready && message.trim() && !send.isPending.value && send.mutate()"
      @keydown.meta.enter.prevent="ready && message.trim() && !send.isPending.value && send.mutate()"
    />
    <div class="composer-footer">
      <p class="muted" role="status">
        {{
          sent
            ? "Message sent. Reachy will answer out loud."
            : "Reachy answers out loud. Ctrl / ⌘ + Enter to send."
        }}
      </p>
      <button
        class="button primary"
        type="submit"
        :disabled="!ready || !message.trim() || send.isPending.value"
      >
        {{ send.isPending.value ? "Sending…" : "Send" }}
      </button>
    </div>
  </form>
</template>

<style scoped>
.conversation-stage {
  padding: 0;
  overflow: hidden;
}
.personality-row {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  padding: 1rem 1.25rem;
  border-bottom: 1px solid var(--border);
}
.personality-row img {
  border-radius: 0.5rem;
  background: var(--soft);
}
.current-profile {
  color: var(--text);
  font-weight: 600;
  text-decoration: none;
}
.current-profile:hover {
  text-decoration: underline;
}
.voice-label {
  margin: 0.15rem 0 0;
  font-size: 0.8rem;
}
.change-profile {
  margin-left: auto;
}
.conversation-activity {
  padding: 1.5rem 1rem;
  text-align: center;
}
.conversation-activity h2 {
  margin: 0.8rem 0 0.25rem;
  font-size: 1.4rem;
}
.live-status {
  font-size: 0.875rem;
  color: var(--muted);
  margin-bottom: 1.25rem;
}
.audio-indicator {
  height: 38px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  color: var(--accent);
}
.audio-indicator span {
  width: 5px;
  height: 14px;
  border-radius: 4px;
  background: currentColor;
}
.audio-indicator span:nth-child(2n) {
  height: 24px;
}
.audio-indicator span:nth-child(3) {
  height: 36px;
}
.audio-indicator.paused {
  color: var(--muted);
}
.audio-indicator.paused span {
  height: 5px;
}
.speaking .audio-indicator span {
  animation: speaking 0.8s ease-in-out infinite alternate;
}
.speaking .audio-indicator span:nth-child(2n) {
  animation-delay: -0.4s;
}
.conversation-controls {
  justify-content: center;
}
.stage-hint {
  font-size: 0.8rem;
  margin: 0.9rem 0 0;
}
.message-composer {
  margin-top: 1rem;
}
.message-composer .field {
  margin-bottom: 0.75rem;
}
.composer-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-top: 0.8rem;
}
.composer-footer p {
  margin: 0;
  font-size: 0.8rem;
}
@keyframes speaking {
  to {
    transform: scaleY(0.4);
  }
}
@media (max-width: 520px) {
  .personality-row {
    flex-wrap: wrap;
    padding: 1rem;
  }
  .change-profile {
    width: 100%;
    margin: 0;
  }
  .composer-footer {
    align-items: flex-end;
  }
  .composer-footer p {
    max-width: 70%;
  }
}
</style>
