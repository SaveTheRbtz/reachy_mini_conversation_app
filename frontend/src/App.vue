<script setup lang="ts">
import { computed, onErrorCaptured, ref } from "vue";
import { RouterLink, RouterView, useRoute } from "vue-router";
import { provideConversation } from "./conversation.ts";
import { profileAvatar } from "./presentation.ts";
import { Conversation_ConnectionState as ConnectionState } from "./gen/reachy/conversation/v1/api_pb.ts";

const { snapshot, connected } = provideConversation();
const route = useRoute();
const renderError = ref(false);
const status = computed(() =>
  !connected.value
    ? "Connecting to robot"
    : snapshot.value?.connectionState === ConnectionState.CONNECTED
      ? "Ready to talk"
      : snapshot.value?.connectionState === ConnectionState.WAITING_FOR_CONFIG
        ? "API key needed"
        : "Connecting to OpenAI",
);
onErrorCaptured((error) => {
  console.error("UI failed", error);
  renderError.value = true;
  return false;
});
</script>

<template>
  <a class="skip-link" href="#main">Skip to content</a>
  <div class="app-shell">
    <header class="app-header">
      <RouterLink class="brand" to="/" aria-label="Reachy Mini home">
        <img :src="profileAvatar('profiles/builtin-default')" alt="" width="44" height="44" />
        <span>Reachy Mini</span>
      </RouterLink>
      <nav aria-label="Main navigation">
        <RouterLink to="/" exact-active-class="active">Conversation</RouterLink>
        <RouterLink to="/profiles" :class="{ active: route.path.startsWith('/profiles') }"
          >Personalities</RouterLink
        >
        <RouterLink to="/settings" active-class="active">Settings</RouterLink>
      </nav>
      <div class="robot-status" role="status">
        <span
          class="status-dot"
          :class="{ online: connected && snapshot?.connectionState === ConnectionState.CONNECTED }"
        />{{ status }}
      </div>
    </header>
    <main id="main" tabindex="-1">
      <section v-if="renderError" class="panel notice error" role="alert">
        <h1 tabindex="-1">Something went wrong</h1>
        <p>Reload the app to reconnect to your robot.</p>
        <a class="button primary" href="/">Reload app</a>
      </section>
      <RouterView v-else :key="route.path" />
    </main>
  </div>
</template>
