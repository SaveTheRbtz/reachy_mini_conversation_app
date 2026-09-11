<script setup lang="ts">
import { computed, reactive, ref, shallowRef, watch } from "vue";
import { useMutation, useQuery, useQueryClient } from "@tanstack/vue-query";
import { api, describeError } from "../api.ts";
import Dialog from "../components/Dialog.vue";
import type { Settings } from "../gen/reachy/conversation/v1/api_pb.ts";
import { profileLabel } from "../presentation.ts";
import { capabilitiesQuery, profilesQuery, settingsQuery } from "../queries.ts";
import { useUnsavedChanges } from "../useUnsavedChanges.ts";

const queryClient = useQueryClient();
const { data: settings, error: settingsError, refetch: reloadSettings } = useQuery(settingsQuery);
const { data: profiles, error: profilesError, refetch: reloadProfiles } = useQuery(profilesQuery);
const {
  data: capabilities,
  error: capabilitiesError,
  refetch: reloadCapabilities,
} = useQuery(capabilitiesQuery);
const baseline = shallowRef<Settings>();
const draft = reactive({ startupProfile: "", voiceOverride: "" });
const apiKey = ref("");
const status = ref("");
const actionError = ref("");
const needsRestart = ref(false);
const preferencesDirty = computed(() =>
  Boolean(
    baseline.value &&
    (draft.startupProfile !== baseline.value.startupProfile ||
      draft.voiceOverride !== baseline.value.voiceOverride),
  ),
);
const dirty = computed(() => preferencesDirty.value || Boolean(apiKey.value));
const loadError = computed(() => settingsError.value || profilesError.value || capabilitiesError.value);
watch(
  settings,
  (loaded) => {
    if (loaded && !preferencesDirty.value) {
      baseline.value = loaded;
      Object.assign(draft, { startupProfile: loaded.startupProfile, voiceOverride: loaded.voiceOverride });
    }
  },
  { immediate: true },
);

const restart = useMutation({
  mutationFn: () => api.restartConversation({ name: "conversation" }),
  onSuccess: () => {
    needsRestart.value = false;
    actionError.value = "";
    status.value = "Saved. The conversation is restarting.";
  },
  onError: (error) => {
    actionError.value = `Saved, but could not restart: ${describeError(error)}`;
  },
});
const preferences = useMutation({
  mutationFn: () =>
    api.updateSettings({
      settings: { name: "settings", ...draft },
      updateMask: {
        paths: [
          ...(draft.startupProfile !== baseline.value?.startupProfile ? ["startup_profile"] : []),
          ...(draft.voiceOverride !== baseline.value?.voiceOverride ? ["voice_override"] : []),
        ],
      },
    }),
  onMutate: () => {
    actionError.value = "";
    status.value = "";
  },
  onSuccess: (saved) => {
    const voiceChanged = saved.voiceOverride !== baseline.value?.voiceOverride;
    baseline.value = saved;
    Object.assign(draft, { startupProfile: saved.startupProfile, voiceOverride: saved.voiceOverride });
    queryClient.setQueryData(settingsQuery.queryKey, saved);
    status.value = "Preferences saved.";
    if (voiceChanged) {
      needsRestart.value = true;
      restart.mutate();
    }
  },
  onError: (error) => {
    actionError.value = `Could not save preferences: ${describeError(error)}`;
  },
});
const credential = useMutation({
  mutationFn: () => api.setSettingsApiKey({ name: "settings", apiKey: apiKey.value.trim() }),
  onMutate: () => {
    actionError.value = "";
    status.value = "";
  },
  onSuccess: (saved) => {
    apiKey.value = "";
    queryClient.setQueryData(settingsQuery.queryKey, saved);
    status.value = "API key saved.";
    needsRestart.value = true;
    restart.mutate();
  },
  onError: (error) => {
    actionError.value = `Could not save API key: ${describeError(error)}`;
  },
});
const busy = computed(
  () => preferences.isPending.value || credential.isPending.value || restart.isPending.value,
);
const { confirming, decide } = useUnsavedChanges(dirty, busy);
</script>

<template>
  <header class="page-heading">
    <div>
      <h1 tabindex="-1">Settings</h1>
      <p class="muted">Manage your OpenAI connection and conversation defaults.</p>
    </div>
  </header>
  <div v-if="loadError" class="notice error" role="alert">
    <p>Could not load settings: {{ describeError(loadError) }}</p>
    <button
      class="button quiet"
      @click="
        reloadSettings();
        reloadProfiles();
        reloadCapabilities();
      "
    >
      Try again
    </button>
  </div>
  <p v-else-if="!settings || !profiles || !capabilities" role="status" class="muted">Loading settings…</p>
  <div v-if="settings && profiles && capabilities" class="settings-layout">
    <form class="panel settings-section" @submit.prevent="preferences.mutate()">
      <h2>Conversation preferences</h2>
      <p class="muted">Set a startup personality and an optional voice override.</p>
      <fieldset class="form-grid" :disabled="busy">
        <label class="field">
          <span id="default-profile-label">Default personality</span>
          <select
            v-model="draft.startupProfile"
            :disabled="Boolean(settings.lockedProfile)"
            aria-labelledby="default-profile-label"
            aria-describedby="default-profile-help"
          >
            <option v-for="profile in profiles" :key="profile.name" :value="profile.name">
              {{ profileLabel(profile.name) }}
            </option>
          </select>
          <small id="default-profile-help" class="muted">{{
            settings.lockedProfile
              ? "Locked by the administrator."
              : "Used on app startup. Your current conversation keeps its personality."
          }}</small>
        </label>
        <label class="field">
          <span id="settings-voice-label">Voice</span>
          <select
            v-model="draft.voiceOverride"
            aria-labelledby="settings-voice-label"
            aria-describedby="settings-voice-help"
          >
            <option value="">Use personality default</option>
            <option v-for="voice in capabilities.availableVoices" :key="voice" :value="voice">
              {{ voice }}
            </option>
          </select>
          <small id="settings-voice-help" class="muted">Changing voice restarts the conversation.</small>
        </label>
        <div class="actions">
          <button class="button primary" type="submit" :disabled="!preferencesDirty">
            {{ preferences.isPending.value ? "Saving…" : "Save preferences" }}</button
          ><span v-if="preferencesDirty" class="muted">Unsaved changes</span>
        </div>
      </fieldset>
    </form>
    <form class="panel settings-section" @submit.prevent="credential.mutate()">
      <div class="section-heading">
        <h2>OpenAI connection</h2>
        <span class="badge">{{ settings.apiKeyConfigured ? "Key configured" : "Key needed" }}</span>
      </div>
      <p class="muted">Your key is stored on the robot and is never shown here.</p>
      <fieldset class="form-grid" :disabled="busy">
        <label class="field">
          <span id="api-key-label">OpenAI API key</span>
          <input
            v-model="apiKey"
            type="password"
            required
            autocomplete="new-password"
            spellcheck="false"
            aria-labelledby="api-key-label"
            aria-describedby="api-key-help"
            :placeholder="settings.apiKeyConfigured ? 'Enter a replacement key' : 'sk-…'"
          />
          <small id="api-key-help" class="muted">Saving a key reconnects Reachy to OpenAI.</small>
        </label>
        <div class="actions">
          <button class="button primary" type="submit" :disabled="!apiKey.trim()">
            {{ credential.isPending.value ? "Saving…" : "Save API key" }}
          </button>
        </div>
      </fieldset>
    </form>
  </div>
  <p v-if="status" class="notice success" role="status">{{ status }}</p>
  <p v-if="actionError" class="notice error" role="alert">{{ actionError }}</p>
  <button v-if="needsRestart" class="button" :disabled="busy" @click="restart.mutate()">
    {{ restart.isPending.value ? "Restarting…" : "Retry restart" }}
  </button>
  <Dialog
    v-if="confirming"
    title="Discard unsaved changes?"
    confirm-label="Discard changes"
    @confirm="decide(true)"
    @cancel="decide(false)"
  >
    Your changes to settings have not been saved.
  </Dialog>
</template>

<style scoped>
.settings-layout {
  display: grid;
  gap: 1.25rem;
  max-width: 760px;
}
.settings-section {
  padding: 1.75rem;
}
.settings-section h2 {
  font-size: 1.1rem;
  margin: 0;
}
.settings-section > p {
  font-size: 0.85rem;
  margin: 0.6rem 0 1.5rem;
}
.settings-section fieldset {
  border: 0;
  padding: 0;
  margin: 0;
  min-width: 0;
}
.section-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.75rem;
}
@media (max-width: 760px) {
  .settings-section {
    padding: 1.25rem;
  }
}
</style>
