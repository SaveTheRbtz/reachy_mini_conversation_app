<script setup lang="ts">
import { computed, reactive, ref, shallowRef, watch } from "vue";
import { RouterLink, useRouter } from "vue-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/vue-query";
import { api, describeError } from "../api.ts";
import Dialog from "../components/Dialog.vue";
import { useConversation } from "../conversation.ts";
import type { Profile } from "../gen/reachy/conversation/v1/api_pb.ts";
import { profileLabel, slug } from "../presentation.ts";
import { capabilitiesQuery, profileQuery, profilesQuery, settingsQuery } from "../queries.ts";
import { useUnsavedChanges } from "../useUnsavedChanges.ts";

const props = defineProps<{ id?: string }>();
const router = useRouter();
const queryClient = useQueryClient();
const { snapshot } = useConversation();
const creating = computed(() => !props.id);
const name = computed(() => (props.id ? `profiles/${props.id}` : ""));
const {
  data: profile,
  error: profileError,
  refetch: reloadProfile,
} = useQuery(computed(() => profileQuery(name.value)));
const {
  data: defaults,
  error: defaultsError,
  refetch: reloadDefaults,
} = useQuery(computed(() => profileQuery(creating.value ? "profiles/builtin-default" : "")));
const { data: settings, error: settingsError, refetch: reloadSettings } = useQuery(settingsQuery);
const {
  data: capabilities,
  error: capabilitiesError,
  refetch: reloadCapabilities,
} = useQuery(capabilitiesQuery);
const baseline = shallowRef<Profile>();
const draft = reactive({ label: "", instructions: "", greeting: "", voice: "", inheritTools: true });
const selectedTools = ref<string[]>([]);
const status = ref("");
const actionError = ref("");
const needsRestart = ref(false);
const showDelete = ref(false);
const locked = computed(() => !settings.value || Boolean(settings.value.lockedProfile));
const editable = computed(() => creating.value || Boolean(baseline.value?.editable));
const ready = computed(() =>
  Boolean(settings.value && capabilities.value && (creating.value ? defaults.value : profile.value)),
);
const loadError = computed(
  () => profileError.value || defaultsError.value || settingsError.value || capabilitiesError.value,
);
const defaultTools = computed(() => baseline.value?.defaultToolIds ?? defaults.value?.defaultToolIds ?? []);
const toolChoices = computed(() =>
  [
    ...(capabilities.value?.availableTools ?? []).map((tool) => ({ ...tool, available: true })),
    ...[...new Set([...defaultTools.value, ...selectedTools.value])]
      .filter((id) => !capabilities.value?.availableTools.some((tool) => tool.id === id))
      .map((id) => ({
        id,
        description: "Unavailable on this Reachy. You can remove this selection.",
        available: false,
      })),
  ].map((tool) => ({
    ...tool,
    label: tool.id.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase()),
  })),
);
const changedPaths = computed(() => {
  const saved = baseline.value;
  if (!saved) return [];
  const paths: string[] = (["instructions", "greeting", "voice"] as const).filter(
    (field) => draft[field] !== saved[field],
  );
  const savedTools = saved.toolOverride?.toolIds ?? [];
  if (
    draft.inheritTools !== !saved.toolOverride ||
    (!draft.inheritTools &&
      (selectedTools.value.length !== savedTools.length ||
        selectedTools.value.some((id) => !savedTools.includes(id))))
  )
    paths.push("tool_override");
  return paths;
});
const dirty = computed(() =>
  creating.value && !baseline.value
    ? Boolean(draft.label || draft.instructions || draft.greeting || draft.voice || !draft.inheritTools)
    : changedPaths.value.length > 0,
);
const canDelete = computed(
  () =>
    baseline.value?.editable &&
    snapshot.value &&
    !locked.value &&
    name.value !== snapshot.value?.profile &&
    name.value !== settings.value?.startupProfile,
);

function reset(saved: Profile) {
  baseline.value = saved;
  Object.assign(draft, {
    label: saved.displayName,
    instructions: saved.instructions,
    greeting: saved.greeting,
    voice: saved.voice,
    inheritTools: !saved.toolOverride,
  });
  selectedTools.value = [...(saved.toolOverride?.toolIds ?? saved.defaultToolIds)];
}

watch(
  profile,
  (loaded) => {
    if (loaded && (loaded.name !== baseline.value?.name || !dirty.value)) reset(loaded);
  },
  { immediate: true },
);
const { mutate: save, isPending: saving } = useMutation({
  mutationFn: async () => {
    status.value = "";
    actionError.value = "";
    const content = {
      instructions: draft.instructions,
      greeting: draft.greeting,
      voice: draft.voice,
      toolOverride: draft.inheritTools ? undefined : { toolIds: [...selectedTools.value] },
    };
    const saved = creating.value
      ? await api.createProfile({
          profileId: `user-${slug(draft.label)}`,
          profile: { ...content, defaultToolIds: [...defaultTools.value] },
        })
      : await api.updateProfile({
          profile: { name: name.value, ...content },
          updateMask: { paths: changedPaths.value },
        });
    const wasNew = creating.value;
    reset(saved);
    queryClient.setQueryData(profileQuery(saved.name).queryKey, saved);
    void queryClient.invalidateQueries({ queryKey: profilesQuery.queryKey });
    if (wasNew) {
      await router.replace(`/profiles/${encodeURIComponent(saved.name.slice(9))}`);
    }
    status.value = "Personality saved.";
    if (saved.name === snapshot.value?.profile) {
      needsRestart.value = true;
      try {
        await api.restartConversation({ name: "conversation" });
        needsRestart.value = false;
        status.value = "Personality saved. The conversation is restarting with your changes.";
      } catch (error) {
        console.warn("Could not restart after saving personality", error);
        actionError.value = `Personality saved, but could not restart: ${describeError(error)}`;
      }
    }
  },
  onError: (error) => {
    actionError.value = `Could not save personality: ${describeError(error)}`;
  },
});
const { mutate: retryRestart, isPending: restarting } = useMutation({
  mutationFn: () => api.restartConversation({ name: "conversation" }),
  onSuccess: () => {
    needsRestart.value = false;
    actionError.value = "";
    status.value = "The conversation is restarting with your saved changes.";
  },
  onError: (error) => {
    actionError.value = `Personality saved, but could not restart: ${describeError(error)}`;
  },
});
const { mutate: apply, isPending: applying } = useMutation({
  mutationFn: () => api.restartConversation({ name: "conversation", profile: name.value }),
  onError: (error) => {
    actionError.value = `Could not use personality: ${describeError(error)}`;
  },
});
const { mutate: remove, isPending: deleting } = useMutation({
  mutationFn: () => api.deleteProfile({ name: name.value }),
  onSuccess: () => {
    showDelete.value = false;
    queryClient.removeQueries({ queryKey: profileQuery(name.value).queryKey });
    void queryClient.invalidateQueries({ queryKey: profilesQuery.queryKey });
    void router.push("/profiles");
  },
  onError: (error) => {
    showDelete.value = false;
    actionError.value = `Could not delete personality: ${describeError(error)}`;
  },
});
const busy = computed(() => saving.value || restarting.value || applying.value || deleting.value);
const { confirming, decide } = useUnsavedChanges(dirty, busy);

function setAllTools(enabled: boolean) {
  selectedTools.value = enabled
    ? toolChoices.value.filter((tool) => tool.available).map((tool) => tool.id)
    : [];
}
</script>

<template>
  <section>
    <RouterLink to="/profiles" class="back-link">← All personalities</RouterLink>
    <header class="page-heading">
      <div>
        <h1 tabindex="-1">{{ creating ? "Create personality" : profileLabel(name) }}</h1>
        <p class="muted">Instructions, voice, and tool access.</p>
      </div>
      <span v-if="snapshot?.profile === name" class="badge">Active personality</span>
    </header>

    <div v-if="loadError" role="alert" class="notice error">
      <p>Could not load personality: {{ describeError(loadError) }}</p>
      <button
        class="button quiet"
        @click="
          creating ? reloadDefaults() : reloadProfile();
          reloadSettings();
          reloadCapabilities();
        "
      >
        Try again
      </button>
    </div>
    <p v-else-if="!ready" role="status" class="muted">Loading personality…</p>
    <form v-if="ready" @submit.prevent="save()" :aria-busy="busy">
      <p v-if="locked" class="notice">Personality editing is locked by the administrator.</p>
      <p v-else-if="!editable" class="notice">
        This built-in personality’s writing and voice are read-only. You can customize its tool access.
      </p>
      <fieldset class="panel character-fields" :disabled="locked || busy || !editable">
        <legend>Character</legend>
        <label v-if="creating" class="field">
          <span id="name-label">Name</span>
          <input
            v-model.trim="draft.label"
            required
            maxlength="58"
            pattern="[A-Za-z].*"
            title="Begin the name with a letter."
            autocomplete="off"
            aria-labelledby="name-label"
            aria-describedby="name-help"
          />
          <small id="name-help" class="muted"
            >Use a name beginning with a letter. Names cannot be changed later.</small
          >
        </label>
        <label class="field">
          <span>Instructions</span>
          <textarea
            v-model="draft.instructions"
            required
            rows="9"
            placeholder="Describe how Reachy should think and respond…"
          />
        </label>
        <label class="field">
          <span>Greeting</span>
          <textarea v-model="draft.greeting" rows="3" placeholder="An optional opening line" />
        </label>
        <label class="field">
          <span id="voice-label">Voice</span>
          <select v-model="draft.voice" aria-labelledby="voice-label" aria-describedby="voice-help">
            <option value="">App default ({{ capabilities?.defaultVoice }})</option>
            <option v-for="voice in capabilities?.availableVoices" :key="voice" :value="voice">
              {{ voice }}
            </option>
          </select>
          <small id="voice-help" class="muted">A voice selected in Settings takes precedence.</small>
        </label>
      </fieldset>

      <fieldset class="panel tool-access" :disabled="locked || busy">
        <legend>Tool access</legend>
        <label class="check-row inherit-choice">
          <input
            v-model="draft.inheritTools"
            type="checkbox"
            aria-labelledby="defaults-label"
            aria-describedby="defaults-help"
            @change="selectedTools = [...defaultTools]"
          />
          <span
            ><strong id="defaults-label">Use profile defaults</strong
            ><small id="defaults-help" class="muted"
              >{{ defaultTools.length }} tools enabled by default</small
            ></span
          >
        </label>
        <div v-if="!draft.inheritTools" class="actions tool-toolbar">
          <span class="muted">{{ selectedTools.length }} tools enabled</span>
          <button type="button" class="button quiet" @click="setAllTools(true)">Enable all</button>
          <button type="button" class="button quiet" @click="setAllTools(false)">Disable all</button>
        </div>
        <div class="tool-grid">
          <div v-for="tool in toolChoices" :key="tool.id">
            <label class="check-row">
              <input
                v-if="draft.inheritTools"
                type="checkbox"
                :checked="defaultTools.includes(tool.id)"
                :aria-labelledby="`tool-${tool.id}-label`"
                :aria-describedby="tool.available ? undefined : `tool-${tool.id}-unavailable`"
                disabled
              />
              <input
                v-else
                v-model="selectedTools"
                type="checkbox"
                :value="tool.id"
                :aria-labelledby="`tool-${tool.id}-label`"
                :aria-describedby="tool.available ? undefined : `tool-${tool.id}-unavailable`"
              />
              <span>
                <strong :id="`tool-${tool.id}-label`">{{ tool.label }}</strong>
                <small v-if="!tool.available" :id="`tool-${tool.id}-unavailable`" class="unavailable"
                  >Unavailable</small
                >
              </span>
            </label>
            <details class="tool-details">
              <summary :aria-label="`About ${tool.label}`">About this tool</summary>
              <p class="muted">{{ tool.description }}</p>
            </details>
          </div>
        </div>
        <p v-if="!draft.inheritTools && !selectedTools.length" class="muted">
          All tools are disabled for this personality.
        </p>
      </fieldset>

      <button
        v-if="canDelete"
        type="button"
        class="button danger delete-button"
        :disabled="busy || dirty"
        @click="showDelete = true"
      >
        Delete personality
      </button>

      <div class="editor-footer">
        <p v-if="status" role="status" class="notice success">{{ status }}</p>
        <p v-if="actionError" role="alert" class="notice error">{{ actionError }}</p>
        <div class="actions editor-actions">
          <button type="submit" class="button primary" :disabled="busy || locked || (!creating && !dirty)">
            {{ saving ? "Saving…" : creating ? "Create personality" : "Save changes" }}
          </button>
          <button
            v-if="!creating"
            type="button"
            class="button"
            :disabled="busy || dirty || Boolean(settings?.lockedProfile && settings.lockedProfile !== name)"
            @click="
              snapshot?.profile === name && !needsRestart
                ? router.push('/')
                : apply(undefined, { onSuccess: () => router.push('/') })
            "
          >
            {{ applying ? "Switching…" : "Use personality" }}
          </button>
          <button
            v-if="needsRestart && snapshot?.profile === name"
            type="button"
            class="button"
            :disabled="busy || dirty"
            @click="retryRestart()"
          >
            {{ restarting ? "Restarting…" : "Retry restart" }}
          </button>
          <span v-if="dirty" class="muted" role="status">Unsaved changes</span>
        </div>
      </div>
    </form>

    <Dialog
      v-if="confirming"
      title="Discard unsaved changes?"
      confirm-label="Discard changes"
      @confirm="decide(true)"
      @cancel="decide(false)"
    >
      Your changes to this personality have not been saved.
    </Dialog>
    <Dialog
      v-if="showDelete"
      title="Delete personality?"
      confirm-label="Delete personality"
      danger
      :busy="deleting"
      @confirm="remove()"
      @cancel="showDelete = false"
    >
      {{ profileLabel(name) }} will be permanently removed.
    </Dialog>
  </section>
</template>

<style scoped>
.back-link {
  display: inline-block;
  margin-bottom: 1rem;
}
fieldset {
  margin: 0 0 1.5rem;
  min-width: 0;
}
legend {
  padding: 0 0.5rem;
  font-size: 1.1rem;
  font-weight: 650;
}
.tool-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr));
  gap: 1.25rem;
}
.check-row {
  display: flex;
  gap: 0.8rem;
  align-items: flex-start;
}
.check-row input {
  flex: none;
  margin-top: 0.25rem;
}
.check-row small {
  display: block;
  margin-top: 0.3rem;
  line-height: 1.45;
}
.inherit-choice {
  margin-bottom: 1.75rem;
}
.character-fields .field + .field {
  margin-top: 1.25rem;
}
.tool-toolbar {
  margin: -0.75rem 0 1.25rem;
}
.tool-details {
  margin: 0.4rem 0 0 1.95rem;
  font-size: 0.8rem;
}
.tool-details summary {
  color: var(--accent);
  cursor: pointer;
}
.tool-details p {
  margin-top: 0.6rem;
  white-space: pre-line;
  overflow-wrap: anywhere;
}
.unavailable {
  color: var(--danger);
}
.editor-footer {
  position: sticky;
  bottom: 0;
  z-index: 2;
  padding: 1rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 0.75rem;
  box-shadow: 0 -4px 18px #102d170d;
}
.editor-footer .notice {
  margin: 0 0 0.75rem;
}
.editor-actions {
  align-items: center;
}
.delete-button {
  margin-bottom: 1rem;
}
</style>
