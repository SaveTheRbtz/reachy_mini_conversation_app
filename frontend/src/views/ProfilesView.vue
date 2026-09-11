<script setup lang="ts">
import { computed, ref } from "vue";
import { RouterLink, useRouter } from "vue-router";
import { useMutation, useQuery } from "@tanstack/vue-query";
import { api, describeError } from "../api.ts";
import { useConversation } from "../conversation.ts";
import { profileAvatar, profileLabel } from "../presentation.ts";
import { profilesQuery, settingsQuery } from "../queries.ts";

const router = useRouter();
const { snapshot } = useConversation();
const { data: profiles, isPending, error, refetch } = useQuery(profilesQuery);
const { data: settings, error: settingsError, refetch: reloadSettings } = useQuery(settingsQuery);
const search = ref("");
const locked = computed(() => settings.value?.lockedProfile);
const visibleProfiles = computed(() =>
  (profiles.value ?? [])
    .filter((profile) => profileLabel(profile.name).toLowerCase().includes(search.value.trim().toLowerCase()))
    .sort(
      (left, right) =>
        Number(right.name === snapshot.value?.profile) - Number(left.name === snapshot.value?.profile) ||
        profileLabel(left.name).localeCompare(profileLabel(right.name)),
    ),
);
const {
  mutate: apply,
  isPending: applying,
  error: applyError,
  variables: applyingProfile,
} = useMutation({
  mutationFn: (profile: string) => api.restartConversation({ name: "conversation", profile }),
});
</script>

<template>
  <section>
    <header class="page-heading">
      <div>
        <h1 tabindex="-1">Personalities</h1>
        <p class="muted">Choose a personality or customize its instructions and tools.</p>
      </div>
      <RouterLink v-if="settings && !locked" to="/profiles/new" class="button primary">
        Create personality
      </RouterLink>
    </header>

    <p v-if="locked" class="notice">
      Personality selection and editing are locked to {{ profileLabel(locked) }}.
    </p>
    <p v-if="isPending" role="status" class="muted">Loading personalities…</p>
    <div v-if="error || settingsError" role="alert" class="notice error">
      <p>{{ describeError(error || settingsError) }}</p>
      <button
        class="button quiet"
        @click="
          refetch();
          reloadSettings();
        "
      >
        Try again
      </button>
    </div>
    <p v-if="applyError" role="alert" class="notice error">
      Could not switch personality: {{ describeError(applyError) }}
    </p>

    <label v-if="profiles" class="field profile-search">
      <span>Search personalities</span>
      <input v-model="search" type="search" placeholder="Search by name" />
    </label>
    <p v-if="profiles && !visibleProfiles.length" role="status" class="muted">
      No personalities match your search.
    </p>

    <div class="profile-grid">
      <article v-for="profile in visibleProfiles" :key="profile.name" class="profile-card">
        <RouterLink :to="`/profiles/${encodeURIComponent(profile.name.slice(9))}`" class="profile-link">
          <img :src="profileAvatar(profile.name)" alt="" loading="lazy" />
          <h2>{{ profileLabel(profile.name) }}</h2>
        </RouterLink>
        <div class="profile-badges">
          <span v-if="snapshot?.profile === profile.name" class="badge">Active</span>
          <span v-if="settings?.startupProfile === profile.name" class="badge">Startup default</span>
          <span class="muted">{{ profile.editable ? "Custom" : "Built in" }}</span>
        </div>
        <p class="muted profile-description">
          {{ profile.voice || "Default voice" }} · {{ profile.effectiveToolIds.length }} tools
        </p>
        <div class="actions">
          <button
            class="button primary"
            :disabled="applying || !settings || Boolean(locked && locked !== profile.name)"
            @click="
              snapshot?.profile === profile.name
                ? router.push('/')
                : apply(profile.name, { onSuccess: () => router.push('/') })
            "
          >
            {{ applying && applyingProfile === profile.name ? "Switching…" : "Use personality" }}
          </button>
          <RouterLink :to="`/profiles/${encodeURIComponent(profile.name.slice(9))}`" class="button quiet">
            Configure
          </RouterLink>
        </div>
      </article>
    </div>
  </section>
</template>

<style scoped>
.profile-search {
  max-width: 28rem;
  margin-bottom: 1.25rem;
}
.profile-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.75rem;
}
.profile-card {
  display: flex;
  flex-direction: column;
  padding: 1rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 0.75rem;
}
.profile-link {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  color: inherit;
  text-decoration: none;
}
.profile-link img {
  width: 40px;
  height: 40px;
  margin: 0;
  object-fit: cover;
  border-radius: 50%;
}
.profile-link h2 {
  margin: 0;
  font-size: 1rem;
  line-height: 1.3;
  overflow-wrap: anywhere;
}
.profile-badges {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin: 0.5rem 0 0.4rem 3.25rem;
  font-size: 0.8rem;
}
.profile-description {
  margin: 0 0 0.5rem 3.25rem;
  font-size: 0.875rem;
}
.profile-card .actions {
  margin-top: auto;
  padding-top: 0.5rem;
}
@media (max-width: 700px) {
  .profile-grid {
    grid-template-columns: 1fr;
  }
}
</style>
