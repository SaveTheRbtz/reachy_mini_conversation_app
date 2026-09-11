import { nextTick } from "vue";
import { createRouter, createWebHistory } from "vue-router";
import ConversationView from "./views/ConversationView.vue";
import ProfilesView from "./views/ProfilesView.vue";
import ProfileEditorView from "./views/ProfileEditorView.vue";
import SettingsView from "./views/SettingsView.vue";
import NotFoundView from "./views/NotFoundView.vue";

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", component: ConversationView, meta: { title: "Conversation" } },
    { path: "/profiles", component: ProfilesView, meta: { title: "Personalities" } },
    { path: "/profiles/new", component: ProfileEditorView, meta: { title: "Create personality" } },
    { path: "/profiles/:id", component: ProfileEditorView, props: true, meta: { title: "Personality" } },
    { path: "/settings", component: SettingsView, meta: { title: "Settings" } },
    { path: "/:pathMatch(.*)*", component: NotFoundView, meta: { title: "Page not found" } },
  ],
  scrollBehavior: () => ({ top: 0 }),
});

router.afterEach(async (to, _from, failure) => {
  if (failure) return;
  document.title = `${String(to.meta.title)} · Reachy Mini`;
  await nextTick();
  document.querySelector<HTMLElement>("main h1")?.focus({ preventScroll: true });
});
