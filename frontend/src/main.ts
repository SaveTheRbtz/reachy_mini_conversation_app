import { createApp } from "vue";
import { VueQueryPlugin } from "@tanstack/vue-query";
import App from "./App.vue";
import { router } from "./router.ts";
import { queryClient } from "./queries.ts";
import "./styles.css";

let theme = new URLSearchParams(location.search).get("theme");
try {
  if (theme === null) theme = sessionStorage.getItem("reachy.ui.theme");
  else if (theme === "light" || theme === "dark") sessionStorage.setItem("reachy.ui.theme", theme);
  else sessionStorage.removeItem("reachy.ui.theme");
} catch (error) {
  console.warn("Could not remember the dashboard theme", error);
}
if (theme === "dark" || theme === "light") document.documentElement.dataset.theme = theme;

createApp(App).use(router).use(VueQueryPlugin, { queryClient }).mount("#app");
