/** Tools view: per-personality access to the fixed tool catalog. */

import type { RouterContext } from "../router.ts";
import { ROUTES } from "../constants.ts";
import { h } from "../ui.ts";
import { buildProfileToolsSection } from "../components/profile-tools.ts";

export async function mountToolsView({ outlet, signal, searchParams, setLeaveGuard, replaceRoute }: RouterContext) {
  const fromPersonalities = searchParams.get("from") === "personalities";
  const profileToolsSection = buildProfileToolsSection({
    signal,
    initialProfile: searchParams.get("profile"),
    onProfileChanged(profile) {
      const params = new URLSearchParams({ profile });
      if (fromPersonalities) params.set("from", "personalities");
      replaceRoute(`${ROUTES.TOOLS}?${params}`);
    },
  });
  setLeaveGuard({
    shouldBlock: profileToolsSection.hasUnsavedChanges,
    confirm: profileToolsSection.confirmDiscard,
  });
  const view = h(
    "section",
    { class: "view view--tools" },
    h(
      "header",
      { class: "view-header" },
      h("h1", { class: "view-title" }, "Tools"),
      h("p", { class: "view-subtitle" }, "Choose which tools each personality can use.")
    ),
    profileToolsSection.element
  );
  outlet.replaceChildren(view);

  await profileToolsSection.refresh();
}
