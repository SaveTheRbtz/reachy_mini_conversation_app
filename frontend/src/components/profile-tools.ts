/** Per-personality tool access controls. */

import { api, describeError, listProfiles } from "../api.ts";
import type { Profile, ToolInfo } from "../gen/reachy/conversation/v1/api_pb.ts";
import { h, prettifyProfileName, prettifyToolName } from "../ui.ts";
import { confirmDialog } from "./confirm-dialog.ts";

interface ProfileToolsOptions {
  signal?: AbortSignal;
  initialProfile?: string | null;
  onProfileChanged?: (profile: string) => void;
}

export function buildProfileToolsSection({ signal, initialProfile = null, onProfileChanged }: ProfileToolsOptions = {}) {
  const profileSelect = h("select", {
    class: "settings-select",
    name: "profile_tools_profile",
    disabled: "disabled",
    "aria-label": "Personality to configure",
  });
  const summary = h("div", { class: "settings-toolset-summary", "aria-live": "polite" });
  const toolGroups = h(
    "div",
    { class: "settings-tool-groups", "aria-live": "polite" },
    h("p", { class: "settings-hint" }, "Loading tool access…")
  );
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const resetButton = h(
    "button",
    { type: "button", class: "btn btn--ghost", disabled: "disabled" },
    "Restore defaults"
  );
  const saveButton = h(
    "button",
    { type: "button", class: "btn btn--primary", disabled: "disabled" },
    "Save tool access"
  );
  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Tool access"),
    h(
      "p",
      { class: "settings-hint settings-section-intro" },
      "Choose exactly which tools each personality can use."
    ),
    h(
      "label",
      { class: "settings-field" },
      h("span", { class: "settings-label" }, "Configure for"),
      profileSelect
    ),
    summary,
    toolGroups,
    h("div", { class: "settings-actions settings-toolset-actions" }, resetButton, saveButton),
    status
  );

  let currentProfile: Profile | null = null;
  let profiles: Profile[] = [];
  let availableTools: ToolInfo[] = [];
  let activeProfile = "";
  let locked = true;
  let initialEnabledTools = new Set<string>();
  let dirty = false;
  let busy = false;

  function selectedToolIds() {
    return Array.from(toolGroups.querySelectorAll<HTMLInputElement>('input[type="checkbox"]:checked')).map(
      (checkbox) => checkbox.value
    );
  }

  function selectionChanged() {
    const selectedTools = selectedToolIds();
    return (
      selectedTools.length !== initialEnabledTools.size ||
      selectedTools.some((toolId) => !initialEnabledTools.has(toolId))
    );
  }

  function syncActions() {
    const editable = Boolean(currentProfile) && !locked;
    profileSelect.disabled = busy || !profiles.length;
    toolGroups.querySelectorAll<HTMLInputElement>('input[type="checkbox"]').forEach((checkbox) => {
      const disabled = busy || !editable;
      checkbox.disabled = disabled;
      checkbox.closest(".settings-tool-choice")?.classList.toggle("is-disabled", disabled);
    });
    resetButton.disabled = busy || !editable || !currentProfile?.toolOverride;
    saveButton.disabled = busy || !editable || !dirty;
  }

  function setBusy(nextBusy: boolean) {
    busy = nextBusy;
    element.toggleAttribute("aria-busy", nextBusy);
    syncActions();
  }

  function setDirty(nextDirty: boolean) {
    dirty = nextDirty;
    syncActions();
  }

  function populateProfiles(payload: Profile) {
    profileSelect.replaceChildren();
    for (const profile of profiles) {
      const label = `${profile.displayName}${profile.name === activeProfile ? " · Active" : ""}`;
      profileSelect.appendChild(
        h("option", { value: profile.name, selected: profile.name === payload.name ? "selected" : null }, label)
      );
    }
  }

  function renderSummary(payload: Profile) {
    const enabledCount = payload.effectiveToolIds.length;
    const pills = [];
    if (payload.name === activeProfile) pills.push(h("span", { class: "settings-pill is-active" }, "Active"));
    if (locked) pills.push(h("span", { class: "settings-pill" }, "Locked"));
    summary.replaceChildren(
      h(
        "div",
        { class: "settings-toolset-summary-copy" },
        h("strong", null, `${enabledCount} ${enabledCount === 1 ? "tool" : "tools"} enabled`),
        h(
          "span",
          { class: "settings-toolset-mode" },
          Boolean(payload.toolOverride) ? "Customized for this personality" : "Using profile defaults"
        )
      ),
      ...pills
    );
  }

  function renderToolGroups(payload: Profile) {
    const enabled = new Set(payload.effectiveToolIds);
    const groups = [
      { title: "Available tools", tools: availableTools, unavailable: false },
      {
        title: "Unavailable selections",
        unavailable: true,
        tools: payload.effectiveToolIds.filter((id) => !availableTools.some((tool) => tool.id === id)).map((toolId) => ({
          id: toolId,
          description: "Its source is not installed or the tool is no longer exposed.",
        })),
      },
    ].filter((group) => group.tools.length > 0);

    toolGroups.replaceChildren();
    if (!groups.length) {
      toolGroups.appendChild(h("p", { class: "settings-hint" }, "No configurable tools are available."));
      return;
    }

    for (const group of groups) {
      const fieldset = h(
        "fieldset",
        { class: "settings-tool-group" },
        h(
          "legend",
          { class: "settings-tool-group-title" },
          group.title,
          h("span", { class: "settings-tool-group-count" }, String(group.tools.length))
        ),
        h(
          "div",
          { class: "settings-tool-grid" },
          ...group.tools.map((tool) => toolChoice(tool, enabled.has(tool.id), group.unavailable))
        )
      );
      fieldset.addEventListener("change", () => {
        const changed = selectionChanged();
        status.textContent = changed ? "Unsaved changes" : "";
        status.classList.remove("is-error");
        setDirty(changed);
      });
      toolGroups.appendChild(fieldset);
    }
  }

  function render(payload: Profile) {
    currentProfile = payload;
    initialEnabledTools = new Set(payload.effectiveToolIds);
    populateProfiles(payload);
    renderSummary(payload);
    renderToolGroups(payload);
    status.classList.remove("is-error");
    status.textContent = !locked ? "" : "Tool editing is locked by the administrator.";
    setDirty(false);
  }

  async function load(profile: string | null = null) {
    setBusy(true);
    status.classList.remove("is-error");
    status.textContent = "Loading tools…";
    try {
      const [listed, capabilities, conversation, settings] = await Promise.all([
        listProfiles(signal),
        api.getCapabilities({ name: "capabilities" }, { signal }),
        api.getConversation({ name: "conversation" }, { signal }),
        api.getSettings({ name: "settings" }, { signal }),
      ]);
      profiles = listed;
      availableTools = capabilities.availableTools;
      activeProfile = conversation.profile;
      locked = Boolean(settings.lockedProfile);
      const payload = await api.getProfile({ name: profile || activeProfile }, { signal });
      if (signal?.aborted) return;
      render(payload);
      onProfileChanged?.(payload.name);
    } catch (error) {
      if (signal?.aborted) return;
      status.textContent = `Could not load tool access: ${describeError(error)}`;
      status.classList.add("is-error");
      if (currentProfile) {
        profileSelect.value = currentProfile.name;
      } else {
        toolGroups.replaceChildren();
      }
    } finally {
      if (!signal?.aborted) setBusy(false);
    }
  }

  async function confirmDiscard() {
    if (!dirty) return true;
    return confirmDialog({
      title: "Discard unsaved tool changes?",
      message: "Your selections for this personality have not been saved.",
      confirmLabel: "Discard changes",
      signal,
    });
  }

  profileSelect.addEventListener("change", async () => {
    const selectedProfile = profileSelect.value;
    if (!(await confirmDiscard())) {
      profileSelect.value = currentProfile?.name ?? "";
      return;
    }
    await load(selectedProfile);
  });

  saveButton.addEventListener("click", async () => {
    if (saveButton.disabled || !currentProfile) return;
    const enabledTools = selectedToolIds();
    setBusy(true);
    status.classList.remove("is-error");
    status.textContent = "Saving tool access…";
    let saved = false;
    try {
      const payload = await api.updateProfile({
        profile: { name: currentProfile.name, toolOverride: { toolIds: enabledTools } },
        updateMask: { paths: ["tool_override"] },
      }, { signal });
      saved = true;
      if (signal?.aborted) return;
      render(payload);
      if (payload.name === activeProfile) await api.restartConversation({ name: "conversation" }, { signal });
      if (signal?.aborted) return;
      status.textContent = payload.name === activeProfile
        ? "Saved. The conversation is restarting with the updated tools." : "Tool access saved.";
    } catch (error) {
      if (signal?.aborted) return;
      status.textContent = `${saved ? "Tool access saved, but could not restart" : "Could not save tool access"}: ${describeError(error)}`;
      status.classList.add("is-error");
    } finally {
      if (!signal?.aborted) setBusy(false);
    }
  });

  resetButton.addEventListener("click", async () => {
    if (resetButton.disabled || !currentProfile) return;
    const confirmed = await confirmDialog({
      title: "Restore profile defaults?",
      message: `This replaces the custom tool selection for “${prettifyProfileName(currentProfile.name)}”.`,
      confirmLabel: "Restore defaults",
      signal,
    });
    if (!confirmed || signal?.aborted) return;

    setBusy(true);
    status.classList.remove("is-error");
    status.textContent = "Restoring defaults…";
    let saved = false;
    try {
      const payload = await api.updateProfile({
        profile: { name: currentProfile.name },
        updateMask: { paths: ["tool_override"] },
      }, { signal });
      saved = true;
      if (signal?.aborted) return;
      render(payload);
      if (payload.name === activeProfile) await api.restartConversation({ name: "conversation" }, { signal });
      if (signal?.aborted) return;
      status.textContent = payload.name === activeProfile
        ? "Saved. The conversation is restarting with the updated tools." : "Tool access saved.";
    } catch (error) {
      if (signal?.aborted) return;
      status.textContent = `${saved ? "Defaults restored, but could not restart" : "Could not restore defaults"}: ${describeError(error)}`;
      status.classList.add("is-error");
    } finally {
      if (!signal?.aborted) setBusy(false);
    }
  });

  return {
    element,
    confirmDiscard,
    hasUnsavedChanges() {
      return dirty;
    },
    async refresh() {
      await load(currentProfile?.name ?? initialProfile);
    },
  };
}

function toolChoice(tool: Pick<ToolInfo, "id" | "description">, checked: boolean, unavailable: boolean) {
  const input = h("input", {
    type: "checkbox",
    value: tool.id,
    checked: checked ? "checked" : null,
  });
  return h(
    "label",
    {
      class: ["settings-tool-choice", unavailable && "is-unavailable"],
    },
    input,
    h(
      "span",
      { class: "settings-tool-choice-copy" },
      h("strong", { class: "settings-tool-choice-name" }, prettifyToolName(tool.id)),
      unavailable ? h("code", { class: "settings-tool-choice-id" }, tool.id) : null,
      tool.description ? h("span", { class: "settings-tool-choice-description" }, tool.description) : null
    )
  );
}
