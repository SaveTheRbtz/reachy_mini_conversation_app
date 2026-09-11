/** Per-personality tool access controls. */

import { describeError, rpcCall } from "../api.ts";
import type { AvailableTool, ProfileTools } from "../contracts.ts";
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

  let currentPayload: ProfileTools | null = null;
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
    const editable = currentPayload?.editable ?? false;
    profileSelect.disabled = busy || !currentPayload?.profiles.length;
    toolGroups.querySelectorAll<HTMLInputElement>('input[type="checkbox"]').forEach((checkbox) => {
      const disabled = busy || !editable;
      checkbox.disabled = disabled;
      checkbox.closest(".settings-tool-choice")?.classList.toggle("is-disabled", disabled);
    });
    resetButton.disabled = busy || !editable || !currentPayload?.overridden;
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

  function populateProfiles(payload: ProfileTools) {
    profileSelect.replaceChildren();
    for (const profile of payload.profiles) {
      const label = `${prettifyProfileName(profile.id)}${profile.active ? " · Active" : ""}`;
      profileSelect.appendChild(
        h("option", { value: profile.id, selected: profile.id === payload.profile ? "selected" : null }, label)
      );
    }
  }

  function renderSummary(payload: ProfileTools) {
    const enabledCount = payload.enabled_tools.length;
    const pills = [];
    if (payload.is_active) pills.push(h("span", { class: "settings-pill is-active" }, "Active"));
    if (!payload.editable) pills.push(h("span", { class: "settings-pill" }, "Locked"));
    summary.replaceChildren(
      h(
        "div",
        { class: "settings-toolset-summary-copy" },
        h("strong", null, `${enabledCount} ${enabledCount === 1 ? "tool" : "tools"} enabled`),
        h(
          "span",
          { class: "settings-toolset-mode" },
          payload.overridden ? "Customized for this personality" : "Using profile defaults"
        )
      ),
      ...pills
    );
  }

  function renderToolGroups(payload: ProfileTools) {
    const enabled = new Set(payload.enabled_tools);
    const groups = [
      { title: "Available tools", tools: payload.available_tools, unavailable: false },
      {
        title: "Unavailable selections",
        unavailable: true,
        tools: payload.unavailable_enabled_tools.map((toolId) => ({
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

  function render(payload: ProfileTools) {
    currentPayload = payload;
    initialEnabledTools = new Set(payload.enabled_tools);
    populateProfiles(payload);
    renderSummary(payload);
    renderToolGroups(payload);
    status.classList.remove("is-error");
    status.textContent = payload.editable ? "" : "Tool editing is locked by the administrator.";
    setDirty(false);
  }

  async function load(profile: string | null = null) {
    setBusy(true);
    status.classList.remove("is-error");
    status.textContent = "Loading tools…";
    try {
      const payload = await rpcCall("profile_tools.get", { profile });
      if (signal?.aborted) return;
      render(payload);
      onProfileChanged?.(payload.profile);
    } catch (error) {
      if (signal?.aborted) return;
      status.textContent = `Could not load tool access: ${describeError(error)}`;
      status.classList.add("is-error");
      if (currentPayload) {
        profileSelect.value = currentPayload.profile;
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
      profileSelect.value = currentPayload?.profile ?? "";
      return;
    }
    await load(selectedProfile);
  });

  saveButton.addEventListener("click", async () => {
    if (saveButton.disabled || !currentPayload) return;
    const enabledTools = selectedToolIds();
    setBusy(true);
    status.classList.remove("is-error");
    status.textContent = "Saving tool access…";
    try {
      const payload = await rpcCall("profile_tools.save", { profile: currentPayload.profile, enabled_tools: enabledTools });
      if (signal?.aborted) return;
      render(payload);
      status.textContent = payload.message;
    } catch (error) {
      if (signal?.aborted) return;
      status.textContent = `Could not save tool access: ${describeError(error)}`;
      status.classList.add("is-error");
    } finally {
      if (!signal?.aborted) setBusy(false);
    }
  });

  resetButton.addEventListener("click", async () => {
    if (resetButton.disabled || !currentPayload) return;
    const confirmed = await confirmDialog({
      title: "Restore profile defaults?",
      message: `This replaces the custom tool selection for “${prettifyProfileName(currentPayload.profile)}”.`,
      confirmLabel: "Restore defaults",
      signal,
    });
    if (!confirmed || signal?.aborted) return;

    setBusy(true);
    status.classList.remove("is-error");
    status.textContent = "Restoring defaults…";
    try {
      const payload = await rpcCall("profile_tools.reset", { profile: currentPayload.profile });
      if (signal?.aborted) return;
      render(payload);
      status.textContent = payload.message;
    } catch (error) {
      if (signal?.aborted) return;
      status.textContent = `Could not restore defaults: ${describeError(error)}`;
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
      await load(currentPayload?.profile ?? initialProfile);
    },
  };
}

function toolChoice(tool: AvailableTool, checked: boolean, unavailable: boolean) {
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
