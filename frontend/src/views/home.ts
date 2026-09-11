/** Home view: grid of personality cards. Select one to apply it and navigate to Talk. */

import { describeError, rpcCall, untilReady } from "../api.ts";
import type { Navigate, RouterContext } from "../router.ts";
import { AVATAR_BY_PROFILE, ROUTES, avatarFor } from "../constants.ts";
import { h, prettifyProfileName } from "../ui.ts";
import { openProfileModal } from "../components/profile-modal.ts";
import { confirmDialog } from "../components/confirm-dialog.ts";
import { setPendingApply } from "../pending-apply.ts";
import { setPersonality } from "../personality-badge.ts";

export async function mountHomeView({ outlet, signal, navigate }: RouterContext & { navigate: Navigate }) {
  const grid = h("div", { class: "personality-grid", role: "list" }, h("p", { class: "muted" }, "Loading…"));
  const view = h(
    "section",
    { class: "view view--home" },
    h(
      "header",
      { class: "view-header" },
      h("h1", { class: "view-title" }, "Choose a personality"),
      h(
        "p",
        { class: "view-subtitle" },
        "Pick how Reachy Mini should think and talk. Tap a card to start a conversation."
      )
    ),
    grid
  );
  outlet.replaceChildren(view);

  const status = h("p", { class: "view-status", role: "status", "aria-live": "polite" });
  view.appendChild(status);

  let personalities;
  try {
    personalities = await untilReady(() => rpcCall("personalities.list"), signal, () => {
      grid.replaceChildren(h("p", { class: "muted" }, "Waiting for Reachy to finish starting…"));
    });
  } catch (error) {
    if (signal.aborted) return;
    grid.replaceChildren(h("div", { class: "view-error" },
      h("p", null, "Could not list personalities"),
      h("p", { class: "muted small" }, describeError(error))
    ));
    return;
  }
  if (signal.aborted) return;

  const choices = personalities.choices;
  const current = personalities.current;
  const lockedTo = personalities.locked ? personalities.locked_to : null;

  grid.replaceChildren();
  for (const name of choices) {
    const disabled = Boolean(lockedTo) && name !== lockedTo;
    const editable = !personalities.locked && name.startsWith("user_personalities/");
    grid.appendChild(
      buildPersonalityCard({
        name,
        isActive: name === current,
        disabled,
        onSelect: () => handleSelection(name),
        onManageTools: () =>
          navigate(`${ROUTES.TOOLS}?profile=${encodeURIComponent(name)}&from=personalities`),
        onEdit: editable ? () => handleEditClick(name) : null,
        // The active personality must keep a card available to manage it.
        onDelete: editable && name !== current ? (slot) => handleDeleteClick(name, slot) : null,
      })
    );
  }
  if (!lockedTo) grid.appendChild(buildCustomCard({ onClick: handleCustomClick }));

  if (lockedTo) {
    status.textContent = `Personality locked to “${prettifyProfileName(lockedTo)}”; switching is disabled.`;
    status.classList.add("is-warning");
  }

  function handleSelection(name: string) {
    setPersonality(name);
    // Talk owns the pending request so it can show connection progress immediately.
    setPendingApply(name === current ? null : {
      name,
      promise: rpcCall("personalities.apply", { name, persist: false }),
    });
    void navigate(ROUTES.TALK);
  }

  async function handleCustomClick() {
    const created = await openProfileModal({
      mode: "create",
      signal,
    });
    if (!created || signal.aborted) return;
    status.classList.remove("is-warning", "is-error");
    status.textContent = `Saving "${created.name}"…`;
    let newName;
    try {
      const saveResult = await rpcCall("personalities.save", {
        name: created.name,
        instructions: created.instructions,
        greeting: created.greeting || null,
        voice: "", // falls back to the app default; user can change it in Settings
      });
      if (signal.aborted) return;
      newName = saveResult.value;
    } catch (error) {
      if (signal.aborted) return;
      status.textContent = `Failed to create profile: ${describeError(error)}`;
      status.classList.add("is-error");
      return;
    }
    setPersonality(newName);
    setPendingApply({ name: newName, promise: rpcCall("personalities.apply", { name: newName, persist: false }) });
    void navigate(ROUTES.TALK);
  }

  async function handleEditClick(name: string) {
    status.classList.remove("is-warning", "is-error");
    let personality;
    try {
      personality = await rpcCall("personalities.load", { name });
    } catch (error) {
      if (signal.aborted) return;
      status.textContent = `Could not load "${prettifyProfileName(name)}": ${describeError(error)}`;
      status.classList.add("is-error");
      return;
    }
    if (signal.aborted) return;

    const edited = await openProfileModal({
      mode: "edit",
      initial: {
        name: stripUserPrefix(name),
        instructions: personality.instructions,
        greeting: personality.greeting,
      },
      signal,
    });
    if (!edited || signal.aborted) return;

    status.textContent = `Saving "${prettifyProfileName(name)}"…`;
    try {
      await rpcCall("personalities.save", {
        // Strip the prefix: the save endpoint always writes under user_personalities/<name>.
        name: stripUserPrefix(name),
        instructions: edited.instructions,
        greeting: edited.greeting,
        voice: personality.voice,
        overwrite: true,
      });
    } catch (error) {
      if (signal.aborted) return;
      status.textContent = `Failed to save: ${describeError(error)}`;
      status.classList.add("is-error");
      return;
    }
    if (signal.aborted) return;

    // Reload a live personality so the session uses the updated instructions.
    if (name === current) {
      setPersonality(name);
      setPendingApply({ name, promise: rpcCall("personalities.apply", { name, persist: false, force: true }) });
      void navigate(ROUTES.TALK);
    } else {
      status.textContent = `Saved "${prettifyProfileName(name)}". It will apply next time you select it.`;
    }
  }

  async function handleDeleteClick(name: string, slot: HTMLElement) {
    const ok = await confirmDialog({
      title: "Delete personality?",
      message: `"${prettifyProfileName(name)}" will be permanently removed.`,
      confirmLabel: "Delete",
      danger: true,
      signal,
    });
    if (!ok || signal.aborted) return;
    status.classList.remove("is-warning", "is-error");
    try {
      await rpcCall("personalities.delete", { name });
    } catch (error) {
      if (signal.aborted) return;
      status.textContent = `Failed to delete: ${describeError(error)}`;
      status.classList.add("is-error");
      return;
    }
    if (signal.aborted) return;
    slot.remove();
    status.textContent = `Deleted "${prettifyProfileName(name)}".`;
  }
}

interface PersonalityCardOptions {
  name: string;
  isActive: boolean;
  disabled: boolean;
  onSelect: () => void;
  onManageTools: () => void;
  onEdit: (() => void) | null;
  onDelete: ((slot: HTMLElement) => void) | null;
}

function buildPersonalityCard({
  name, isActive, disabled, onSelect, onManageTools, onEdit, onDelete,
}: PersonalityCardOptions) {
  const hasAvatar = Object.prototype.hasOwnProperty.call(AVATAR_BY_PROFILE, stripUserPrefix(name));
  const card = h(
    "button",
    {
      type: "button",
      class: ["personality-card", isActive && "is-active", disabled && "is-disabled"],
      disabled: disabled ? "disabled" : null,
      "aria-pressed": isActive ? "true" : "false",
      "aria-label": `Use personality ${prettifyProfileName(name)}`,
      onClick: disabled ? undefined : onSelect,
    },
    h(
      "span",
      { class: "personality-card__avatar" },
      h("img", {
        src: avatarFor(stripUserPrefix(name)),
        alt: "",
        loading: "lazy",
        "aria-hidden": "true", // card label already names the personality

        class: !hasAvatar ? "personality-card__avatar--fallback" : null,
      })
    ),
    h("span", { class: "personality-card__name" }, prettifyProfileName(name)),
    isActive && checkBadge()
  );
  // Wrap so the edit/delete buttons are siblings, not nested <button>s inside the card button.
  const slot = h(
    "div",
    {
      class: ["personality-card-slot", isActive && "is-active", "has-tools-action"],
      role: "listitem",
    },
    card
  );
  slot.appendChild(h("button", {
    type: "button",
    class: "personality-card__tools",
    "aria-label": `Manage tools for ${prettifyProfileName(name)}`,
    onClick: onManageTools,
  }, "Manage tools"));
  if (onDelete) slot.appendChild(buildDeleteButton({ name, onDelete: () => onDelete(slot) }));
  if (onEdit) slot.appendChild(buildEditButton({ name, onEdit }));
  return slot;
}

/** Small overlay button to delete a user personality, anchored left of the edit button. */
function buildDeleteButton({ name, onDelete }: { name: string; onDelete: () => void }) {
  return h("button", {
    type: "button",
    class: "personality-card__delete",
    "aria-label": `Delete personality ${prettifyProfileName(name)}`,
    onClick: onDelete,
    html: `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M3 6h18"/>
        <path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2"/>
        <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
        <path d="M10 11v6"/>
        <path d="M14 11v6"/>
      </svg>`,
  });
}

/** Small overlay button to edit a user personality, anchored to the card corner. */
function buildEditButton({ name, onEdit }: { name: string; onEdit: () => void }) {
  return h("button", {
    type: "button",
    class: "personality-card__edit",
    "aria-label": `Edit personality ${prettifyProfileName(name)}`,
    onClick: onEdit,
    html: `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M12 20h9"/>
        <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/>
      </svg>`,
  });
}

function checkBadge() {
  const badge = h("span", { class: "personality-card__badge", "aria-hidden": "true" });
  badge.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="20 6 9 17 4 12"/>
    </svg>`;
  return badge;
}

function buildCustomCard({ onClick }: { onClick: () => void }) {
  const card = h(
    "button",
    {
      type: "button",
      class: "personality-card personality-card--custom",
      "aria-label": "Create a custom personality",
      onClick,
    },
    h("span", { class: "personality-card__plus", "aria-hidden": "true" }, "+"),
    h("span", { class: "personality-card__name" }, "Custom"),
    h("span", { class: "personality-card__hint" }, "Write your own prompt")
  );
  return h("div", { class: "personality-card-slot", role: "listitem" }, card);
}

function stripUserPrefix(name: string) {
  return name.replace(/^user_personalities\//, "");
}
