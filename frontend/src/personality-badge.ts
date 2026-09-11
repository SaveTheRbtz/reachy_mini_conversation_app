import { avatarFor } from "./constants.ts";
import { prettifyProfileName } from "./ui.ts";

let rootEl: HTMLElement | null = null;
let rowEl: HTMLElement | null = null;
let nameEl: HTMLElement | null = null;
let avatarImg: HTMLImageElement | null = null;

/** Bind setters to the static markup. Safe to call multiple times. */
export function mountPersonalityBadge(headerRoot: ParentNode = document) {
  const next = headerRoot.querySelector<HTMLElement>('[data-component="personality-badge"]');
  if (!next) return;
  rootEl = next;
  rowEl = next.closest<HTMLElement>('[data-component="personality-row"]');
  nameEl = next.querySelector<HTMLElement>(".app-shell__personality-name");
  avatarImg = next.querySelector<HTMLImageElement>(".app-shell__personality-avatar img");
}

/** Update the badge content. Pass a falsy name to keep the previous value. */
export function setPersonality(rawName: string) {
  if (!rootEl || !nameEl || !avatarImg) return;
  if (!rawName) return;
  const cleanName = rawName.replace(/^user_personalities\//, "");
  nameEl.textContent = prettifyProfileName(rawName);
  avatarImg.src = avatarFor(cleanName);
}

export function showPersonalityBadge() {
  if (rowEl) rowEl.hidden = false;
  else if (rootEl) rootEl.hidden = false;
}

export function hidePersonalityBadge() {
  if (rowEl) rowEl.hidden = true;
  else if (rootEl) rootEl.hidden = true;
}
