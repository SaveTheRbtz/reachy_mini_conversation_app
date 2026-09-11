import { avatarFor } from "./constants.js";
import { prettifyProfileName } from "./ui.js";
let rootEl = null;
let rowEl = null;
let nameEl = null;
let avatarImg = null;
export function mountPersonalityBadge(headerRoot = document) {
    const next = headerRoot.querySelector('[data-component="personality-badge"]');
    if (!next)
        return;
    rootEl = next;
    rowEl = next.closest('[data-component="personality-row"]');
    nameEl = next.querySelector(".app-shell__personality-name");
    avatarImg = next.querySelector(".app-shell__personality-avatar img");
}
export function setPersonality(rawName) {
    if (!rootEl || !nameEl || !avatarImg)
        return;
    if (!rawName)
        return;
    const cleanName = rawName.replace(/^user_personalities\//, "");
    nameEl.textContent = prettifyProfileName(rawName);
    avatarImg.src = avatarFor(cleanName);
}
export function showPersonalityBadge() {
    if (rowEl)
        rowEl.hidden = false;
    else if (rootEl)
        rootEl.hidden = false;
}
export function hidePersonalityBadge() {
    if (rowEl)
        rowEl.hidden = true;
    else if (rootEl)
        rootEl.hidden = true;
}
