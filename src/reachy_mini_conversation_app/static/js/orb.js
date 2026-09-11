/** Microphone button with connection status and local playback animation. */

import { h } from "./ui.js";
import { GLOW_BY_STATE, ORB_STATES } from "./constants.js";

/** Build the microphone control and its independent playback animation. */
export function createOrb({ initialState = ORB_STATES.IDLE } = {}) {
  const indicator = h(
    "span",
    { class: "convo-orb__indicator", "aria-hidden": "true" },
    micIcon(),
    micOffIcon(),
    spinnerIndicator(),
    errorIcon()
  );

  const root = h(
    "button",
    {
      type: "button",
      class: "convo-orb",
      dataset: { state: initialState, playing: "false" },
      "aria-label": "Conversation status",
      style: { "--glow": GLOW_BY_STATE[initialState] },
    },
    h("span", { class: "convo-orb__glow", "aria-hidden": "true" }),
    h("span", { class: "convo-orb__ring", "aria-hidden": "true" }),
    h("span", { class: "convo-orb__ring-outer", "aria-hidden": "true" }),
    h("span", { class: "convo-orb__core" }, indicator)
  );

  function setState(nextState, playing = false) {
    root.dataset.state = nextState;
    root.dataset.playing = String(playing);
    root.style.setProperty("--glow", GLOW_BY_STATE[nextState]);
  }

  return { root, setState };
}

function spinnerIndicator() {
  return h("span", { class: "ind ind-spinner" });
}

function micIcon() {
  return h("span", {
    class: "ind ind-mic",
    html: `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="9" y="2" width="6" height="12" rx="3" fill="currentColor" stroke="none"/>
        <path d="M5 10a7 7 0 0 0 14 0"/>
        <line x1="12" y1="19" x2="12" y2="22"/>
        <line x1="8" y1="22" x2="16" y2="22"/>
      </svg>`,
  });
}

function micOffIcon() {
  return h("span", {
    class: "ind ind-mic-off",
    html: `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <rect x="9" y="2" width="6" height="12" rx="3" fill="currentColor" stroke="none"/>
        <path d="M5 10a7 7 0 0 0 14 0"/>
        <line x1="12" y1="19" x2="12" y2="22"/>
        <line x1="8" y1="22" x2="16" y2="22"/>
        <line x1="4" y1="3" x2="20" y2="21" stroke-width="2"/>
      </svg>`,
  });
}

function errorIcon() {
  return h("span", {
    class: "ind ind-error",
    html: `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="9"/>
        <line x1="12" y1="8" x2="12" y2="13"/>
        <line x1="12" y1="16" x2="12" y2="16"/>
      </svg>`,
  });
}
