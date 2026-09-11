/** Talk view: microphone control and playback status from the robot. */

import { applyPersonality, getStatus, listPersonalities, setMicMuted, subscribe } from "../api.js";
import { ORB_STATES } from "../constants.js";
import { createOrb } from "../orb.js";
import { consumePendingApply } from "../pending-apply.js";
import { setPersonality } from "../personality-badge.js";
import { h, prettifyProfileName } from "../ui.js";

const CAPTION_BY_STATE = Object.freeze({
  [ORB_STATES.MUTED]: "Microphone muted",
  [ORB_STATES.IDLE]: "Microphone on",
  [ORB_STATES.CONNECTING]: "Connecting to OpenAI Live...",
  [ORB_STATES.ERROR]: "Connection error",
});

export async function mountTalkView({ outlet, signal }) {
  const pending = consumePendingApply();
  let muted = false;
  let micReady = false;
  let togglePending = false;
  let activePersonality = null;
  let connectionState = ORB_STATES.CONNECTING;
  let playing = false;
  let connectionVersion = 0;
  let playbackVersion = 0;
  let statusRequest = null;
  const subscriptions = [];

  const caption = h(
    "p",
    { class: "talk__caption", role: "status", "aria-live": "polite" },
    CAPTION_BY_STATE[ORB_STATES.CONNECTING]
  );
  const defaultAction = document.querySelector('[data-component="default-personality-action"]');
  if (defaultAction) {
    defaultAction.hidden = true;
    defaultAction.addEventListener("click", onSetDefault);
  }
  const orb = createOrb({ initialState: ORB_STATES.CONNECTING });
  orb.root.disabled = true;
  orb.root.addEventListener("click", onMicTap);
  syncMicAria();

  signal.addEventListener("abort", cleanup, { once: true });

  const view = h(
    "section",
    { class: "view view--talk" },
    h("div", { class: "talk__orb-wrap" }, orb.root),
    caption
  );
  outlet.replaceChildren(view);

  if (pending) {
    caption.textContent = `Applying "${prettifyProfileName(pending.name)}"…`;
    try {
      await pending.promise;
    } catch (error) {
      if (signal.aborted) return;
      orb.setState(ORB_STATES.ERROR);
      caption.textContent = `Failed to apply personality: ${error?.message || error}`;
      return;
    }
    if (signal.aborted) return;
    caption.textContent = CAPTION_BY_STATE[ORB_STATES.CONNECTING];
    void refreshPersonalityState();
  } else {
    // Deep link to /talk with no pending apply: refresh the header badge.
    void refreshPersonalityState();
  }

  subscriptions.push(
    subscribe("conversation.activity", ({ reason }) => {
      switch (reason) {
        case "connected":
          connectionVersion++;
          connectionState = ORB_STATES.IDLE;
          break;
        case "disconnected":
          connectionVersion++;
          playbackVersion++;
          connectionState = ORB_STATES.ERROR;
          playing = false;
          break;
        case "playback_started":
        case "playback_stopped":
          playbackVersion++;
          playing = reason === "playback_started";
          break;
        default:
          return;
      }
      renderStatus();
    }),
    subscribe("rpc.connection", ({ connected }) => {
      if (connected) {
        void refreshStatus();
      } else {
        connectionVersion++;
        playbackVersion++;
        connectionState = ORB_STATES.ERROR;
        playing = false;
        renderStatus();
      }
    })
  );
  await refreshStatus();

  function cleanup() {
    for (const unsubscribe of subscriptions) unsubscribe();
    if (defaultAction) {
      defaultAction.hidden = true;
      defaultAction.removeEventListener("click", onSetDefault);
    }
  }

  function refreshStatus() {
    if (statusRequest) return statusRequest;
    orb.root.disabled = true;
    const requestedConnectionVersion = connectionVersion;
    const requestedPlaybackVersion = playbackVersion;
    statusRequest = getStatus().then((status) => {
      if (signal.aborted) return;
      if (connectionVersion === requestedConnectionVersion) {
        connectionState = status.connected ? ORB_STATES.IDLE :
          status.connection_state === "disconnected" ? ORB_STATES.ERROR : ORB_STATES.CONNECTING;
      }
      if (playbackVersion === requestedPlaybackVersion) playing = Boolean(status.playing);
      if (!togglePending) muted = Boolean(status.muted);
      micReady = true;
      renderStatus();
    }).catch((error) => {
      console.warn("Failed to load conversation status", error);
      if (signal.aborted || connectionVersion !== requestedConnectionVersion) return;
      connectionState = ORB_STATES.ERROR;
      if (playbackVersion === requestedPlaybackVersion) playing = false;
      renderStatus();
    }).finally(() => {
      statusRequest = null;
      orb.root.disabled = !micReady || togglePending;
    });
    return statusRequest;
  }

  function renderStatus() {
    const state = connectionState === ORB_STATES.IDLE
      ? (muted ? ORB_STATES.MUTED : ORB_STATES.IDLE) : connectionState;
    const showingPlayback = connectionState === ORB_STATES.IDLE && playing;
    orb.setState(state, showingPlayback);
    caption.textContent = `${showingPlayback ? "Speaking · " : ""}${CAPTION_BY_STATE[state]}`;
    orb.root.disabled = !micReady || togglePending || statusRequest !== null;
    syncMicAria();
  }

  async function onMicTap() {
    if (!micReady || togglePending || statusRequest) return;
    togglePending = true;
    try {
      const data = await setMicMuted(!muted);
      muted = Boolean(data?.muted);
    } catch (error) {
      if (!signal.aborted) {
        caption.textContent = `Failed to toggle the microphone: ${error?.message || error}`;
      }
      return;
    } finally {
      togglePending = false;
      orb.root.disabled = !micReady || statusRequest !== null;
    }
    if (signal.aborted) return;
    renderStatus();
  }

  async function refreshPersonalityState() {
    const personalityState = await fetchPersonalityState();
    if (signal.aborted || personalityState == null) return;
    activePersonality = personalityState.current;
    setPersonality(personalityState.current);
    const shouldHide = personalityState.locked || personalityState.current === personalityState.startup;
    if (defaultAction) {
      defaultAction.hidden = shouldHide;
    }
  }

  async function onSetDefault() {
    if (!defaultAction || !activePersonality) return;
    defaultAction.disabled = true;
    caption.textContent = `Saving "${prettifyProfileName(activePersonality)}" as default...`;
    try {
      await applyPersonality(activePersonality, { persist: true });
      if (signal.aborted) return;
      defaultAction.hidden = true;
      caption.textContent = `"${prettifyProfileName(activePersonality)}" will be used at startup.`;
    } catch (error) {
      if (!signal.aborted) {
        caption.textContent = `Failed to save default: ${error?.message || error}`;
      }
    } finally {
      defaultAction.disabled = false;
    }
  }

  function syncMicAria() {
    if (!micReady) {
      orb.root.setAttribute("aria-pressed", "false");
      orb.root.setAttribute("aria-label", "Loading microphone state");
      return;
    }
    orb.root.setAttribute("aria-pressed", String(!muted));
    orb.root.setAttribute("aria-label", muted ? "Unmute microphone" : "Mute microphone");
  }
}

async function fetchPersonalityState() {
  try {
    const data = await listPersonalities();
    const current = data?.current;
    if (!current) return null;
    return {
      current,
      startup: data?.startup || "default",
      locked: Boolean(data?.locked),
    };
  } catch {
    return null;
  }
}
