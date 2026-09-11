import { api, describeError, watchConversation } from "../api.ts";
import { Conversation_ConnectionState as ConnectionState, type Conversation } from "../gen/reachy/conversation/v1/api_pb.ts";
import type { RouterContext } from "../router.ts";
import { ORB_STATES, type OrbState } from "../constants.ts";
import { createOrb } from "../orb.ts";
import { consumePendingApply } from "../pending-apply.ts";
import { setPersonality } from "../personality-badge.ts";
import { h, prettifyProfileName } from "../ui.ts";

const CAPTION_BY_STATE: Record<OrbState, string> = {
  muted: "Microphone muted",
  idle: "Microphone on",
  connecting: "Connecting to OpenAI Live...",
  error: "Connection error",
};

export async function mountTalkView({ outlet, signal }: RouterContext) {
  let conversation: Conversation | undefined;
  let connected = true;
  let togglePending = false;
  let startupProfile = "";
  let locked = true;
  const caption = h("p", { class: "talk__caption", role: "status", "aria-live": "polite" },
    CAPTION_BY_STATE.connecting);
  const orb = createOrb({ initialState: ORB_STATES.CONNECTING });
  orb.root.disabled = true;
  orb.root.addEventListener("click", onMicTap);
  const defaultAction = document.querySelector<HTMLButtonElement>('[data-component="default-personality-action"]');
  if (defaultAction) {
    defaultAction.hidden = true;
    defaultAction.addEventListener("click", onSetDefault);
  }
  signal.addEventListener("abort", () => {
    if (defaultAction) {
      defaultAction.hidden = true;
      defaultAction.removeEventListener("click", onSetDefault);
    }
  }, { once: true });
  outlet.replaceChildren(h("section", { class: "view view--talk" },
    h("div", { class: "talk__orb-wrap" }, orb.root), caption));

  const pending = consumePendingApply();
  if (pending) {
    caption.textContent = `Applying "${prettifyProfileName(pending.name)}"…`;
    try {
      await pending.promise;
    } catch (error) {
      if (signal.aborted) return;
      orb.setState(ORB_STATES.ERROR);
      caption.textContent = `Failed to apply personality: ${describeError(error)}`;
      return;
    }
  }
  if (signal.aborted) return;
  void watchConversation(signal, (snapshot) => {
    conversation = snapshot;
    connected = true;
    setPersonality(snapshot.profile);
    render();
  }, () => {
    connected = false;
    render();
  });
  try {
    const settings = await api.getSettings({ name: "settings" }, { signal });
    if (signal.aborted) return;
    startupProfile = settings.startupProfile;
    locked = Boolean(settings.lockedProfile);
    render();
  } catch (error) {
    if (!signal.aborted) console.warn("Failed to load startup personality", error);
  }

  function render() {
    const state: OrbState = !connected || conversation?.connectionState === ConnectionState.DISCONNECTED
      ? ORB_STATES.ERROR : conversation?.connectionState !== ConnectionState.CONNECTED
        ? ORB_STATES.CONNECTING : conversation.muted ? ORB_STATES.MUTED : ORB_STATES.IDLE;
    const playing = connected && conversation?.connectionState === ConnectionState.CONNECTED && conversation.playing;
    orb.setState(state, playing);
    caption.textContent = `${playing ? "Speaking · " : ""}${CAPTION_BY_STATE[state]}`;
    orb.root.disabled = !conversation || togglePending;
    orb.root.setAttribute("aria-pressed", String(Boolean(conversation && !conversation.muted)));
    orb.root.setAttribute("aria-label", !conversation ? "Loading microphone state" :
      conversation.muted ? "Unmute microphone" : "Mute microphone");
    if (defaultAction) defaultAction.hidden = locked || !conversation || conversation.profile === startupProfile;
  }

  async function onMicTap() {
    if (!conversation || togglePending) return;
    togglePending = true;
    render();
    try {
      const updated = await api.updateConversation({
        conversation: { name: "conversation", muted: !conversation.muted },
        updateMask: { paths: ["muted"] },
      }, { signal });
      if (!signal.aborted && conversation) {
        conversation.muted = updated.muted;
        render();
      }
    } catch (error) {
      if (!signal.aborted) caption.textContent = `Failed to toggle the microphone: ${describeError(error)}`;
    } finally {
      togglePending = false;
      if (!signal.aborted) orb.root.disabled = !conversation;
    }
  }

  async function onSetDefault() {
    if (!defaultAction || !conversation) return;
    defaultAction.disabled = true;
    const profile = conversation.profile;
    try {
      const settings = await api.updateSettings({
        settings: { name: "settings", startupProfile: profile },
        updateMask: { paths: ["startup_profile"] },
      }, { signal });
      if (signal.aborted) return;
      startupProfile = settings.startupProfile;
      render();
      caption.textContent = `"${prettifyProfileName(profile)}" will be used at startup.`;
    } catch (error) {
      if (!signal.aborted) caption.textContent = `Failed to save default: ${describeError(error)}`;
    } finally {
      defaultAction.disabled = false;
    }
  }
}
