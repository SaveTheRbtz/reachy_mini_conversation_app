import { describeError, rpcCall, subscribe } from "../api.js";
import { ORB_STATES } from "../constants.js";
import { createOrb } from "../orb.js";
import { consumePendingApply } from "../pending-apply.js";
import { setPersonality } from "../personality-badge.js";
import { h, prettifyProfileName } from "../ui.js";
const CAPTION_BY_STATE = {
    [ORB_STATES.MUTED]: "Microphone muted",
    [ORB_STATES.IDLE]: "Microphone on",
    [ORB_STATES.CONNECTING]: "Connecting to OpenAI Live...",
    [ORB_STATES.ERROR]: "Connection error",
};
export async function mountTalkView({ outlet, signal }) {
    const pending = consumePendingApply();
    let microphone = "loading";
    let togglePending = false;
    let activePersonality = null;
    let connectionState = ORB_STATES.CONNECTING;
    let playing = false;
    let connectionVersion = 0;
    let playbackVersion = 0;
    let statusRequest = null;
    const subscriptions = [];
    const caption = h("p", { class: "talk__caption", role: "status", "aria-live": "polite" }, CAPTION_BY_STATE[ORB_STATES.CONNECTING]);
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
    const view = h("section", { class: "view view--talk" }, h("div", { class: "talk__orb-wrap" }, orb.root), caption);
    outlet.replaceChildren(view);
    if (pending) {
        caption.textContent = `Applying "${prettifyProfileName(pending.name)}"…`;
        try {
            await pending.promise;
        }
        catch (error) {
            if (signal.aborted)
                return;
            orb.setState(ORB_STATES.ERROR);
            caption.textContent = `Failed to apply personality: ${describeError(error)}`;
            return;
        }
        if (signal.aborted)
            return;
        caption.textContent = CAPTION_BY_STATE[ORB_STATES.CONNECTING];
    }
    void refreshPersonalityState();
    subscriptions.push(subscribe("conversation.activity", ({ reason }) => {
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
    }), subscribe("rpc.connection", ({ connected }) => {
        if (connected) {
            void refreshStatus();
        }
        else {
            connectionVersion++;
            playbackVersion++;
            connectionState = ORB_STATES.ERROR;
            playing = false;
            renderStatus();
        }
    }));
    await refreshStatus();
    function cleanup() {
        for (const unsubscribe of subscriptions)
            unsubscribe();
        if (defaultAction) {
            defaultAction.hidden = true;
            defaultAction.removeEventListener("click", onSetDefault);
        }
    }
    function refreshStatus() {
        if (statusRequest)
            return statusRequest;
        orb.root.disabled = true;
        const requestedConnectionVersion = connectionVersion;
        const requestedPlaybackVersion = playbackVersion;
        statusRequest = rpcCall("conversation.status").then((status) => {
            if (signal.aborted)
                return;
            if (connectionVersion === requestedConnectionVersion) {
                connectionState = status.connected ? ORB_STATES.IDLE :
                    status.connection_state === "disconnected" ? ORB_STATES.ERROR : ORB_STATES.CONNECTING;
            }
            if (playbackVersion === requestedPlaybackVersion)
                playing = status.playing;
            if (!togglePending)
                microphone = status.muted ? "muted" : "on";
            renderStatus();
        }).catch((error) => {
            console.warn("Failed to load conversation status", error);
            if (signal.aborted || connectionVersion !== requestedConnectionVersion)
                return;
            connectionState = ORB_STATES.ERROR;
            if (playbackVersion === requestedPlaybackVersion)
                playing = false;
            renderStatus();
        }).finally(() => {
            statusRequest = null;
            orb.root.disabled = microphone === "loading" || togglePending;
        });
        return statusRequest;
    }
    function renderStatus() {
        const state = connectionState === ORB_STATES.IDLE
            ? (microphone === "muted" ? ORB_STATES.MUTED : ORB_STATES.IDLE) : connectionState;
        const showingPlayback = connectionState === ORB_STATES.IDLE && playing;
        orb.setState(state, showingPlayback);
        caption.textContent = `${showingPlayback ? "Speaking · " : ""}${CAPTION_BY_STATE[state]}`;
        orb.root.disabled = microphone === "loading" || togglePending || statusRequest !== null;
        syncMicAria();
    }
    async function onMicTap() {
        if (microphone === "loading" || togglePending || statusRequest)
            return;
        togglePending = true;
        try {
            const result = await rpcCall("conversation.mic", { muted: microphone === "on" });
            microphone = result.muted ? "muted" : "on";
        }
        catch (error) {
            if (!signal.aborted) {
                caption.textContent = `Failed to toggle the microphone: ${describeError(error)}`;
            }
            return;
        }
        finally {
            togglePending = false;
            orb.root.disabled = statusRequest !== null;
        }
        if (signal.aborted)
            return;
        renderStatus();
    }
    async function refreshPersonalityState() {
        try {
            const personality = await rpcCall("personalities.list");
            if (signal.aborted)
                return;
            activePersonality = personality.current;
            setPersonality(personality.current);
            if (defaultAction) {
                defaultAction.hidden = personality.locked || personality.current === personality.startup;
            }
        }
        catch (error) {
            console.warn("Failed to load the active personality", error);
        }
    }
    async function onSetDefault() {
        if (!defaultAction || !activePersonality)
            return;
        defaultAction.disabled = true;
        caption.textContent = `Saving "${prettifyProfileName(activePersonality)}" as default...`;
        try {
            await rpcCall("personalities.apply", { name: activePersonality, persist: true });
            if (signal.aborted)
                return;
            defaultAction.hidden = true;
            caption.textContent = `"${prettifyProfileName(activePersonality)}" will be used at startup.`;
        }
        catch (error) {
            if (!signal.aborted) {
                caption.textContent = `Failed to save default: ${describeError(error)}`;
            }
        }
        finally {
            defaultAction.disabled = false;
        }
    }
    function syncMicAria() {
        if (microphone === "loading") {
            orb.root.setAttribute("aria-pressed", "false");
            orb.root.setAttribute("aria-label", "Loading microphone state");
            return;
        }
        orb.root.setAttribute("aria-pressed", String(microphone === "on"));
        orb.root.setAttribute("aria-label", microphone === "muted" ? "Unmute microphone" : "Mute microphone");
    }
}
