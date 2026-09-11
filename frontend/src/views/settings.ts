/** Settings view for the OpenAI Live connection and voice. */

import { api, describeError, untilReady } from "../api.ts";
import { Conversation_ConnectionState as ConnectionState, type Conversation, type Settings } from "../gen/reachy/conversation/v1/api_pb.ts";
import type { RouterContext } from "../router.ts";
import { h } from "../ui.ts";

export async function mountSettingsView({ outlet, signal }: RouterContext) {
  const connectionSection = buildConnectionSection(refreshStatus, signal);
  const voiceSection = buildVoiceSection(signal, refreshStatus);
  const statusSection = buildStatusSection();

  outlet.replaceChildren(
    h(
      "section",
      { class: "view view--settings" },
      h(
        "header",
        { class: "view-header" },
        h("h1", { class: "view-title" }, "Settings"),
        h("p", { class: "view-subtitle" }, "OpenAI Live connection, voice, and session state.")
      ),
      connectionSection.element,
      voiceSection.element,
      statusSection.element
    )
  );

  await Promise.all([refreshStatus(), voiceSection.refresh(signal)]);

  async function refreshStatus() {
    try {
      const [conversation, settings] = await untilReady(() => Promise.all([
        api.getConversation({ name: "conversation" }, { signal }),
        api.getSettings({ name: "settings" }, { signal }),
      ]), signal);
      if (signal.aborted) return;
      statusSection.render(conversation, settings);
      connectionSection.syncFromStatus(settings);
    } catch (error) {
      if (!signal.aborted) statusSection.renderUnavailable(error);
    }
  }
}

function buildConnectionSection(onSaved: () => Promise<void>, signal: AbortSignal) {
  const apiKey = h("input", {
    type: "password",
    name: "api_key",
    autocomplete: "off",
    placeholder: "sk-…",
    class: "settings-input",
  });
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const submitButton = h("button", { type: "submit", class: "btn btn--primary" }, "Save API key");
  const form = h(
    "form",
    { class: "settings-form" },
    h(
      "label",
      { class: "settings-field" },
      h("span", { class: "settings-label" }, "OpenAI API key"),
      apiKey
    ),
    h(
      "p",
      { class: "settings-hint" },
      "Stored in the app instance .env file. The current key is never returned to the browser."
    ),
    h("div", { class: "settings-actions" }, submitButton),
    status
  );
  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Connection"),
    form
  );

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (submitButton.disabled || !apiKey.value.trim()) return;
    submitButton.disabled = true;
    apiKey.disabled = true;
    form.setAttribute("aria-busy", "true");
    status.classList.remove("is-error");
    status.textContent = "Saving and reconnecting…";
    let saved = false;
    try {
      await api.setSettingsApiKey({ name: "settings", apiKey: apiKey.value }, { signal });
      apiKey.value = "";
      saved = true;
      await api.restartConversation({ name: "conversation" }, { signal });
      if (signal.aborted) return;
      status.textContent = "Saved. The Live session is reconnecting.";
      await onSaved();
    } catch (error) {
      status.textContent = `${saved ? "API key saved, but could not reconnect" : "Failed to save"}: ${describeError(error)}`;
      status.classList.add("is-error");
    } finally {
      submitButton.disabled = false;
      apiKey.disabled = false;
      form.removeAttribute("aria-busy");
    }
  });

  return {
    element,
    syncFromStatus(payload: Settings) {
      apiKey.placeholder = payload.apiKeyConfigured ? "Configured" : "sk-…";
    },
  };
}

function buildVoiceSection(signal: AbortSignal, onSaved: () => Promise<void>) {
  const select = h(
    "select",
    { class: "settings-select", name: "voice", disabled: "disabled" },
    h("option", { value: "" }, "Loading voices…")
  );
  const status = h("p", { class: "settings-status", role: "status", "aria-live": "polite" });
  const submitButton = h(
    "button",
    { type: "submit", class: "btn btn--primary", disabled: "disabled" },
    "Apply voice"
  );
  const form = h(
    "form",
    { class: "settings-form" },
    h("label", { class: "settings-field" }, h("span", { class: "settings-label" }, "Voice"), select),
    h("div", { class: "settings-actions" }, submitButton),
    status
  );
  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Voice"),
    form
  );

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (submitButton.disabled) return;
    submitButton.disabled = true;
    select.disabled = true;
    form.setAttribute("aria-busy", "true");
    status.classList.remove("is-error");
    status.textContent = "Applying…";
    let saved = false;
    try {
      await api.updateSettings({
        settings: { name: "settings", voiceOverride: select.value },
        updateMask: { paths: ["voice_override"] },
      }, { signal });
      saved = true;
      await api.restartConversation({ name: "conversation" }, { signal });
      if (signal.aborted) return;
      status.textContent = "Voice saved. The Live session is reconnecting.";
      await onSaved();
    } catch (error) {
      status.textContent = `${saved ? "Voice saved, but could not reconnect" : "Failed to save voice"}: ${describeError(error)}`;
      status.classList.add("is-error");
    } finally {
      submitButton.disabled = false;
      select.disabled = false;
      form.removeAttribute("aria-busy");
    }
  });

  return {
    element,
    async refresh(signal: AbortSignal) {
      let voices: string[] = [];
      let current = "";
      try {
        const [capabilities, settings] = await untilReady(() => Promise.all([
          api.getCapabilities({ name: "capabilities" }, { signal }),
          api.getSettings({ name: "settings" }, { signal }),
        ]), signal);
        voices = capabilities.availableVoices;
        current = settings.voiceOverride;
      } catch (error) {
        console.warn("Failed to load voices", error);
        voices = [];
      }
      if (signal.aborted) return;
      select.replaceChildren();
      if (!voices.length) {
        select.appendChild(h("option", { value: "" }, "No voices available"));
        select.disabled = true;
        submitButton.disabled = true;
        status.textContent = "Voices are unavailable right now.";
        return;
      }
      select.appendChild(h("option", { value: "" }, "Use personality default"));
      for (const voice of voices) {
        const option = h("option", { value: voice }, voice);
        if (voice === current) option.selected = true;
        select.appendChild(option);
      }
      select.disabled = false;
      submitButton.disabled = false;
      status.textContent = "";
    },
  };
}

function buildStatusSection() {
  const list = h("dl", { class: "settings-status-grid" }, statusRow("Live", "Loading…"));
  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Current state"),
    list
  );

  return {
    element,
    render(payload: Conversation, settings: Settings) {
      const labels = new Map<ConnectionState, string>([
        [ConnectionState.CONNECTED, "Connected"],
        [ConnectionState.CONNECTING, "Connecting…"],
        [ConnectionState.DISCONNECTED, "Disconnected"],
        [ConnectionState.NOT_STARTED, "Not started"],
        [ConnectionState.WAITING_FOR_CONFIG, "Waiting for API key"],
      ]);
      list.replaceChildren(
        statusRow("API key", settings.apiKeyConfigured ? "Configured" : "Missing", settings.apiKeyConfigured ? "ok" : "warn"),
        statusRow("Model", payload.model || "-"),
        statusRow("Voice", payload.voice || "-"),
        statusRow("Live", labels.get(payload.connectionState) ?? "Unknown", payload.connectionState === ConnectionState.CONNECTED ? "ok" : "warn")
      );
      if (payload.connectionError) {
        list.appendChild(statusRow("Connection error", payload.connectionError, "warn"));
      }
    },
    renderUnavailable(error: unknown) {
      list.replaceChildren(statusRow("Live", `Unavailable: ${describeError(error)}`, "warn"));
    },
  };
}

function statusRow(label: string, value: string, tone?: "ok" | "warn") {
  return h(
    "div",
    { class: ["settings-status-row", tone && `is-${tone}`] },
    h("dt", { class: "settings-status-label" }, label),
    h("dd", { class: "settings-status-value" }, value)
  );
}
