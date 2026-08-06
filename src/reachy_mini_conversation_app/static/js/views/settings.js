/** Settings view for the OpenAI Realtime connection and voice. */

import {
  applyVoice,
  describeError,
  getCurrentVoice,
  getStatus,
  listVoices,
  saveOpenAIConfig,
  untilReady,
} from "../api.js";
import { h } from "../ui.js";

export async function mountSettingsView({ outlet, signal }) {
  const connectionSection = buildConnectionSection({
    onSaved: () => refreshStatus({ statusSection, connectionSection, signal }),
  });
  const voiceSection = buildVoiceSection();
  const statusSection = buildStatusSection();

  outlet.replaceChildren(
    h(
      "section",
      { class: "view view--settings" },
      h(
        "header",
        { class: "view-header" },
        h("h1", { class: "view-title" }, "Settings"),
        h("p", { class: "view-subtitle" }, "OpenAI Realtime connection, voice, and session state.")
      ),
      connectionSection.element,
      voiceSection.element,
      statusSection.element
    )
  );

  await Promise.all([
    refreshStatus({ statusSection, connectionSection, signal }),
    refreshVoices({ voiceSection, signal }),
  ]);
}

function buildConnectionSection({ onSaved } = {}) {
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
    try {
      await saveOpenAIConfig(apiKey.value);
      apiKey.value = "";
      status.textContent = "Saved. The Realtime session is reconnecting.";
      await onSaved?.();
    } catch (error) {
      status.textContent = `Failed to save: ${describeError(error)}`;
      status.classList.add("is-error");
    } finally {
      submitButton.disabled = false;
      apiKey.disabled = false;
      form.removeAttribute("aria-busy");
    }
  });

  return {
    element,
    syncFromStatus(payload) {
      apiKey.placeholder = payload?.has_key ? "Configured" : "sk-…";
    },
  };
}

function buildVoiceSection() {
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
    if (submitButton.disabled || !select.value) return;
    submitButton.disabled = true;
    select.disabled = true;
    form.setAttribute("aria-busy", "true");
    status.classList.remove("is-error");
    status.textContent = "Applying…";
    try {
      const result = await applyVoice(select.value);
      status.textContent = result?.status || "Voice applied.";
    } catch (error) {
      status.textContent = `Failed to apply: ${describeError(error)}`;
      status.classList.add("is-error");
    } finally {
      submitButton.disabled = !select.value;
      select.disabled = !select.value;
      form.removeAttribute("aria-busy");
    }
  });

  return {
    element,
    setOptions(voices, current) {
      select.replaceChildren();
      if (!voices.length) {
        select.appendChild(h("option", { value: "" }, "No voices available"));
        select.disabled = true;
        submitButton.disabled = true;
        status.textContent = "Voices are unavailable right now.";
        return;
      }
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
  const list = h("dl", { class: "settings-status-grid" }, statusRow("Realtime", "Loading…"));
  const element = h(
    "section",
    { class: "settings-section" },
    h("h2", { class: "settings-section-title" }, "Current state"),
    list
  );

  return {
    element,
    render(payload) {
      const state = payload.connected ? "connected" : payload.connection_state || "not_started";
      const labels = {
        connected: "Connected",
        connecting: "Connecting…",
        disconnected: "Disconnected",
        not_started: "Not started",
        waiting_for_config: "Waiting for API key",
      };
      list.replaceChildren(
        statusRow("API key", payload.has_key ? "Configured" : "Missing", payload.has_key ? "ok" : "warn"),
        statusRow("Model", payload.model || "-"),
        statusRow("Voice", payload.voice || "-"),
        statusRow("Realtime", labels[state] || "Unavailable", state === "connected" ? "ok" : "warn")
      );
      if (payload.connection_error) {
        list.appendChild(statusRow("Connection error", payload.connection_error, "warn"));
      }
    },
    renderUnavailable(error) {
      list.replaceChildren(statusRow("Realtime", `Unavailable: ${describeError(error)}`, "warn"));
    },
  };
}

function statusRow(label, value, tone) {
  return h(
    "div",
    { class: ["settings-status-row", tone && `is-${tone}`] },
    h("dt", { class: "settings-status-label" }, label),
    h("dd", { class: "settings-status-value" }, value)
  );
}

async function refreshStatus({ statusSection, connectionSection, signal }) {
  try {
    const payload = await untilReady(getStatus, signal);
    if (signal.aborted) return;
    statusSection.render(payload);
    connectionSection.syncFromStatus(payload);
  } catch (error) {
    if (!signal.aborted) statusSection.renderUnavailable(error);
  }
}

async function refreshVoices({ voiceSection, signal }) {
  let voices = [];
  let current = "";
  try {
    voices = await untilReady(listVoices, signal);
    current = (await getCurrentVoice())?.voice || "";
  } catch {
    voices = [];
  }
  if (!signal.aborted) voiceSection.setOptions(voices, current);
}
