import { h, prettifyProfileName } from "../ui.ts";

const NAME_PATTERN = /^[a-z](?:[a-z0-9-]{0,56}[a-z0-9])?$/;

export interface ProfileForm {
  name: string;
  instructions: string;
  greeting: string;
}

type ProfileModalOptions = { signal?: AbortSignal } & (
  | { mode?: "create" }
  | { mode: "edit"; initial: ProfileForm }
);

export function openProfileModal(options: ProfileModalOptions = {}): Promise<ProfileForm | null> {
  const { signal } = options;
  const isEdit = options.mode === "edit";
  const initial = options.mode === "edit" ? options.initial : { name: "", instructions: "", greeting: "" };

  return new Promise((resolve) => {
    if (signal?.aborted) {
      resolve(null);
      return;
    }

    const returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const nameInput = h("input", {
      type: "text",
      name: "name",
      required: !isEdit,
      readonly: isEdit,
      autocomplete: "off",
      spellcheck: "false",
      placeholder: "e.g. zen-master",
      pattern: "[a-z][a-z0-9-]*",
      value: initial.name,
      class: ["modal__input", isEdit && "is-readonly"],
    });
    const instructionsInput = h("textarea", {
      name: "instructions",
      required: true,
      rows: "8",
      placeholder: "You are a calm, slow-speaking zen guide. Pause between sentences. Encourage the user to breathe.",
      class: "modal__textarea",
    }, initial.instructions);
    const greetingInput = h("textarea", {
      name: "greeting",
      rows: "3",
      placeholder: "Start the conversation with a short greeting in character.",
      class: "modal__textarea",
    }, initial.greeting);
    const errorBox = h("p", { class: "modal__error", role: "alert", "aria-live": "polite" });
    const cancelButton = h("button", { type: "button", class: "btn btn--ghost" }, "Cancel");
    const submitButton = h("button", { type: "submit", class: "btn btn--primary" }, isEdit ? "Save changes" : "Create & start");
    const fields = [
      { input: nameInput, label: "Name" },
      { input: instructionsInput, label: "Instructions" },
      { input: greetingInput, label: "Startup greeting prompt" },
    ];
    const form = h(
      "form",
      { class: "modal__form" },
      ...fields.map(({ input, label }) => h(
        "label",
        { class: "modal__field" },
        h("span", { class: "modal__label" }, label),
        input
      )),
      errorBox,
      h("div", { class: "modal__actions" }, cancelButton, submitButton)
    );
    const dialog = h(
      "div",
      { class: "modal", role: "dialog", "aria-modal": "true", "aria-labelledby": "custom-profile-title" },
      h(
        "header",
        { class: "modal__header" },
        h("h2", { id: "custom-profile-title", class: "modal__title" },
          isEdit ? `Edit ${prettifyProfileName(initial.name)}` : "Create a custom personality"),
        h("p", { class: "modal__subtitle" }, "Define how Reachy should behave and greet people.")
      ),
      form
    );
    const overlay = h("div", { class: "modal-overlay", role: "presentation" }, dialog);
    document.body.appendChild(overlay);
    requestAnimationFrame(() => (isEdit ? instructionsInput : nameInput).focus());

    let settled = false;
    function close(value: ProfileForm | null) {
      if (settled) return;
      settled = true;
      window.removeEventListener("keydown", onKeydown);
      signal?.removeEventListener("abort", onAbort);
      overlay.remove();
      if (returnFocus?.isConnected) returnFocus.focus();
      resolve(value);
    }

    function onKeydown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        close(null);
      } else if (event.key === "Tab") {
        if (event.shiftKey && document.activeElement === nameInput) {
          event.preventDefault();
          submitButton.focus();
        } else if (!event.shiftKey && document.activeElement === submitButton) {
          event.preventDefault();
          nameInput.focus();
        }
      }
    }

    function onAbort() {
      close(null);
    }

    overlay.addEventListener("click", (event) => {
      if (event.target === overlay) close(null);
    });
    window.addEventListener("keydown", onKeydown);
    signal?.addEventListener("abort", onAbort, { once: true });
    cancelButton.addEventListener("click", () => close(null));
    for (const { input } of fields) {
      input.addEventListener("input", () => errorBox.classList.remove("is-visible"));
    }
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const name = isEdit ? initial.name : nameInput.value.trim();
      const instructions = instructionsInput.value.trim();
      const greeting = greetingInput.value.trim();

      if (!isEdit && !name) return showError(errorBox, "Please pick a name.");
      if (!isEdit && !NAME_PATTERN.test(name)) {
        return showError(errorBox, "Use up to 58 lowercase letters, numbers or dashes; start with a letter and end with a letter or number.");
      }
      if (!instructions) return showError(errorBox, "Please write some instructions.");
      close({ name, instructions, greeting });
    });
  });
}

function showError(errorBox: HTMLElement, message: string) {
  errorBox.textContent = message;
  errorBox.classList.add("is-visible");
}
