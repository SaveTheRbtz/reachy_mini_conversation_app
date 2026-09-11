type Child = Node | string | number | boolean | null | undefined | readonly Child[];
type ClassName = string | readonly (string | false | null | undefined)[];
type Attributes = {
  class?: ClassName | null;
  style?: Record<string, string>;
  dataset?: Record<string, string>;
  html?: string;
  onClick?: (event: MouseEvent) => void;
  [name: string]: string | number | boolean | null | undefined | ClassName
    | Record<string, string> | ((event: MouseEvent) => void);
};

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K, attrs: Attributes | null = {}, ...children: Child[]
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  const { class: classes, style, dataset, html, onClick, ...attributes } = attrs ?? {};
  if (classes) element.className = typeof classes === "string" ? classes : classes.filter(Boolean).join(" ");
  if (style) for (const [name, value] of Object.entries(style)) element.style.setProperty(name, value);
  if (dataset) Object.assign(element.dataset, dataset);
  if (html) element.innerHTML = html;
  if (onClick) element.onclick = onClick;
  for (const [name, value] of Object.entries(attributes)) {
    if (value != null && value !== false) element.setAttribute(name, String(value));
  }
  appendChildren(element, children);
  return element;
}

function appendChildren(parent: HTMLElement, children: readonly Child[]): void {
  for (const child of children) {
    if (child == null || typeof child === "boolean") continue;
    if (Array.isArray(child)) appendChildren(parent, child);
    else parent.append(child instanceof Node ? child : String(child));
  }
}

export function prettifyProfileName(name: string): string {
  return name.replace(/^profiles\/(?:builtin-|user-)/, "")
    .split(/[_-]/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function prettifyToolName(toolId: string): string {
  return prettifyProfileName(toolId.split("__").at(-1) ?? toolId);
}
