export function h(tag, attrs = {}, ...children) {
    const element = document.createElement(tag);
    const { class: classes, style, dataset, html, onClick, ...attributes } = attrs ?? {};
    if (classes)
        element.className = typeof classes === "string" ? classes : classes.filter(Boolean).join(" ");
    if (style)
        for (const [name, value] of Object.entries(style))
            element.style.setProperty(name, value);
    if (dataset)
        Object.assign(element.dataset, dataset);
    if (html)
        element.innerHTML = html;
    if (onClick)
        element.onclick = onClick;
    for (const [name, value] of Object.entries(attributes)) {
        if (value != null && value !== false)
            element.setAttribute(name, String(value));
    }
    appendChildren(element, children);
    return element;
}
function appendChildren(parent, children) {
    for (const child of children) {
        if (child == null || typeof child === "boolean")
            continue;
        if (Array.isArray(child))
            appendChildren(parent, child);
        else
            parent.append(child instanceof Node ? child : String(child));
    }
}
export function prettifyProfileName(name) {
    return name.replace(/^user_personalities\//, "")
        .split(/[_-]/)
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
}
export function prettifyToolName(toolId) {
    return prettifyProfileName(toolId.split("__").at(-1) ?? toolId);
}
