let pending = null;
export function setPendingApply(entry) {
    pending = entry;
    entry?.promise.catch((error) => console.warn("Personality apply failed", error));
}
export function consumePendingApply() {
    const entry = pending;
    pending = null;
    return entry;
}
