import { ROUTES } from "./constants.js";
import { createRouter } from "./router.js";
import { hidePersonalityBadge, mountPersonalityBadge, showPersonalityBadge, } from "./personality-badge.js";
import { mountHomeView } from "./views/home.js";
import { mountTalkView } from "./views/talk.js";
import { mountSettingsView } from "./views/settings.js";
import { mountToolsView } from "./views/tools.js";
const SETTINGS_RETURN_KEY = "settings-return-route";
function readSettingsReturn() {
    try {
        const route = sessionStorage.getItem(SETTINGS_RETURN_KEY) || "";
        return [ROUTES.TALK, ROUTES.PERSONALITIES, ROUTES.TOOLS].some((name) => name === route.split("?")[0])
            ? route
            : ROUTES.TALK;
    }
    catch (error) {
        console.warn("Could not read the settings return route", error);
        return ROUTES.TALK;
    }
}
function storeSettingsReturn(route) {
    try {
        sessionStorage.setItem(SETTINGS_RETURN_KEY, route);
    }
    catch (error) {
        console.warn("Could not persist the settings return route", error);
    }
}
function applyEmbeddedTheme() {
    const theme = new URLSearchParams(window.location.search).get("theme");
    if (theme !== "dark" && theme !== "light")
        return;
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
}
function boot() {
    applyEmbeddedTheme();
    const outlet = document.querySelector("#view-outlet");
    if (!outlet) {
        console.error("#view-outlet missing from index.html");
        return;
    }
    const router = createRouter({
        [ROUTES.TALK]: mountTalkView,
        [ROUTES.PERSONALITIES]: (ctx) => mountHomeView({ ...ctx, navigate: router.navigate }),
        [ROUTES.SETTINGS]: mountSettingsView,
        [ROUTES.TOOLS]: mountToolsView,
    }, { fallback: ROUTES.TALK, outlet, onRouteChange: syncHeaderForRoute });
    let settingsReturn = readSettingsReturn();
    const tools = document.querySelector('[data-action="open-tools"]');
    if (tools) {
        tools.addEventListener("click", (event) => {
            event.preventDefault();
            router.navigate(ROUTES.TOOLS);
        });
    }
    const gear = document.querySelector('[data-action="open-settings"]');
    if (gear) {
        gear.addEventListener("click", () => {
            const route = router.currentRoute() || ROUTES.TALK;
            if (route.split("?")[0] === ROUTES.SETTINGS) {
                router.navigate(settingsReturn);
            }
            else {
                settingsReturn = route;
                storeSettingsReturn(settingsReturn);
                router.navigate(ROUTES.SETTINGS);
            }
        });
    }
    const brand = document.querySelector('[data-action="go-home"]');
    if (brand) {
        brand.addEventListener("click", (event) => {
            event.preventDefault();
            router.navigate(ROUTES.TALK);
        });
    }
    const personalityBadge = document.querySelector('[data-action="open-personalities"]');
    if (personalityBadge) {
        personalityBadge.addEventListener("click", () => router.navigate(ROUTES.PERSONALITIES));
    }
    const back = document.querySelector('[data-action="go-back"]');
    if (back) {
        back.addEventListener("click", () => {
            const route = router.currentRoute() || ROUTES.TALK;
            const routeName = route.split("?")[0];
            if (routeName === ROUTES.SETTINGS) {
                router.navigate(settingsReturn);
            }
            else if (routeName === ROUTES.TOOLS &&
                new URLSearchParams(route.split("?")[1] || "").get("from") === "personalities") {
                router.navigate(ROUTES.PERSONALITIES);
            }
            else {
                router.navigate(ROUTES.TALK);
            }
        });
    }
    mountPersonalityBadge(document);
    function syncHeaderForRoute(route = router.currentRoute() || ROUTES.TALK) {
        const routeName = route.split("?")[0];
        if (tools) {
            const onTools = routeName === ROUTES.TOOLS;
            tools.classList.toggle("is-active", onTools);
            if (onTools)
                tools.setAttribute("aria-current", "page");
            else
                tools.removeAttribute("aria-current");
        }
        if (gear) {
            const onSettings = routeName === ROUTES.SETTINGS;
            gear.classList.toggle("is-active", onSettings);
            gear.setAttribute("aria-label", onSettings ? "Close settings" : "Open settings");
        }
        if (back) {
            back.hidden = routeName === ROUTES.TALK;
            let backLabel = "Back to conversation";
            if (routeName === ROUTES.SETTINGS) {
                const returnName = settingsReturn.split("?")[0];
                if (returnName === ROUTES.PERSONALITIES)
                    backLabel = "Back to personalities";
                if (returnName === ROUTES.TOOLS)
                    backLabel = "Back to tools";
            }
            else if (routeName === ROUTES.TOOLS &&
                new URLSearchParams(route.split("?")[1] || "").get("from") === "personalities") {
                backLabel = "Back to personalities";
            }
            back.setAttribute("aria-label", backLabel);
        }
        if (routeName === ROUTES.TALK)
            showPersonalityBadge();
        else
            hidePersonalityBadge();
    }
    router.start();
}
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
}
else {
    boot();
}
