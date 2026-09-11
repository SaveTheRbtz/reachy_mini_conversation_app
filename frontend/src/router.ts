export type Navigate = (route: string) => Promise<boolean>;
export interface LeaveGuard {
  shouldBlock(): boolean;
  confirm(): Promise<boolean>;
}
export interface RouterContext {
  outlet: HTMLElement;
  signal: AbortSignal;
  searchParams: URLSearchParams;
  setLeaveGuard(guard: LeaveGuard | null): void;
  replaceRoute(route: string): void;
}
type RouteHandler = (context: RouterContext) => void | Promise<void>;
interface RouterOptions {
  outlet: HTMLElement;
  fallback?: string;
  onRouteChange?: (route: string) => void;
}

export function createRouter(
  routes: Record<string, RouteHandler>, { fallback = "#/", outlet, onRouteChange }: RouterOptions
) {
  let currentController: AbortController | null = null;
  let currentRoute: string | null = null;
  let leaveGuard: LeaveGuard | null = null;
  let pendingTransition: Promise<unknown> = Promise.resolve();

  function resolve(route = window.location.hash || fallback) {
    const routeName = route.split("?")[0] ?? fallback;
    return Object.prototype.hasOwnProperty.call(routes, routeName) ? route : fallback;
  }

  function renderRouteError(route: string, error: unknown) {
    const div = document.createElement("div");
    div.className = "route-error";
    div.textContent = `Failed to render ${route}: ${error instanceof Error ? error.message : String(error)}`;
    return div;
  }

  function mount(route: string) {
    leaveGuard = null;
    currentController?.abort();
    outlet.replaceChildren();

    currentRoute = route;
    currentController = new AbortController();
    const controller = currentController;
    const routeName = route.split("?")[0] ?? fallback;
    const queryStart = route.indexOf("?");
    const searchParams = new URLSearchParams(queryStart === -1 ? "" : route.slice(queryStart + 1));
    const context: RouterContext = {
      outlet,
      signal: controller.signal,
      searchParams,
      setLeaveGuard(guard) {
        if (!controller.signal.aborted) leaveGuard = guard;
      },
      replaceRoute(nextRoute) {
        if (controller.signal.aborted) return;
        const resolvedRoute = resolve(nextRoute);
        if (resolvedRoute.split("?")[0] !== routeName) {
          throw new Error("replaceRoute cannot change views");
        }
        currentRoute = resolvedRoute;
        window.history.replaceState(null, "", resolvedRoute);
        onRouteChange?.(resolvedRoute);
      },
    };
    const handler = routes[routeName];
    void Promise.resolve().then(() => {
      if (!handler) throw new Error(`Unknown route: ${routeName}`);
      return handler(context);
    }).catch((error: unknown) => {
      if (context.signal.aborted) return;
      console.error("Route handler failed for", route, error);
      outlet.replaceChildren(renderRouteError(route, error));
    });
    onRouteChange?.(route);
  }

  async function transitionTo(route: string, updateHash: boolean) {
    const nextRoute = resolve(route);
    if (nextRoute === currentRoute) {
      if (currentRoute && window.location.hash !== currentRoute) {
        window.history.replaceState(null, "", currentRoute);
      }
      return true;
    }

    if (leaveGuard?.shouldBlock() && !(await leaveGuard.confirm())) {
      if (currentRoute && window.location.hash !== currentRoute) {
        window.location.hash = currentRoute;
      }
      return false;
    }

    if (updateHash && window.location.hash !== nextRoute) {
      window.location.hash = nextRoute;
    } else if (window.location.hash !== nextRoute) {
      window.history.replaceState(null, "", nextRoute);
    }
    mount(nextRoute);
    return true;
  }

  function enqueueTransition(route: string, updateHash = false) {
    const transition = pendingTransition.then(() => transitionTo(route, updateHash));
    pendingTransition = transition.catch((error) => {
      console.error("Route transition failed", error);
    });
    return transition;
  }

  function onBeforeUnload(event: BeforeUnloadEvent) {
    if (!leaveGuard?.shouldBlock()) return;
    event.preventDefault();
    event.returnValue = "";
  }

  return {
    start() {
      window.addEventListener("hashchange", () => enqueueTransition(window.location.hash));
      window.addEventListener("beforeunload", onBeforeUnload);
      const target = resolve();
      if (window.location.hash !== target) {
        window.history.replaceState(null, "", target);
      }
      void enqueueTransition(target);
    },
    navigate(route: string) {
      return enqueueTransition(route, true);
    },
    currentRoute() {
      return currentRoute;
    },
  };
}
