# Testing

Tests should make changing the application safer and easier. They are executable examples of behavior,
not a record of individual patches. This guide owns the testing strategy for both Python and TypeScript.
The root [README](../README.md) owns installation and application usage.

## Choose the boundary before writing the test

Use the cheapest test that can observe the failure. Scope and resource cost are different:
an in-process API test can exercise validation, storage, and serialization together without a real network.
Do not enforce a unit/integration percentage or a coverage target. Coverage helps find omissions; it does
not establish that assertions are useful.

| Suite | Primary responsibility | Resources |
| --- | --- | --- |
| `tests/test_*.py`, `tests/audio/`, `tests/tools/` | Configuration, profile and memory persistence, API rules, supervisor and robot/audio behavior | Real application objects and temporary files; simulated hardware and remote I/O |
| `tests/realtime/` | Audio continuity, session readiness, backend tool ordering/failure, dialogue retention | Production conversation loop with controlled SDK-boundary events |
| `tests/integration/` | Application assumptions about SDK/WebSocket cancellation and shutdown | Actual SDK objects with controlled transport |
| `frontend/tests/unit/` | Frontend orchestration: pagination, retry, renewal, cancellation, readable errors | Vitest; no backend or browser |
| `frontend/tests/integration/` | Generated TypeScript client ↔ Python Connect interoperability, presence semantics, error codes, cancellation | Fresh Python process and temporary instance per test |
| `frontend/e2e/` | Complete user journeys, navigation, native dialogs, offline recovery, desktop/mobile usability | Playwright against the production-built SPA and Python application |
| `tests/packaging/` | Clean source → sdist → wheel, installed assets/profiles/imports, Windows path limits | Build tools and registry access; installed-package checks also run offline without Node |
| `tests/live/` | Stochastic speech, vision, search, tool-use, and memory semantics | Explicitly selected paid OpenAI evaluations; simulated robot |

The Python suite owns exhaustive API field-mask, validation, persistence, and error-mapping cases.
The cross-language suite owns the wire boundary. Browser tests prove that a user can complete a journey
and see the saved result; they should not repeat the entire API edge-case matrix.

## What a good case looks like

- Name the situation and observable outcome. Arrange only relevant state, perform one action or coherent
  workflow, and assert its result. Several assertions are appropriate when they describe the same outcome.
- Keep the important inputs and expected values visible in the test. Share resource setup and teardown,
  not assertions hidden behind a generic scenario runner. A little repeated setup is easier to read than
  a configurable fake application.
- Use named `pytest.param` or Vitest `test.each` cases when only inputs and expected outcomes change.
  Different workflows deserve separate tests; do not add branches to make them fit one table.
- Assert returned values, persisted state, emitted protocol messages, or visible UI behavior. Do not pin
  private fields, CSS classes, entire generated objects, or prompt wording merely because they exist.
  Prompt tests verify assembly and context boundaries; language quality belongs in live evaluations.
- Interaction counts are useful for contracts such as “never replay a command” or “never capture when
  disabled.” They are not a substitute for verifying the requested outcome.
- Preserve regression intent when consolidating. For each deleted case, identify the equivalent behavior
  test or explain why it only detected harmless implementation changes.

## Fixtures, dependencies, and cleanup

Prefer real application code, real temporary storage, and real protocol models. Replace unavailable or
nondeterministic boundaries: robot hardware, remote services, and controllable transport/time. A fake
must implement only the behavior the test needs; keep it typed and maintain its agreement with the real
interface. Use autospecced mocks for SDK methods when a recording boundary is sufficient.

Pytest fixtures are function-scoped unless a resource is immutable. Put narrowly shared fixtures in the
nearest `conftest.py`; reusable typed doubles belong in `tests/support/`. Never import a helper from a
`test_*.py` module. Pytest uses the installed editable application and importlib collection, without
manually changing `sys.path`. Unknown markers and invalid configuration fail the run.

The root configuration disables `.env` loading and removes machine-specific profile, voice, and idle
timeout settings before test collection, restoring the environment afterward. Tests that change runtime
configuration must use `monkeypatch` or an owning fixture so the next case starts clean.
`pytest-socket` restricts ordinary Python socket connections to loopback. It is a guard against accidental
external calls, not a sandbox for build subprocesses; packaging and live evaluation are separate gates.

Every task, thread, subprocess, client, and stream has an owner that closes and joins it, including on
assertion failure. Unexpected worker failures must fail the test. Suppress only expected cancellation,
never arbitrary exceptions from background work. Browser/API fixtures own fresh backend processes and
temporary storage; Playwright owns a fresh browser context for each case.

## Time and concurrency

Synchronize on an event, queue, task completion, or observable state. A fixed sleep is not proof that work
finished; repeating `Promise.resolve()` a fixed number of times is not synchronization either.
Use Vitest's async fake-timer advancement for retry/expiry logic. Use bounded polling for a real
cross-process state change and Playwright's retrying assertions for the DOM.

Timeouts bound test failure; they should not stand in for an assertion. Tests of actual SDK shutdown and
backpressure retain short real deadlines because those library lifecycles are the subject of the test.
Live microphone replay is intentionally paced in real time. Keep these exceptions local and explained.
Do not add test retries to conceal flaky ownership or timing. Run suspected cases independently and in
a different order; a test must not depend on a preceding case's state.

## Why these frameworks

Keep pytest: fixtures, parametrization, introspected assertions, and pytest-asyncio already fit the Python
application. Vitest replaces the separate Node test runner because it uses the existing Vite transform
pipeline and provides isolated modules/mocks and async fake timers. Type-check tests separately with
`vue-tsc`; successful TypeScript transpilation alone is not a type check.

Keep Playwright for native browser behavior and complete journeys. Locate controls by role and accessible
name; test what the user sees. Keep tracing/screenshots/video on failure. The current UI cases need real
navigation and native dialogs, so a second DOM emulator/component library would duplicate infrastructure.
If substantial standalone component logic develops, use Vue Test Utils with Vitest for its public
props/events/rendered behavior; do not create a second copy of every browser journey.

## Commands and CI

Start with `uv sync --frozen` and activate `.venv`; this also installs npm dependencies and builds the
generated contracts/SPA. Install Chromium once with `npx playwright install chromium`.

| Command | Use |
| --- | --- |
| `pytest` | Default Python behavior and local SDK tests; excludes packaging and live evaluations |
| `pytest tests/realtime/ -q` | Focused conversation-loop work |
| `pytest -m integration` | Local SDK boundary tests |
| `pytest -m packaging` | Clean build and installed distribution gate |
| `npm run test:unit` | Fast TypeScript behavior tests |
| `npm run test:integration` | Real generated-client/Python boundary |
| `npm run test:e2e` | Packaged desktop/mobile journeys |
| `npm test` | Frontend format/types/schema checks, unit tests, build, and integration tests |
| `pytest -m live tests/live/ -v` | Paid evaluations, with `OPENAI_API_KEY` set explicitly in the shell |

The default marker expression is overridden by an explicit `-m`. Live tests enable remote sockets and
fail immediately if credentials are absent; ordinary test commands never select them. Do not put a real
key into a command example or committed configuration. Live failures need investigation of the prompt,
model, service, and application; they are not deterministic PR gates.

Before review, run Ruff checks/formatting, strict mypy, `pytest`, `pytest -m packaging`, `npm test`,
`npm run test:e2e`, and `uv lock --check`. CI keeps Python behavior and packaging as separate steps on
Linux, macOS, and Windows, and runs frontend checks, unit, cross-language, and browser tests on Linux.
The distribution build also proves code generation from clean source. No emotions dataset is downloaded
for ordinary tests. Browser failure artifacts are retained for seven days.

## Research and references

Reviewed September 2026. Revisit a framework's documentation when changing its version or execution model;
the behavioral principles below should not need to be rediscovered for every patch.

- [Software Engineering at Google: Testing Overview](https://abseil.io/resources/swe-book/html/ch11.html):
  distinguish scope from resource cost, control nondeterminism, and use coverage diagnostically.
- [Software Engineering at Google: Unit Testing](https://abseil.io/resources/swe-book/html/ch12.html):
  test behavior through stable interfaces and keep tests understandable in isolation.
- [Google Testing Blog: Test Behavior, Not Implementation](https://testing.googleblog.com/2013/08/testing-on-toilet-test-behavior-not.html):
  assertions should survive behavior-preserving refactoring.
- [Google Testing Blog: Increase Test Fidelity by Avoiding Mocks](https://testing.googleblog.com/2024/02/increase-test-fidelity-by-avoiding-mocks.html):
  prefer real implementations where practical; maintain fakes against their actual boundaries.
- [Google Testing Blog: Tests Too DRY? Make Them DAMP!](https://testing.googleblog.com/2019/12/testing-on-toilet-tests-too-dry-make.html):
  local clarity takes priority over removing every repeated line.
- [pytest: Good Integration Practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
  and [fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html): installed source layout,
  importlib collection, explicit dependencies, and reliable teardown.
- [Vue: Testing](https://vuejs.org/guide/scaling-up/testing.html)
  and [Vitest](https://vitest.dev/guide/): select tools that fit Vite and test public component behavior.
- [Playwright: Best Practices](https://playwright.dev/docs/best-practices): isolated tests, resilient
  user-facing locators, auto-waiting, and failure diagnostics.
- [Go: Testing Time](https://go.dev/blog/testing-time): control time and synchronize concurrent work
  rather than guessing how long it takes.
- [Google Go Style: Table-Driven Tests](https://google.github.io/styleguide/go/decisions.html#table-driven-tests)
  and [Go Test Comments](https://go.dev/wiki/TestComments): named equivalent cases and diagnostic failures.
- [pytest-socket](https://github.com/miketheman/pytest-socket): supported network restrictions and explicit
  exceptions for tests that require remote services.
