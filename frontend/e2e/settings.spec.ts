import { test, expect } from "./fixtures.ts";

test("dark theme survives navigation and unsaved settings require an explicit decision", async ({
  page,
  request,
}, testInfo) => {
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto("/?theme=dark");
  await expect(page.getByText("Microphone on", { exact: true })).toBeVisible();
  await expect(page.locator("html")).toHaveCSS("color-scheme", "dark");
  await page.screenshot({ path: testInfo.outputPath("conversation-dark.png"), fullPage: true });
  await page.getByRole("link", { name: "Settings", exact: true }).click();
  await expect(page).toHaveURL(/\/settings$/);
  await page.reload();
  await expect(page.locator("html")).toHaveCSS("color-scheme", "dark");
  await expect(page.getByLabel("Voice", { exact: true })).toHaveValue("");
  await page.screenshot({ path: testInfo.outputPath("settings-dark.png"), fullPage: true });
  await page.getByLabel("Voice", { exact: true }).selectOption("marin");
  await page.getByRole("link", { name: "Personalities", exact: true }).click();
  const discard = page.getByRole("dialog", { name: "Discard unsaved changes?" });
  await expect(discard).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(discard).not.toBeVisible();
  await expect(page.getByLabel("Voice", { exact: true })).toHaveValue("marin");
  await page.getByRole("link", { name: "Personalities", exact: true }).click();
  await discard.getByRole("button", { name: "Discard changes", exact: true }).click();
  await expect(page).toHaveURL(/\/profiles$/);
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(page.getByRole("link", { name: "Default", exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("personalities-dark.png") });
  await page.getByRole("link", { name: "Settings", exact: true }).click();
  await expect(page.getByLabel("Voice", { exact: true })).toHaveValue("");
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({ startup_settings: { voice: null } });
});

test("voice preferences persist and can return to the personality default", async ({ page, request }) => {
  await page.goto("/settings");
  await page.getByLabel("Voice", { exact: true }).selectOption("marin");
  await page.getByRole("button", { name: "Save preferences", exact: true }).click();
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({
      startup_settings: { voice: "marin" },
      conversation: { voice: "marin" },
    });
  await page.reload();
  await expect(page.getByLabel("Voice", { exact: true })).toHaveValue("marin");
  await page.getByLabel("Voice", { exact: true }).selectOption("");
  await page.getByRole("button", { name: "Save preferences", exact: true }).click();
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({
      startup_settings: { voice: null },
      conversation: { voice: "meridian" },
    });
  await page.reload();
  await expect(page.getByLabel("Voice", { exact: true })).toHaveValue("");
});

test("a saved key is cleared immediately and is not resent when reconnect fails", async ({
  page,
  request,
}) => {
  await page.goto("/settings");
  await request.put("/__test/faults", { data: { rpc_unavailable_methods: ["RestartConversation"] } });
  await page.getByLabel("OpenAI API key", { exact: true }).fill("sk-synthetic-e2e-test");
  await page.getByRole("button", { name: "Save API key", exact: true }).click();
  await expect(page.getByLabel("OpenAI API key", { exact: true })).toBeEmpty();
  await expect(page.getByRole("alert")).toContainText(/saved/i);
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({
      credential_saved: true,
      rpc_calls: { SetSettingsApiKey: 1, RestartConversation: 1 },
    });
  await request.put("/__test/faults", { data: {} });
  await page.getByRole("link", { name: "Conversation", exact: true }).click();
  await page.getByRole("button", { name: "Reconnect", exact: true }).click();
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({
      rpc_calls: { SetSettingsApiKey: 1, RestartConversation: 2 },
    });
  await expect(page.getByText("Ready to talk", { exact: true })).toBeVisible();
});

test.describe("first use", () => {
  test.use({ backendArgs: ["--no-key"] });
  test("saving a key starts the waiting conversation @mobile", async ({ page, request }, testInfo) => {
    await page.goto("/settings");
    await expect(page.getByLabel("OpenAI API key", { exact: true })).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath("settings.png"), fullPage: true });
    await page.getByLabel("OpenAI API key", { exact: true }).fill("sk-synthetic-first-use");
    await page.getByRole("button", { name: "Save API key", exact: true }).click();
    await expect(page.getByLabel("OpenAI API key", { exact: true })).toBeEmpty();
    await expect
      .poll(async () => (await request.get("/__test/state")).json())
      .toMatchObject({
        credential_saved: true,
        conversation: { connectionState: "CONNECTION_STATE_CONNECTED" },
        rpc_calls: { SetSettingsApiKey: 1, RestartConversation: 1 },
      });
    await page.getByRole("link", { name: "Conversation", exact: true }).click();
    await expect(page.getByRole("button", { name: "Mute microphone", exact: true })).toBeEnabled();
  });
});
