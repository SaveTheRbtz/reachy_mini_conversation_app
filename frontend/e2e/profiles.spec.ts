import { test, expect, expectVisibleImages } from "./fixtures.ts";

test("personality CRUD preserves tool presence, deep links, and keyboard dialogs @mobile", async ({
  page,
  request,
}, testInfo) => {
  await page.goto("/profiles");
  await expect(page.getByRole("heading", { level: 1, name: "Personalities" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Default", exact: true })).toBeVisible();
  await expectVisibleImages(page);
  await page.screenshot({ path: testInfo.outputPath("personalities.png") });
  await page.getByLabel("Search personalities", { exact: true }).fill("captain");
  await expect(page.getByRole("link", { name: "Captain Circuit", exact: true })).toBeVisible();
  await expect(page.getByRole("link", { name: "Default", exact: true })).toHaveCount(0);
  await page.getByLabel("Search personalities", { exact: true }).fill("");
  await page.getByRole("link", { name: "Create personality", exact: true }).click();
  await page.getByRole("textbox", { name: /^Name/ }).fill("Browser guide");
  await page.getByLabel("Instructions", { exact: true }).fill("Keep conversations clear and friendly.");
  await page.getByLabel("Greeting", { exact: true }).fill("Hello from your browser guide.");
  await page.getByRole("button", { name: "Create personality", exact: true }).click();
  await expect(page).toHaveURL(/\/profiles\/user-browser-guide$/);
  await expect(page.getByRole("link", { name: "Personalities", exact: true })).toHaveClass(/\bactive\b/);
  await expect(page.getByRole("heading", { level: 1, name: "Browser Guide" })).toBeVisible();
  await page.reload();
  await expect(page.getByLabel("Instructions", { exact: true })).toHaveValue(
    "Keep conversations clear and friendly.",
  );
  await page.screenshot({ path: testInfo.outputPath("personality-editor.png") });
  await page.getByLabel("Instructions", { exact: true }).fill("Unsaved draft");
  await page.getByRole("link", { name: "All personalities" }).click();
  const discard = page.getByRole("dialog", { name: "Discard unsaved changes?" });
  await expect(discard).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(discard).not.toBeVisible();
  await expect(page.getByRole("link", { name: "All personalities" })).toBeFocused();
  await expect(page.getByLabel("Instructions", { exact: true })).toHaveValue("Unsaved draft");
  await page.getByRole("link", { name: "All personalities" }).click();
  await discard.getByRole("button", { name: "Discard changes", exact: true }).click();
  await expect(page).toHaveURL(/\/profiles$/);
  await page.goto("/profiles/user-browser-guide");
  await expect(page.getByLabel("Instructions", { exact: true })).toHaveValue(
    "Keep conversations clear and friendly.",
  );
  await page.getByRole("checkbox", { name: /^Use profile defaults/ }).uncheck();
  await page.getByRole("button", { name: "Disable all", exact: true }).click();
  await page.getByRole("button", { name: "Save changes", exact: true }).click();
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({
      toolsets: { "user_personalities/browser-guide": [] },
      effective_tools: { "profiles/user-browser-guide": [] },
    });
  await page.reload();
  await expect(page.getByRole("checkbox", { name: /^Use profile defaults/ })).not.toBeChecked();
  await page.getByRole("checkbox", { name: /^Use profile defaults/ }).check();
  await page.getByRole("button", { name: "Save changes", exact: true }).click();
  await expect(page.getByRole("button", { name: "Save changes", exact: true })).toBeDisabled();
  await page.reload();
  await expect(page.getByRole("checkbox", { name: /^Use profile defaults/ })).toBeChecked();
  await page.getByRole("button", { name: "Use personality", exact: true }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({
      conversation: { profile: "profiles/user-browser-guide" },
    });
  await page.goto("/profiles/builtin-default");
  await page.getByRole("button", { name: "Use personality", exact: true }).click();
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({
      conversation: { profile: "profiles/builtin-default" },
    });
  await page.goto("/profiles/user-browser-guide");
  await page.getByRole("button", { name: "Delete personality", exact: true }).click();
  const deletion = page.getByRole("dialog", { name: "Delete personality?" });
  await expect(deletion.getByRole("button", { name: "Cancel", exact: true })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(deletion).not.toBeVisible();
  await expect(page.getByRole("button", { name: "Delete personality", exact: true })).toBeFocused();
  await page.getByRole("button", { name: "Delete personality", exact: true }).click();
  await deletion.getByRole("button", { name: "Delete personality", exact: true }).click();
  await expect(page).toHaveURL(/\/profiles$/);
  await expect(page.getByRole("link", { name: /Browser Guide/ })).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

test("saving active tool access stays saved when applying it fails", async ({ page, request }) => {
  await page.goto("/profiles/builtin-default");
  await expect(page.getByText("Active personality", { exact: true })).toBeVisible();
  await request.put("/__test/faults", { data: { rpc_unavailable_methods: ["RestartConversation"] } });
  await page.getByRole("checkbox", { name: /^Use profile defaults/ }).uncheck();
  await page.getByRole("button", { name: "Disable all", exact: true }).click();
  await page.getByRole("button", { name: "Save changes", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("Personality saved, but could not restart");
  await expect(page.getByRole("button", { name: "Save changes", exact: true })).toBeDisabled();
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({
      toolsets: { default: [] },
      effective_tools: { "profiles/builtin-default": [] },
      rpc_calls: { UpdateProfile: 1, RestartConversation: 1 },
    });
  await request.put("/__test/faults", { data: {} });
  await page.getByRole("button", { name: "Retry restart", exact: true }).click();
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({
      rpc_calls: { UpdateProfile: 1, RestartConversation: 2 },
    });
  await page.getByRole("link", { name: "All personalities" }).click();
  await expect(page).toHaveURL(/\/profiles$/);
  await expect(page.getByRole("dialog")).toHaveCount(0);
});
