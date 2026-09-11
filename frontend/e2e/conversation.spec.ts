import { test, expect, expectVisibleImages } from "./fixtures.ts";

test("conversation controls work with keyboard, playback, and navigation @mobile", async ({
  page,
  request,
}, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.getByText("Microphone on", { exact: true })).toBeVisible();
  const documentStartedAt = await page.evaluate(() => performance.timeOrigin);
  await expectVisibleImages(page);
  await page.screenshot({ path: testInfo.outputPath("conversation.png"), fullPage: true });
  await request.put("/__test/faults", { data: { playback_seconds: 20 } });
  await page.getByLabel("Your message").fill("Hello from the browser");
  await page.getByRole("button", { name: "Send", exact: true }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("button", { name: "Stop speaking" })).toBeEnabled();
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({
      completed_says: 1,
      sent_texts: ["Hello from the browser"],
      conversation: { playing: true },
    });
  await page.getByRole("button", { name: "Mute microphone", exact: true }).focus();
  await page.keyboard.press("Space");
  await expect(page.getByRole("button", { name: "Unmute microphone", exact: true })).toBeEnabled();
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({
      conversation: { muted: true, playing: true },
    });
  await page.getByRole("button", { name: "Stop speaking" }).click();
  await expect.poll(async () => (await request.get("/__test/state")).json()).toMatchObject({ interrupts: 1 });
  await expect(page.getByText("Microphone muted", { exact: true })).toBeVisible();
  await page.getByRole("link", { name: "Settings", exact: true }).click();
  await expect(page).toHaveURL(/\/settings$/);
  expect(await page.evaluate(() => performance.timeOrigin)).toBe(documentStartedAt);
  await expect(page.getByRole("heading", { level: 1 })).toBeFocused();
  await page.goBack();
  await expect(page).toHaveURL(/\/$/);
  expect(await page.evaluate(() => performance.timeOrigin)).toBe(documentStartedAt);
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({ active_watches: 1 });
  await expect(page.getByRole("button", { name: "Unmute microphone", exact: true })).toBeEnabled();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test("a stalled send times out without blocking microphone controls or replaying text", async ({
  page,
  request,
}) => {
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Mute microphone", exact: true })).toBeEnabled();
  await request.put("/__test/faults", { data: { send_stalled: true } });
  await page.getByLabel("Your message").fill("Keep this draft");
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({ active_says: 1 });
  await page.getByRole("button", { name: "Mute microphone", exact: true }).click();
  await expect(page.getByRole("button", { name: "Unmute microphone", exact: true })).toBeEnabled();
  await expect(page.getByRole("alert")).toContainText(/deadline|timed out|timeout/i);
  await expect(page.getByLabel("Your message")).toHaveValue("Keep this draft");
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({
      active_says: 0,
      completed_says: 0,
      worker_alive: true,
      conversation: { muted: true },
      rpc_calls: { SendConversationText: 1, UpdateConversation: 1 },
    });
  await request.put("/__test/faults", { data: {} });
  await page.getByRole("button", { name: "Send", exact: true }).click();
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({
      completed_says: 1,
      sent_texts: ["Keep this draft"],
      rpc_calls: { SendConversationText: 2 },
    });
  await expect(page.getByLabel("Your message")).toBeEmpty();
});

test("a stalled status stream recovers without reloading or repeating a mutation", async ({
  page,
  request,
}) => {
  await page.goto("/");
  await expect(page.getByText("Ready to talk", { exact: true })).toBeVisible();
  await request.put("/__test/faults", { data: { watch_stalled: true } });
  await page.getByRole("button", { name: "Mute microphone", exact: true }).click();
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({
      conversation: { muted: true },
      rpc_calls: { UpdateConversation: 1 },
    });
  await expect(page.getByText("Connecting to robot", { exact: true })).toBeVisible({ timeout: 35000 });
  await request.put("/__test/faults", { data: {} });
  await expect(page.getByText("Ready to talk", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Unmute microphone", exact: true })).toBeEnabled();
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({
      worker_alive: true,
      rpc_calls: { UpdateConversation: 1 },
    });
});

test("offline commands fail promptly and are never queued for reconnection", async ({
  page,
  request,
  context,
}) => {
  await page.goto("/");
  await expect(page.getByText("Ready to talk", { exact: true })).toBeVisible();
  await context.setOffline(true);
  await page.getByRole("button", { name: "Mute microphone", exact: true }).click();
  await expect(
    page
      .getByRole("alert")
      .filter({ hasText: /fetch|network/i })
      .first(),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "Mute microphone", exact: true })).toBeEnabled();
  await context.setOffline(false);
  await expect(page.getByText("Ready to talk", { exact: true })).toBeVisible();
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({
      rpc_calls: expect.not.objectContaining({ UpdateConversation: expect.anything() }),
    });
  await page.getByRole("button", { name: "Mute microphone", exact: true }).click();
  await expect(page.getByRole("button", { name: "Unmute microphone", exact: true })).toBeEnabled();
  await expect
    .poll(async () => (await request.get("/__test/state")).json())
    .toMatchObject({
      conversation: { muted: true },
      rpc_calls: { UpdateConversation: 1 },
    });
});
