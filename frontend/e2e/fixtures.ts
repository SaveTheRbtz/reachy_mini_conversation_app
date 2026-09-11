import { test as base, expect, type Page } from "@playwright/test";
import { startBackend, type TestBackend } from "../tests/support/backend.ts";

export const test = base.extend<{ backend: TestBackend; backendArgs: string[]; pageErrors: void }>({
  backendArgs: [[], { option: true }],
  backend: async ({ backendArgs }, use) => {
    const server = await startBackend(backendArgs);
    try {
      await use(server);
    } finally {
      await server.stop();
    }
  },
  baseURL: async ({ backend }, use) => {
    await use(backend.origin);
  },
  pageErrors: [
    async ({ page }, use) => {
      const errors: string[] = [];
      page.on("pageerror", (error) => errors.push(error.message));
      await use();
      expect(errors).toEqual([]);
    },
    { auto: true },
  ],
});

export async function expectVisibleImages(page: Page): Promise<void> {
  await expect
    .poll(() =>
      page.evaluate(() =>
        Array.from(document.images)
          .filter((image) => {
            const bounds = image.getBoundingClientRect();
            return bounds.bottom > 0 && bounds.top < innerHeight;
          })
          .every((image) => image.complete && image.naturalWidth > 0),
      ),
    )
    .toBe(true);
}

export { expect };
