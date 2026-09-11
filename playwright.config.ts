import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./frontend/e2e",
  timeout: 60000,
  expect: { timeout: 10000 },
  workers: 1,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    trace: process.env.E2E_RECORD === "1" ? "on" : "retain-on-failure",
    video: process.env.E2E_RECORD === "1" ? "on" : "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1280, height: 900 } } },
    { name: "mobile", grep: /@mobile/, use: { ...devices["Pixel 7"] } },
  ],
});
