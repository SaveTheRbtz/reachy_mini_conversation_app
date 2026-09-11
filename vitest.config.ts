import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: "unit",
          include: ["frontend/tests/unit/**/*.test.ts"],
          restoreMocks: true,
        },
      },
      {
        test: {
          name: "integration",
          include: ["frontend/tests/integration/**/*.test.ts"],
          testTimeout: 30000,
          hookTimeout: 20000,
        },
      },
    ],
  },
});
