import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig(({ command }) => ({
  root: "frontend",
  base: command === "build" ? "/static/" : "/",
  plugins: [vue()],
  server: { proxy: { "/rpc": "http://127.0.0.1:7860" } },
  build: {
    outDir: fileURLToPath(new URL("./src/reachy_mini_conversation_app/static", import.meta.url)),
    emptyOutDir: true,
    target: "es2022",
    rolldownOptions: {
      output: {
        entryFileNames: "assets/[hash].js",
        chunkFileNames: "assets/[hash].js",
        assetFileNames: "assets/[hash][extname]",
      },
    },
  },
}));
