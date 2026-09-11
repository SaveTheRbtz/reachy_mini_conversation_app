import { rmSync } from "node:fs";
import { build } from "esbuild";

const outdir = "src/reachy_mini_conversation_app/static/js";
rmSync(outdir, { recursive: true, force: true });
await build({
  entryPoints: ["frontend/src/main.ts"],
  outdir,
  bundle: true,
  format: "esm",
  target: "es2022",
  minify: true,
});
