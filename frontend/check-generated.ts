import { execFileSync } from "node:child_process";
import { rmSync } from "node:fs";

const outputDirectory = "src/reachy_mini_conversation_app/static/js";
rmSync(outputDirectory, { recursive: true, force: true });
execFileSync(process.execPath, ["node_modules/typescript/bin/tsc"], { stdio: "inherit" });
const changes = execFileSync("git", [
  "status", "--porcelain", "--untracked-files=all", "--", outputDirectory,
], { encoding: "utf8" });
if (changes) {
  console.error("Commit the generated browser modules with their TypeScript sources:\n" + changes);
  process.exitCode = 1;
}
