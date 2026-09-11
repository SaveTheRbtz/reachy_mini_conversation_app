import { execFileSync } from "node:child_process";

const outputDirectories = [
  "src/reachy_mini_conversation_app/gen",
  "frontend/src/gen",
  "src/reachy_mini_conversation_app/static/js",
];
for (const command of ["lint", "generate"]) {
  execFileSync(process.execPath, ["node_modules/@bufbuild/buf/bin/buf", command], { stdio: "inherit" });
}
execFileSync(process.execPath, ["--experimental-strip-types", "frontend/build.ts"], { stdio: "inherit" });
const changes = execFileSync("git", [
  "status", "--porcelain", "--untracked-files=all", "--", ...outputDirectories,
], { encoding: "utf8" });
if (changes) {
  console.error("Commit the generated API and browser code with their sources:\n" + changes);
  process.exitCode = 1;
}
