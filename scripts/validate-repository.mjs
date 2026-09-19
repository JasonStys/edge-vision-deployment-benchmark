/** File: Enforces required documentation, source headers, artifact hashes, size limits, and action pins.
 * Functions: walk, record, and main validation logic collect actionable policy failures. Variables:
 * requiredDocuments, sourceExtensions, exclusions, and problems define the repository contract.
 */
import { createHash } from "node:crypto";
import { access, readFile, readdir, stat } from "node:fs/promises";
import { relative, resolve } from "node:path";

const repository = resolve(import.meta.dirname, "..");
const requiredDocuments = [
  "README.md", "CONTRIBUTING.md", "SECURITY.md", "docs/REQUIREMENTS.md",
  "docs/ARCHITECTURE.md", "docs/adr/0001-portable-model-and-onnx.md", "docs/DATASET_CARD.md",
  "docs/BENCHMARKING.md", "docs/TESTING.md", "docs/OPERATIONS.md", "docs/SECURITY.md",
  "docs/LIMITATIONS.md", "docs/RESEARCH.md", "docs/COMPLEXITY.md", "docs/DEPLOYMENT_CHECKLIST.md",
  "docs/FILE_CATALOG.md", "docs/CODE_INDEX.md", "docs/reports/VALIDATION.md",
  "docs/reports/TEST_SUMMARY.md", "docs/reports/PERFORMANCE.md",
];
const sourceExtensions = /(?:\.py|\.mjs|\.sh|\.cpp|\.hpp|\.rs|\.yml|\.yaml|CMakeLists\.txt|Dockerfile)$/;
const exclusions = new Set([
  ".git", ".venv", ".runtime", ".mypy_cache", ".ruff_cache", ".pytest-runtime",
  ".pytest_cache", "build", "target", "__pycache__", "htmlcov",
]);
const problems = [];

async function walk(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (exclusions.has(entry.name) || entry.name.startsWith("build-")) continue;
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) files.push(...await walk(path));
    else files.push(path);
  }
  return files;
}

function record(condition, message) {
  if (!condition) problems.push(message);
}

for (const document of requiredDocuments) {
  try { await access(resolve(repository, document)); } catch { problems.push(`missing required document: ${document}`); }
}

const files = await walk(repository);
for (const file of files) {
  const path = relative(repository, file).replaceAll("\\", "/");
  const metadata = await stat(file);
  record(metadata.size <= 2_000_000, `repository file exceeds 2 MB: ${path}`);
  if (sourceExtensions.test(path)) {
    const prefix = (await readFile(file, "utf8")).split(/\r?\n/).slice(0, 8).join("\n");
    record(/File:/i.test(prefix), `missing File header in ${path}`);
  }
  if (path.startsWith(".github/workflows/") && path.endsWith(".yml")) {
    const content = await readFile(file, "utf8");
    for (const line of content.split(/\r?\n/).filter((candidate) => /uses:\s*/.test(candidate))) {
      record(
        /uses:\s*[\w.-]+\/[\w.-]+(?:\/[\w.-]+)?@[0-9a-f]{40}(?:\s+#\s*v?[0-9])/i.test(line),
        `workflow action is not pinned to a full SHA: ${path}: ${line.trim()}`,
      );
    }
  }
}

const searchable = (await Promise.all(files.map((file) => readFile(file, "utf8").catch(() => "")))).join("\n");
const unresolvedMarkers = [new RegExp("\\bTO" + "DO\\b"), new RegExp("\\bFIX" + "ME\\b")];
record(!unresolvedMarkers.some((pattern) => pattern.test(searchable)), "unresolved task marker found");

try {
  const manifest = JSON.parse(await readFile(resolve(repository, "artifacts/manifest.json"), "utf8"));
  for (const [relativePath, expected] of Object.entries(manifest.files ?? {})) {
    const artifact = resolve(repository, relativePath);
    const content = await readFile(artifact);
    const metadata = await stat(artifact);
    record(metadata.size === expected.bytes, `artifact size mismatch: ${relativePath}`);
    record(createHash("sha256").update(content).digest("hex") === expected.sha256, `artifact hash mismatch: ${relativePath}`);
  }
} catch (error) {
  problems.push(`artifact manifest validation failed: ${error.message}`);
}

if (problems.length) {
  console.error(problems.join("\n"));
  process.exitCode = 1;
} else {
  console.log("repository policy validation passed");
}
