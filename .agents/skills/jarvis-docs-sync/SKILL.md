---
name: jarvis-docs-sync
description: "Audit or sync Jarvis documentation when requested or when a change affects documented behavior, commands, or workflows."
---

# Jarvis Docs Sync

Run this workflow from the Jarvis repository root.

Use this skill for a requested documentation audit/sync or a change with a
concrete effect on documented behavior, commands, or workflows. An ordinary
edit does not require a documentation audit. For audit-only requests, report
findings without editing; for requested updates, carry relevant edits through
validation without pausing for a first-draft review.

## Scope

Start with the changed fact and its affected documentation. Use the full
inventory below only for a repo-wide documentation audit:

- `AGENTS.md`
- nested `**/AGENTS.md` files that carry active project guidance (currently `backend/AGENTS.md` and `src/ocr/AGENTS.md`)
- `README.md`
- `backend/README.md`
- `docs/**/*.md`

Also check local skill docs only if they are directly affected by the change being documented.
Check linked local images or generated diagrams when their source description changed.

## Workflow

1. Build the current repo facts from source, not memory.
   - Read only the source needed to establish the affected facts. The following
     mapping is a lookup aid, not a mandatory reading list:
     - `package.json` and `playwright.config.*` -> local test commands and test scope
     - `.github/workflows/*.yml` -> CI-only claims
     - `backend/app.py` -> endpoints, environment variables, limits, readiness, and errors
     - `src/ocr/config.js` and `src/ocr/pipeline.js` -> active OCR behavior and guardrails
     - dataset manifests and `assets/meter_readings.csv` -> current counts and split claims
     - safety scripts under `scripts/` -> DVC, QA startup, and artifact-retention instructions

2. Compare the docs against those facts.
   - Look for stale commands, ports, file paths, model names, dataset counts, workflow steps, artifact-retention rules, and benchmark expectations.
   - Use `rg` to find repeated claims across docs before editing.
   - Separate active claims from dated history. Keep historical benchmark snapshots intact when they are clearly dated or described as `then-current`; update wording that incorrectly presents an old snapshot as current.
   - Do not hard-code volatile active corpus or split counts in operational docs when they can be derived from manifests. Keep exact counts only when they are part of a dated benchmark/result or a stable invariant.

3. Update only what is actually stale.
   - Do not rewrite docs for style alone unless the user requested that scope.
   - Preserve existing structure and tone unless the current structure is actively misleading.
   - Keep documentation concise and operational.

4. Keep cross-file consistency.
   - If a command, path, or rule changes, update other docs stating the same
     fact within the user-authorized scope. For an explicit single-file or
     bounded edit, report affected references outside that scope without editing them.
   - Pay special attention to duplicated operational guidance in root and nested `AGENTS.md` files, `README.md`, and `backend/README.md`.

5. Validate after editing.
   - Re-run `rg` for the old value to make sure stale references are gone where appropriate.
   - Confirm commands, paths, and filenames in the affected passages exist.
   - If those passages mention tests or benchmarks, verify their entry points.
   - Run `git diff --check`.
   - Do not infer a passing test count by counting declarations. Execute the corresponding suite before adding or refreshing a `passes (N/N)` claim.
   - Treat UI OCR metrics as verified only when a fresh **Run test set** result is available. Otherwise remove or soften the current claim instead of copying an older number forward.
   - If a Markdown diagram has a checked-in rendered image, update both or leave the diagram source unchanged.
   - Documentation-only edits normally need source/link checks and
     `git diff --check`, not application test suites or OCR benchmarks. Run a
     suite when refreshing its result claim or changing executable behavior.
     Once relevant checks pass, repeat them only for subsequent changes or
     unresolved failures.

6. Report the audit boundary.
   - List the documents changed and the primary facts used.
   - Call out checks that remain local-only or claims that could not be re-verified.
   - Do not run training, DVC push, artifact publishing, dataset ingestion, or other state-changing workflows merely to validate documentation.

## Useful Commands

Use only the commands relevant to the affected facts.

- Package scripts:
  - `cat package.json`
- Workflow inventory:
  - `find .github/workflows -maxdepth 1 -type f | sort`
- Markdown inventory:
  - `find docs -type f -name '*.md' | sort`
- AGENTS inventory:
  - `rg --files --hidden -g AGENTS.md -g '!node_modules' -g '!.git' -g '!.venv'`
- Local skill inventory:
  - `find .agents/skills -type f -name 'SKILL.md' | sort`
- Current corpus and ROI split facts:
  - `awk 'END { print NR - 1 }' assets/meter_readings.csv`
  - `backend/.venv/bin/python -c "import collections,json; data=json.load(open('backend/data/roi_dataset/splits.json'))['assignments']; print(collections.Counter(data.values()))"`
- Configured test entry points:
  - `npm pkg get 'scripts.test:scripts' 'scripts.test:backend' 'scripts.test:e2e'`
- Validation:
  - `git diff --check`
- Search stale values:
  - `rg -n "<old-term>|<old-path>|<old-command>" AGENTS.md backend/AGENTS.md src/ocr/AGENTS.md README.md backend/README.md docs .agents/skills`
- Search current values:
  - `rg -n "<new-term>|<new-path>|<new-command>" AGENTS.md backend/AGENTS.md src/ocr/AGENTS.md README.md backend/README.md docs .agents/skills`

## Notes

- Prefer fixing the docs in the same change set as the code change that made them stale.
- If the code is ambiguous, resolve the ambiguity from the implementation before editing docs.
- If a statement cannot be verified from the repo, remove or soften it instead of leaving a hard claim.
- Avoid bare words such as `current` next to old dates or corpus sizes; use `historical`, `then-current`, or point to the active baseline source.
