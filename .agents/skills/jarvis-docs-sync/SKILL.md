---
name: jarvis-docs-sync
description: "Audit and update Jarvis project documentation so it matches the current repository state. Use when root or nested AGENTS.md files, README.md, backend/README.md, or files under docs/ may be stale after code, config, workflow, model, or dataset-process changes."
---

# Jarvis Docs Sync

Run this workflow from the Jarvis repository root.

Use this skill whenever the repo has changed and documentation may now be inaccurate.

## Scope

Check these documentation targets:

- `AGENTS.md`
- nested `**/AGENTS.md` files that carry active project guidance (currently `backend/AGENTS.md` and `src/ocr/AGENTS.md`)
- `README.md`
- `backend/README.md`
- `docs/**/*.md`

Also check local skill docs only if they are directly affected by the change being documented.

## Workflow

1. Build the current repo facts from source, not memory.
   - Inspect the code, scripts, config, package scripts, workflow files, and model/data paths that define current behavior.
   - Prefer primary sources such as:
     - `package.json`
     - `playwright.config.*`
     - `src/**/*.js`
     - `backend/app.py`
     - `backend/*.py`
     - `.github/workflows/*.yml`
     - `.dvc/config*`
     - `backend/models/*.dvc`

2. Compare the docs against those facts.
   - Look for stale commands, ports, file paths, model names, dataset counts, workflow steps, artifact-retention rules, and benchmark expectations.
   - Use `rg` to find repeated claims across docs before editing.

3. Update only what is actually stale.
   - Do not rewrite docs for style alone.
   - Preserve existing structure and tone unless the current structure is actively misleading.
   - Keep documentation concise and operational.

4. Keep cross-file consistency.
   - If a command, path, or rule changes in one doc, update all other docs that state the same fact.
   - Pay special attention to duplicated operational guidance in root and nested `AGENTS.md` files, `README.md`, and `backend/README.md`.

5. Validate after editing.
   - Re-run `rg` for the old value to make sure stale references are gone where appropriate.
   - Confirm every referenced command, path, and filename exists.
   - If documentation mentions tests or benchmarks, verify the names and entry points still match the repo.

## Audit Checklist

Always verify these categories when relevant:

- Dev/test commands and required environments
- Frontend and backend ports/endpoints
- OCR pipeline behavior and guardrails
- Model filenames and default checkpoint paths
- Dataset rebuild commands and artifact-retention process
- DVC usage and backup expectations
- CI workflows and local validation steps
- Skill-specific workflows under `.agents/skills/`

## Useful Commands

- Package scripts:
  - `cat package.json`
- Workflow inventory:
  - `find .github/workflows -maxdepth 1 -type f | sort`
- Markdown inventory:
  - `find docs -type f -name '*.md' | sort`
- AGENTS inventory:
  - `find . -type f -name 'AGENTS.md' | sort`
- Search stale values:
  - `rg -n "<old-term>|<old-path>|<old-command>" AGENTS.md backend/AGENTS.md src/ocr/AGENTS.md README.md backend/README.md docs .agents/skills`
- Search current values:
  - `rg -n "<new-term>|<new-path>|<new-command>" AGENTS.md backend/AGENTS.md src/ocr/AGENTS.md README.md backend/README.md docs .agents/skills`

## Notes

- Prefer fixing the docs in the same change set as the code change that made them stale.
- If the code is ambiguous, resolve the ambiguity from the implementation before editing docs.
- If a statement cannot be verified from the repo, remove or soften it instead of leaving a hard claim.
