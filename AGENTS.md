# Repository Guidelines

## Scope
- This file covers repo-wide guidance only.
- Backend-specific runtime, training, and API instructions live in `backend/AGENTS.md`.
- OCR-specific behavior, benchmarks, and tuning policy live in `src/ocr/AGENTS.md`.

## Task Scope and Completion
- Read the guidance and source relevant to the requested change; a small edit does not require a full repository or documentation audit.
- Follow the user's requested scope and existing session authorization. Continue routine authorized work through implementation and relevant validation; do not add a first-draft approval stop.
- Reuse human annotation review for the same unchanged source and annotation. For user-supplied manual ROI and digit-box exports, verified equivalent import plus agent visual QA satisfies the corresponding review gate without a second human preview confirmation; follow the relevant ingestion skill for checks and exceptions. Automatically generated boxes still require human Make Sense review. Changed source/annotation geometry or unresolved QA requires renewed review. ROI and digit-box review remain distinct, and artifact-retention and model-promotion gates still apply.
- When human input is pending, finish independent authorized checks and report the prepared result, pending decision, and next step. Do not describe pending annotations as training-ready.

## Project Structure & Module Organization
- `index.html`: Single-page UI layout and content.
- `styles.css`: Global styles and visual system.
- `app.js`: Thin module entrypoint that imports `src/main.js`.
- `src/main.js`: UI orchestration and event wiring.
- `src/ocr/`: OCR pipeline and selection logic. See `src/ocr/AGENTS.md`.
- `src/email/`: Email draft generation and link helpers.
- `src/testset/`: Manual test-set runner logic.
- `src/debug/`: Debug overlay rendering helpers.
- `backend/`: FastAPI service, training scripts, and model/data tooling. See `backend/AGENTS.md`.
- `package.json`: Local dev scripts.
- `README.md`: Project overview and setup notes.
- `assets/`: Static assets and example uploads.

## Build, Test, and Development Commands
- `npm run jarvis:start`: Start the local frontend and canonical OCR backend, verify readiness, and open Jarvis.
- `npm run jarvis:status`: Report whether both local Jarvis services are ready.
- `npm run jarvis:stop`: Stop only the local services created by the Jarvis launcher.
- `npm run serve`: Start the local web server on port `8000`.
- `npm run dev`: Alias of `npm run serve`.
- `npm run test:scripts`: Run launcher, QA service guard, and artifact/DVC safety tests.
- `npm run test:backend`: Auto-discover and run the fast Python tests in `backend/test_*.py`.
- `npm run test:e2e`: Run Playwright end-to-end tests.
- `npm run benchmark:roi-diff`: Generate ROI checkpoint diff artifacts.

Open `http://localhost:8000` after starting the frontend server.

## Coding Style & Naming Conventions
- Use 2-space indentation in HTML/CSS/JS.
- Keep files ASCII-only unless there is a strong reason for Unicode.
- Use descriptive, lower-case IDs and class names (for example `photo-input`, `module-grid`).
- Prefer clear, small functions in `src/` modules and avoid deep nesting.

## Testing Guidelines
- `test:backend` contains fast unit, component, and confirmed-regression tests. Mock model inference and use temporary directories; model training and full checkpoint benchmarks do not belong in this suite.
- `test:scripts` protects launcher, service-guard, DVC, and artifact-safety contracts.
- `test:e2e` contains Playwright browser integration and user-flow regressions; mock the backend unless the behavior specifically requires a live service.
- `qa:*` commands and the UI `Run test set` are model/data benchmarks, not automated code-test cases. Their image counts must not be added to the automated test count.
- Add an automated test when it protects durable behavior, retained data/artifacts, a user-facing workflow, or a confirmed regression. Prefer one table-driven test over several near-identical helper tests, and keep one-off experiment/report checks in the relevant QA workflow.
- Run checks appropriate to the changed behavior. Documentation-only edits need source/link validation and `git diff --check`; they do not trigger OCR benchmarks merely because they describe OCR. After relevant checks pass, repeat them only for further changes, failures, or unresolved concerns.
- CI: `.github/workflows/e2e.yml` runs on each pull request and on pushes to `master`.
- Frontend manual checks: upload an image, run OCR, verify the email draft fields, and confirm the Gmail draft link.
- Backend sanity checks: `GET /health` and confirm `ready: true`, `roi_ready: true`, `digit_ready: true`, `strip_digit_ready: true`, and the expected model paths when all checkpoints are present.
- For OCR changes, run both `npm run test:e2e` and the UI `Run test set`. See `src/ocr/AGENTS.md` for the active benchmark baseline and promotion guardrails.

## Commit & Pull Request Guidelines
- No commit message convention is established in this repo.
- Suggested pattern: short, imperative subject (for example `Improve OCR preview`).
- PRs should include a summary of changes, screenshots for UI changes, and any manual test notes.

## Security & Configuration Tips
- The Gmail draft flow opens a client-side draft; no credentials are stored in code.
- OCR runs in the browser; avoid adding API keys to the client without a secure proxy.
- Backend is intended for local use; keep host/CORS scoped to localhost unless explicitly deploying.

## Artifact Retention
- Treat canonical `assets/` meter photos, `assets/meter_readings.csv`, ROI images/labels, `backend/data/roi_dataset/splits.json`, digit manifests (including `backend/data/full_image_digit_dataset/manifests/**`), and promoted `backend/models/*.pt` checkpoints as must-retain artifacts.
- Use DVC for large Tier 1 binaries:
  - per-file DVC tracking for canonical meter photos in `assets/`
  - `backend/data/roi_dataset/images.dvc`
  - `backend/data/digit_dataset/windows.dvc`
  - `backend/data/digit_dataset/windows_canonical.dvc`
  - `backend/data/digit_dataset/sections.dvc`
  - `backend/data/digit_dataset/sections_labeled.dvc`
  - `backend/data/digit_dataset/sections_synthetic/train.dvc`
  - per-file DVC tracking for promoted `backend/models/*.pt`
- After dataset ingestion or model promotion, run the relevant `dvc add ...` commands, then push with `scripts/dvc-push-safe.sh`.
- For a meter-photo batch, reuse one explicit DVC upload authorization across ROI and digit-box ingestion. Its scope must identify the canonical filenames, the resolved remote destination, and the included artifacts: source photos, ROI images, and the batch's derived windows, canonical windows, sections, and labeled sections. Include downstream derivatives only when that workflow is requested.
- If authorization is missing, prepare and validate the artifacts for the first upload, then ask once for the full requested batch scope, including any named derivatives still to be generated and reviewed. Record the user's authorization and its scope in batch QA evidence and pass it to the downstream skill; a QA record preserves consent but does not create it.
- Reuse that authorization for subsequent uploads of the covered artifacts to the same destination, including on resume. Ask only for uncovered scope, such as additional photos, artifact categories, or a changed destination; honor narrower consent and revocation. Verify each upload contains only authorized changes. Annotation/orientation review gates still apply, and batch upload consent does not authorize training, model promotion, release publishing, or Git commit/push. These instructions do not bypass execution approval controls.
- Never run raw `dvc push` directly in this repo. Always use `scripts/dvc-push-safe.sh`; it requires a configured non-local DVC remote and refuses plain local paths and `file://` URLs.
- For cloud storage, prefer Backblaze B2. Install `dvc[s3]` in `backend/.venv` and configure the DVC remote through B2's S3-compatible endpoint.
- Use `scripts/package-tier1-artifacts.sh` plus the manual `Publish Artifacts` workflow for release-style snapshots after the DVC remote is up to date.

## Important
- When using Playwright in this environment, global `playwright-cli` may be more reliable than the wrapper if npm network is flaky.
- In this Codex environment, long-running local services that must be consumed by the DevTools browser may need to be started with escalated permissions instead of inside the sandbox.
- If shell `curl` works but the browser still gets `ERR_CONNECTION_REFUSED` or `Failed to fetch`, verify connectivity from the page context and restart the service outside the sandbox.
