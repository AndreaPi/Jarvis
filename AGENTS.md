# Repository Guidelines

## Scope
- This file covers repo-wide guidance only.
- Backend-specific runtime, training, and API instructions live in `backend/AGENTS.md`.
- OCR-specific behavior, benchmarks, and tuning policy live in `src/ocr/AGENTS.md`.

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
- `npm run serve`: Start the local web server on port `8000`.
- `npm run dev`: Alias of `npm run serve`.
- `npm run test:e2e`: Run Playwright end-to-end tests.
- `npm run benchmark:roi-diff`: Generate ROI checkpoint diff artifacts.

Open `http://localhost:8000` after starting the frontend server.

## Coding Style & Naming Conventions
- Use 2-space indentation in HTML/CSS/JS.
- Keep files ASCII-only unless there is a strong reason for Unicode.
- Use descriptive, lower-case IDs and class names (for example `photo-input`, `module-grid`).
- Prefer clear, small functions in `src/` modules and avoid deep nesting.

## Testing Guidelines
- Automated browser tests are configured with Playwright.
- CI: `.github/workflows/e2e.yml` runs on each pull request and on pushes to `master`.
- Frontend manual checks: upload an image, run OCR, verify the email draft fields, and confirm the Gmail draft link.
- Backend sanity checks: `GET /health` and confirm `ready: true`, `roi_ready: true`, and the expected `model_path`.
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
- Treat raw `assets/` photos, `assets/meter_readings.csv`, ROI images/labels, digit manifests, and promoted `backend/models/*.pt` checkpoints as must-retain artifacts.
- Use DVC for large Tier 1 binaries:
  - per-file DVC tracking for raw meter photos in `assets/`
  - `backend/data/roi_dataset/images.dvc`
  - `backend/data/digit_dataset/windows.dvc`
  - `backend/data/digit_dataset/windows_canonical.dvc`
  - `backend/data/digit_dataset/sections.dvc`
  - `backend/data/digit_dataset/sections_labeled.dvc`
  - `backend/data/digit_dataset/sections_synthetic/train.dvc`
  - per-file DVC tracking for promoted `backend/models/*.pt`
- After dataset ingestion or model promotion, run the relevant `dvc add ...` commands, then push with `scripts/dvc-push-safe.sh`.
- Never run raw `dvc push` directly in this repo. Always use `scripts/dvc-push-safe.sh`; it requires a configured non-local DVC remote and refuses to push to plain local paths.
- For cloud storage, prefer Backblaze B2. Install `dvc[s3]` in `backend/.venv` and configure the DVC remote through B2's S3-compatible endpoint.
- Use `scripts/package-tier1-artifacts.sh` plus the manual `Publish Artifacts` workflow for release-style snapshots after the DVC remote is up to date.

## Important
- When using Playwright in this environment, global `playwright-cli` may be more reliable than the wrapper if npm network is flaky.
- In this Codex environment, long-running local services that must be consumed by the DevTools browser may need to be started with escalated permissions instead of inside the sandbox.
- If shell `curl` works but the browser still gets `ERR_CONNECTION_REFUSED` or `Failed to fetch`, verify connectivity from the page context and restart the service outside the sandbox.
