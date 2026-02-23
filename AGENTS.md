# Repository Guidelines

## Project Structure & Module Organization
- `index.html`: Single-page UI layout and content.
- `styles.css`: Global styles and visual system.
- `app.js`: Thin module entrypoint that imports `src/main.js`.
- `src/main.js`: UI orchestration and event wiring.
- `src/ocr/`: OCR pipeline, image processing modules, and optional neural ROI client.
- `src/email/`: Email draft generation and link helpers.
- `src/testset/`: Manual test-set runner logic.
- `src/debug/`: Debug overlay rendering helpers.
- `backend/`: Optional FastAPI service for neural ROI + digit classifier inference and training scripts.
- `backend/build_digit_dataset.py`: Export strip/cell OCR datasets + QA previews from ROI labels.
- `backend/plan_digit_expansion.py`: Generate prioritized capture plan for underrepresented digits.
- `backend/validate_digit_dataset.py`: Validate manifest consistency and QA preview coverage.
- `backend/train_digit_classifier.py`: Train per-cell digit classifier checkpoint.
- `package.json`: Local dev scripts.
- `README.md`: Project overview and setup notes.
- `assets/`: Static assets and example uploads.

## Build, Test, and Development Commands
- `npm run serve`: Start a simple local web server on port 8000.
- `npm run dev`: Alias of `npm run serve`.
- `cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`: Backend setup.
- `cd backend && source .venv/bin/activate && python train_roi.py --data data/roi_dataset.yaml --base-model yolov8n.pt`: Fine-tune pretrained ROI detector.
- `cd backend && source .venv/bin/activate && python build_digit_dataset.py --clean`: Rebuild digit strip/cell exports and QA previews.
- `cd backend && source .venv/bin/activate && python validate_digit_dataset.py`: Validate dataset/manifests before training.
- `cd backend && source .venv/bin/activate && python plan_digit_expansion.py --target-train-per-digit 12 --priority-digits 4,5,6,9`: Refresh targeted capture checklist.
- `cd backend && source .venv/bin/activate && python train_digit_classifier.py --device cpu`: Train per-cell digit classifier model.
- `cd backend && source .venv/bin/activate && uvicorn app:app --host 127.0.0.1 --port 8001 --reload`: Run neural ROI API.

Open `http://localhost:8000` after running a serve command. Backend endpoints default to `http://127.0.0.1:8001/roi/detect` and `http://127.0.0.1:8001/digit/predict-cells`.

## Coding Style & Naming Conventions
- Use 2-space indentation in HTML/CSS/JS.
- Keep files ASCII-only unless there is a strong reason for Unicode.
- Use descriptive, lower-case IDs and class names (e.g., `photo-input`, `module-grid`).
- Prefer clear, small functions in `src/` modules and avoid deep nesting.

## Testing Guidelines
- Automated browser tests are configured with Playwright.
- `npm run test:e2e`: Runs `tests/e2e/neural-roi.spec.js` (neural ROI failure handling + success-path coverage).
- CI: `.github/workflows/e2e.yml` runs on each pull request and on pushes to `master`.
- Frontend manual checks: upload image, run OCR, verify email draft fields, and confirm Gmail draft link.
- OCR test-set checks: run "Run test set" and inspect `Value Match`, `Failure Reason`, and debug stages.
- Backend sanity checks: `GET /health` and confirm `ready: true`, `roi_ready: true`, and expected `model_path`.
- Prefer running the test set from UI with debug overlay enabled.

## Commit & Pull Request Guidelines
- No commit message convention is established in this repo.
- Suggested pattern: short, imperative subject (e.g., "Improve OCR preview").
- PRs should include: summary of changes, screenshots for UI changes, and any manual test notes.

## Security & Configuration Tips
- The Gmail draft flow opens a client-side draft; no credentials are stored in code.
- OCR runs in the browser; avoid adding API keys to the client without a secure proxy.
- Backend is intended for local use; keep host/CORS scoped to localhost unless explicitly deploying.

## IMPORTANT
- When using Playwright in this environment, global `playwright-cli` may be more reliable than the wrapper if npm network is flaky.

## OCR Working State

- App + backend run locally on `127.0.0.1:8000` and `127.0.0.1:8001`.
- Neural ROI is mandatory in the frontend OCR flow (heuristic ROI fallback removed).
- On neural ROI failure, the UI shows an explicit reason and asks for manual measurement input.
- Backend default ROI model is pinned to `backend/models/roi-rotaug-e30-640.pt` (override with `ROI_MODEL_PATH`).
- Digit-classifier inference is optional behind `OCR_CONFIG.digitClassifier.enabled` (default `false`).
- Backend serves ROI + digit endpoints and reports readiness via `GET /health`.
- Test-set table includes `Detected`, `Value Match`, `Failure Reason`, and `Result`.

## OCR Priorities

1. Keep neural-ROI-only policy while improving decoded value reliability.
2. Tune ROI crop geometry (`expandX`, `expandY`) and strip normalization gates to reduce `candidate-bad-aspect`.
3. Review ROI cell-decoding strictness (`requireAllCells` and split assumptions) to lower `missing-cell-digit`.
4. Re-run UI `Run test set` after each tuning change and track both `Detected` coverage and mismatch rate.
5. Use selection logs (`window.__jarvisOcrSelectionLogs`) and `Failure Reason` output to guide each iteration.

## Dataset Expansion Loop (`4/5/6/9`)

1. Refresh capture planning:
   - `cd backend && source .venv/bin/activate && python plan_digit_expansion.py --target-train-per-digit 12 --priority-digits 4,5,6,9`
2. Add labeled captures with QA previews.
3. Validate manifests after each dataset update:
   - `cd backend && source .venv/bin/activate && python validate_digit_dataset.py`
4. Retrain classifier only after class coverage improves:
   - `cd backend && source .venv/bin/activate && python train_digit_classifier.py --device cpu`
5. Re-enable classifier only as a gated fallback path, then re-run OCR test-set benchmarks.
