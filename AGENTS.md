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
- No automated tests are configured.
- Frontend manual checks: upload image, run OCR, verify email draft fields, and confirm Gmail draft link.
- OCR test-set checks: run "Run test set" and inspect `Value Match` (scaled-MSE style similarity vs expected), `OCR Confidence`, and debug stages.
- Backend sanity checks: `GET /health` and confirm `ready: true` after model weights are available.
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

## OCR Current Status (2026-02-16)

### Current validated state
- App + backend run locally on `127.0.0.1:8000` and `127.0.0.1:8001`.
- Neural ROI detection is integrated and stable.
- Digit-classifier inference is integrated behind `OCR_CONFIG.digitClassifier.enabled` (default `false`), with automatic fallback to Tesseract.
- Backend serves both ROI and digit endpoints and reports readiness in `GET /health`.

### Latest benchmark snapshot (same 11 images)
- Summary artifact: `/tmp/jarvis_testset_summary.json`
- Run mode for this snapshot: classifier path enabled for evaluation.
- Most recent run:
  - Accuracy: `0/11`
  - Mean `Value Match`: `0.000`
  - Mean `OCR Confidence`: `0.940`
- Pattern:
  - Near-constant wrong outputs (`8888`, `8777`, `8898`), indicating class collapse from severe data scarcity/imbalance.

### Dataset coverage snapshot
- Source: `backend/data/digit_dataset/manifests/cells.csv`
- Current cell counts:
  - train `28`, val `12`, test `4`
  - per digit: `0:5, 1:8, 2:13, 3:7, 4:1, 5:0, 6:0, 7:4, 8:3, 9:3`
- Priority deficits with target train count `12` (from `plan_digit_expansion.py`):
  - `4`: deficit `11`
  - `5`: deficit `12`
  - `6`: deficit `12`
  - `9`: deficit `9`

### Active workstream (Option 2)
1. Expand dataset with targeted captures for `4/5/6/9`.
2. Keep QA previews mandatory when adding labels.
3. Run `validate_digit_dataset.py` after each dataset update.
4. Refresh `capture_plan.json`/`capture_plan.md` with `plan_digit_expansion.py`.
5. Retrain only after improved class coverage.

### Tomorrow fallback option (Option 1, if needed)
- If data collection is blocked, tune classifier training recipe before new captures:
  - loss/sampling strategy improvements for imbalance,
  - stronger augmentation and regularization sweep,
  - per-class calibration/error analysis with the same 11-image benchmark.
