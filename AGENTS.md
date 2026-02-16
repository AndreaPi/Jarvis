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
- `backend/`: Optional FastAPI service for neural ROI detection and YOLO fine-tuning scripts.
- `package.json`: Local dev scripts.
- `README.md`: Project overview and setup notes.
- `assets/`: Static assets and example uploads.

## Build, Test, and Development Commands
- `npm run serve`: Start a simple local web server on port 8000.
- `npm run dev`: Alias of `npm run serve`.
- `cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`: Backend setup.
- `cd backend && source .venv/bin/activate && python train_roi.py --data data/roi_dataset.yaml --base-model yolov8n.pt`: Fine-tune pretrained ROI detector.
- `cd backend && source .venv/bin/activate && uvicorn app:app --host 127.0.0.1 --port 8001 --reload`: Run neural ROI API.

Open `http://localhost:8000` after running a serve command. Neural ROI endpoint defaults to `http://127.0.0.1:8001/roi/detect`.

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

## OCR Current Status (2026-02-15)

### Current validated state
- App + backend run locally on `127.0.0.1:8000` and `127.0.0.1:8001`.
- Neural ROI detection quality is much better than before (rot-aug model in use), but final OCR is still Tesseract-based.
- Latest full UI test-set run remains `0/11` correct.

### Latest benchmark snapshot (same 11 images)
- Summary artifact: `/tmp/jarvis_testset_summary.json`
- Most recent run:
  - Accuracy: `0/11`
  - Mean `Value Match`: `0.025`
  - Mean `OCR Confidence`: `0.700`
- Pattern:
  - Most rows return no final read.
  - Occasional high-confidence wrong reads still occur (for example `4441` with confidence `100`).

### What is stale / what to ignore
- Do not spend more cycles on threshold-only or branch-order-only tuning as the primary strategy.
- Do not treat confidence as a reliable correctness proxy in this dataset.
- Do not assume ROI detection improvements will automatically improve final OCR.

### Lessons learned (important)
1. ROI detection and OCR decoding are separate bottlenecks.
2. We have crossed the point of diminishing returns on heuristic tweaks in `src/ocr/*`.
3. High-confidence wrong outputs indicate model mismatch, not just ranking mistakes.
4. The current bottleneck is decoding quality, not candidate routing.
5. Data quality/coverage is now the limiting factor for progress.

### New dataset tooling added (today)
- Added `backend/build_digit_dataset.py`.
- Purpose:
  - export ROI strip crops with 4-digit labels,
  - export per-cell crops grouped by digit class,
  - generate manifests and QA previews.
- Output root: `backend/data/digit_dataset/`
- Quick run:
  - `cd backend && source .venv/bin/activate && python build_digit_dataset.py --clean`
- Current export stats:
  - rows: `11` (train `7`, val `3`, test `1`)
  - cell crops: train `28`, val `12`, test `4`
  - strong class imbalance exists (`5` and `6` have zero samples).

### Next steps (next session)
1. Expand dataset with targeted captures for missing/rare digits (`4/5/6/9` priority).
2. Keep QA previews in the loop when adding labels to avoid bad supervision.
3. Train a dedicated digit OCR model (start with per-cell digit classifier).
4. Integrate model inference in `src/ocr/recognition.js` as a replacement for Tesseract cell decode (behind a flag first).
5. Re-run the same 11-image UI benchmark and compare row-level diffs before further heuristic edits.

### Session update (2026-02-16)
- Added dedicated digit-classifier training + inference stack:
  - `backend/train_digit_classifier.py`
  - `backend/digit_model.py`
  - `backend/digit_classifier.py`
  - API endpoints: `POST /digit/predict`, `POST /digit/predict-cells`
- Integrated classifier-first cell decoding in `src/ocr/recognition.js` with automatic fallback to Tesseract.
- Added frontend feature flag in `src/ocr/config.js` (`digitClassifier.enabled`, default `false`).
- New benchmark run with classifier enabled (`/tmp/jarvis_testset_summary.json`):
  - Accuracy: `0/11`
  - Mean `Value Match`: `0.000`
  - Mean `OCR Confidence`: `0.940`
  - Failure mode: near-constant predictions (`8888` / `8777` / `8898`), indicating class-collapse/overfit from low-data imbalance.
