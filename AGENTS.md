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
- OCR test-set checks: run "Run test set" and inspect `Score` (scaled-MSE against expected), `OCR` confidence, and debug stages.
- Backend sanity checks: `GET /health` and confirm `ready: true` after model weights are available.

## Commit & Pull Request Guidelines
- No commit message convention is established in this repo.
- Suggested pattern: short, imperative subject (e.g., "Improve OCR preview").
- PRs should include: summary of changes, screenshots for UI changes, and any manual test notes.

## Security & Configuration Tips
- The Gmail draft flow opens a client-side draft; no credentials are stored in code.
- OCR runs in the browser; avoid adding API keys to the client without a secure proxy.
- Backend is intended for local use; keep host/CORS scoped to localhost unless explicitly deploying.

## OCR Handoff Notes (2026-02-10)

### Current status
- App runs on `http://127.0.0.1:8000` (`npm run serve`).
- Frontend now supports optional neural ROI detection via backend API before heuristic ROI search.
- Test-set `Score` now measures scaled-MSE divergence between `Expected` and `Detected`; OCR heuristic confidence is shown separately in `OCR`.
- Backend environment and dependencies are installed locally (`backend/.venv`), but `/health` remains `ready: false` until `backend/models/roi.pt` exists.

### What worked
- Added visual debug pipeline to UI test section:
  - Stage `0`: neural ROI overlay + ROI crop (when backend/model is available)
  - Stage `1`: face detection overlay
  - Stage `2`: aligned frame + ROI boxes
  - Stage `3`: strip score top-k
  - Stage `4`: strip binary/edge decision map
  - Stage `5`: selected strip/fallback crop
  - Stage `6`: OCR input candidate
- Debug capture is available directly from "Run test set" output, so each image can be inspected.
- Frontend pipeline safely falls back to heuristic ROI if backend is unavailable or low-confidence.

### What did not work
- Neural ROI model is not trained yet in this workspace, so backend ROI cannot improve accuracy yet.
- End-to-end OCR accuracy after neural integration has not been re-benchmarked on the full test set.

### Evidence artifacts
- Debug screenshots exported to: `output/playwright/debug-roi/`
- Final test snapshot with table + debug panels: `.playwright-cli/page-2026-02-09T22-52-09-298Z.yml`
- Neural integration code paths: `src/ocr/neural-roi.js`, `src/ocr/pipeline.js`, `backend/app.py`, `backend/train_roi.py`

### Next plan (priority order)
1. Label dataset for `digit_window` and populate `backend/data/roi_dataset.yaml` path.
2. Fine-tune pretrained YOLO (`yolov8n.pt` or `yolov8s.pt`) and produce `backend/models/roi.pt` using:
   `cd backend && source .venv/bin/activate && python train_roi.py --data data/roi_dataset.yaml --base-model yolov8n.pt`
3. Start backend and confirm `GET /health` reports `ready: true` using:
   `cd backend && source .venv/bin/activate && uvicorn app:app --host 127.0.0.1 --port 8001 --reload`
4. Re-run full test set with backend enabled and compare:
   - pass/fail accuracy,
   - scaled-MSE `Score`,
   - OCR confidence (`OCR`),
   - debug overlays (neural ROI vs heuristic fallback).
5. Tune `OCR_CONFIG.neuralRoi` thresholds (`minConfidence`, `expandX`, `expandY`, `includeFullFallbackCandidates`) based on failures.

### Practical notes for tomorrow
- Run frontend and backend in separate terminals (`8000` + `8001`) during OCR benchmarking.
- Health check command once backend is running: `curl http://127.0.0.1:8001/health` (target: `"ready": true`).
- Keep `includeFullFallbackCandidates: true` until neural ROI model quality is verified; disable only after stable ROI recall.
- If `src/ocr/canvas-utils.js` keeps growing, split it into focused modules (for example `image-ops.js` and `region-analysis.js`) before adding new OCR features.
