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

## OCR Benchmark Update (2026-02-13)

### What was completed
- Verified `backend/models/roi.pt` is present and backend health is `ready: true`.
- Re-ran full UI test set with backend enabled and debug overlay enabled.
- Tuned ROI thresholds:
  - `backend/app.py`: `ROI_DEFAULT_CONFIDENCE` default from `0.25` to `0.05`.
  - `src/ocr/config.js`: `neuralRoi.minConfidence` to `0.05`, `expandX` to `0.18`, `expandY` to `0.24`, kept `includeFullFallbackCandidates: true`.
  - `src/ocr/config.js`: reduced `fallbackCandidates` from `10` to `6` to limit refinement cost.

### Measured results
- Baseline tuned-off snapshot comparison point:
  - `.playwright-cli/page-2026-02-11T22-41-27-697Z.yml`
  - Accuracy: `0/9`.
  - ROI stage-0 coverage in debug sessions: `2/9` images.
  - Mean `Score`: `0.536`.
- After threshold tuning:
  - `.playwright-cli/page-2026-02-13T07-56-25-235Z.yml`
  - Accuracy: `0/9`.
  - ROI stage-0 coverage in debug sessions: `7/9` images.
  - Mean `Score`: `0.389`.

### Key observations
- ROI recall improved materially (2/9 to 7/9), but end-to-end OCR accuracy did not improve.
- Some samples moved closer numerically (for example `meter_10092025_b.JPEG` detected `2200` vs expected `2279`, score `1.00` by scaled metric), but all rows still failed exact-match.
- Runtime remained high for full-suite runs (~17-18 minutes), with repeated fallback refinement phases still dominant.

### Suggested next tuning pass
1. Keep current ROI recall settings (`ROI_DEFAULT_CONFIDENCE=0.05`, `minConfidence=0.05`) but tune strip/fallback selection quality rather than ROI recall.
2. Investigate exact-match objective mismatch vs scaled score (some near-miss values score high but still fail pass/fail).
3. Add per-image candidate logs (top-3 OCR values + scores before final pick) to identify why fallback selects wrong 4-digit sequence.

### Follow-up implementation (2026-02-13, later)
- Implemented strip/fallback selection pass in code:
  - `src/ocr/recognition.js`: value-level ranking now aggregates repeated candidates per OCR pass (`hits`, `bestScore`, `averageScore`) instead of picking only the single max raw candidate.
  - `src/ocr/pipeline.js`: fallback candidate ordering now uses source-aware priority (aligned/strip-first, fallback-deprioritized) before digit-cell refinement.
- Implemented per-image top-candidate logging:
  - `src/ocr/pipeline.js`: final OCR selection now pushes structured logs to `window.__jarvisOcrSelectionLogs` and emits `console.info('[OCR] selection', payload)`.
  - Log payload includes `selected` plus top 3 candidates with score/hits/refined/source-count metadata.
- Validation probe (single image via dynamic module import):
  - `meter_07012020.JPEG` produced one selection log entry with populated `topCandidates` in `window.__jarvisOcrSelectionLogs`.
- Full UI rerun after this pass:
  - `.playwright-cli/page-2026-02-13T08-26-10-467Z.yml`
  - Accuracy remained `0/9`; row outputs and mean `Score` (`0.389`) were unchanged versus `.playwright-cli/page-2026-02-13T07-56-25-235Z.yml`.

### Restart clarification
- Neural network usage is ROI detection only (backend `/roi/detect` + frontend `detectNeuralRoi`); digit OCR is still Tesseract-based.
- To visually verify ROI on a fresh run:
  1. Start frontend (`npm run serve`) and backend (`uvicorn ... --port 8001`).
  2. In UI section "4. Debug the test set", keep "Enable overlay capture" checked.
  3. Click "Run test set".
  4. Inspect per-image stages `0. neural roi detection` and `0b. neural roi crop` in the debug panel.
