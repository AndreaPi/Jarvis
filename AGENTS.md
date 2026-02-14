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

## OCR Current Status (2026-02-14)

### Latest validated state
- App and backend run locally on `127.0.0.1:8000` and `127.0.0.1:8001`.
- Neural ROI is detection-only; final digit OCR is still Tesseract-based.
- Current `src/ocr/config.js` keeps `neuralRoi.includeFullFallbackCandidates: false`.
- Test-set table labels are now `Value Match` and `OCR Confidence`.

### Dataset and model snapshot
- ROI dataset split currently in repo:
  - train: `7`
  - val: `3`
  - test: `1`
- New val samples were committed in `a9e7048`:
  - `backend/data/roi_dataset/images/val/meter_02132026.JPEG`
  - `backend/data/roi_dataset/images/val/meter_02142026.JPEG`
  - `backend/data/roi_dataset/labels/val/meter_02132026.txt`
  - `backend/data/roi_dataset/labels/val/meter_02142026.txt`
- Latest ROI retrain used CPU and produced `backend/models/roi.pt` from `backend/runs/roi-finetune4/weights/best.pt`.

### 2026-02-14 benchmark findings
- Backend ROI sanity:
  - At `ROI_DEFAULT_CONFIDENCE=0.05`: `/roi/detect` returned `0/11` detections on `assets/`.
  - At `ROI_DEFAULT_CONFIDENCE=0.001`: `/roi/detect` returned `11/11`, but with very low confidence (~`0.013` to `0.025`) and one out-of-bounds geometry case.
- Full UI test set with low backend threshold (`0.001`) did not help:
  - Accuracy `0/11`, mean `Value Match` `0.044`, mean `OCR Confidence` `0.491`.
  - Reference snapshot: `.playwright-cli/page-2026-02-14T10-53-40-002Z.yml`.
- A/B check for review comment (`includeFullFallbackCandidates`):
  - `false` vs `true` at backend default confidence `0.05`.
  - Result was identical row-by-row:
    - Accuracy `0/11`
    - Mean `Value Match` `0.483`
    - Mean `OCR Confidence` `0.775`
  - Decision: keep `includeFullFallbackCandidates: false` for now; revisit after ROI quality improves.

### Current priorities
1. Expand ROI dataset further (main blocker).
2. Retrain ROI model on CPU after dataset growth.
3. Re-check offline ROI quality first (`/health`, `/roi/detect` confidence + geometry), then rerun UI test set.
4. Tune OCR selection/fallback behavior only after ROI reliability improves.
