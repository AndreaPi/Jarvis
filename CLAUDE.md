# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jarvis is a personal assistant web app. The active module lets users upload water meter photos, detect the digit window via a YOLOv8 neural network, OCR the reading with Tesseract.js, and draft a Gmail email with the result. The frontend is vanilla ES modules served by Python's http.server (no bundler); the backend is a Python FastAPI microservice for neural inference.

## Development Commands

### Frontend
```bash
npm run serve                    # Dev server on http://localhost:8000
npm run test:e2e                 # Playwright E2E tests (headless, chromium)
npm run test:e2e:headed          # Playwright E2E tests (visible browser)
npx playwright test tests/e2e/neural-roi.spec.js  # Run a single test file
```

### Backend
```bash
cd backend && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # GPU deps included
pip install -r requirements-cpu.txt      # CPU-only alternative

uvicorn app:app --host 127.0.0.1 --port 8001 --reload   # Start API
curl -s http://127.0.0.1:8001/health                     # Verify readiness
```

### Training & Dataset
```bash
# All commands assume: cd backend && source .venv/bin/activate
python train_roi.py --data data/roi_dataset.yaml --base-model yolov8n.pt
python build_digit_dataset.py --clean
python validate_digit_dataset.py
python plan_digit_expansion.py --target-train-per-digit 12 --priority-digits 4,5,6,9
python train_digit_classifier.py --device cpu
```

## Architecture

**Frontend** (`src/`) — vanilla JS ES modules, no build step:
- `src/ocr/pipeline.js` — main OCR flow: image → neural ROI → crop → strip detection → cell split → Tesseract → scored reading
- `src/ocr/config.js` — all OCR tuning parameters (ROI expansion, strip gates, alignment windows, classifier toggle)
- `src/ocr/recognition.js` — Tesseract.js worker management, candidate scoring
- `src/ocr/neural-roi.js` — client for backend `/roi/detect` with geometry sanity gating
- `src/ocr/digit-classifier.js` — optional client for `/digit/predict-cells` with cooldown/failure tracking
- `src/ocr/alignment.js`, `strip-detection.js`, `canvas-utils.js` — image processing helpers
- `src/email/draft.js` — Gmail draft link generation (Italian locale)
- `src/testset/run-test-set.js` — UI-driven OCR regression runner against `assets/meter_readings.csv`
- `src/main.js` — UI wiring and event handlers
- `app.js` — thin entrypoint importing `src/main.js`

**Backend** (`backend/`) — FastAPI on port 8001:
- `app.py` — endpoints: `GET /health`, `POST /roi/detect`, `POST /digit/predict`, `POST /digit/predict-cells`
- `detector.py` — YOLOv8 wrapper for ROI detection (default model: `models/roi-rotaug-e30-640.pt`, override via `ROI_MODEL_PATH`)
- `digit_classifier.py` / `digit_model.py` — PyTorch CNN digit classifier
- Training scripts: `train_roi.py`, `train_digit_classifier.py`, `build_digit_dataset.py`

**UI** — single-page app in `index.html` + `styles.css`, no framework.

## Key Design Decisions

- **Neural ROI is mandatory** — heuristic ROI fallback has been removed. On neural failure, the UI asks for manual input.
- **OCR runs client-side** in the browser via Tesseract.js; the backend only handles detection/classification inference.
- **Digit classifier is optional**, gated by `OCR_CONFIG.digitClassifier.enabled` in `config.js` (default: `false`).
- **No bundler** — the frontend uses native ES module imports. The dev server is just `python3 -m http.server 8000`.
- **CORS** is scoped to localhost origins only.

## Coding Style

- 2-space indentation in HTML/CSS/JS
- ASCII-only source files
- Descriptive lower-case IDs and class names (`photo-input`, `module-grid`)
- Small, pure functions; avoid deep nesting; prefer early returns
- Python backend uses type hints, FastAPI patterns, env var configuration

## Testing

- **E2E**: Playwright (chromium only), tests in `tests/e2e/`. CI runs on every PR and push to `master` via `.github/workflows/e2e.yml`.
- **OCR regression**: Click "Run test set" in the UI with debug overlay enabled. Reads `assets/meter_readings.csv` and reports `Detected`, `Value Match`, `Failure Reason` per image.
- **Debug tools**: selection logs at `window.__jarvisOcrSelectionLogs` in browser console; debug overlay shows pipeline stages.
- **Backend health**: `GET /health` should return `ready: true`, `roi_ready: true`, and the expected `model_path`.

## OCR Tuning Workflow

1. Adjust parameters in `src/ocr/config.js` (e.g., `expandX`, `expandY`, strip aspect gates, alignment bands).
2. Re-run the UI test set and check both detection coverage and mismatch rate.
3. Use `Failure Reason` column and `window.__jarvisOcrSelectionLogs` to diagnose issues.
4. Iterate: geometry tuning → strip normalization gates → cell decoding strictness.
