# Jarvis

Jarvis is a lightweight personal assistant web app. The first module helps you read a water meter photo, review the detected value, and draft an email in Gmail.

## Features
- Upload a meter photo and preview it.
- OCR the reading (manual override supported).
- Auto-fill an email draft with the current date in Italian format.
- Open a Gmail draft or use a mailto fallback.

## Local Development
1. Install dependencies (none required beyond Python).
2. Run the dev server:

```bash
npm run serve
```

Then open `http://localhost:8000`.

### Optional Neural ROI Backend (recommended)
You can run a Python backend that detects the meter digit window using a fine-tuned pretrained model.

1. Open a second terminal and set up backend dependencies:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CPU-only environments (for example Vercel), install:

```bash
pip install -r requirements-cpu.txt
```


2. Train/fine-tune a model (copies best checkpoint to `backend/models/roi.pt`):

```bash
python train_roi.py --data data/roi_dataset.yaml --base-model yolov8n.pt
```

The API default ROI checkpoint is pinned to `backend/models/roi-rotaug-e30-640.pt`.
To run with a newly trained checkpoint, set `ROI_MODEL_PATH` explicitly before starting the backend.

Optional: train the per-cell digit classifier checkpoint:

```bash
python train_digit_classifier.py --device cpu
```

For dataset expansion/QA before retraining:

```bash
python plan_digit_expansion.py --target-train-per-digit 12 --priority-digits 4,5,6,9
python validate_digit_dataset.py
```

3. Start the API:

```bash
uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```

By default, the frontend calls `http://127.0.0.1:8001/roi/detect` and requires neural ROI detection before OCR.
The frontend can also call `http://127.0.0.1:8001/digit/predict-cells` when `OCR_CONFIG.digitClassifier.enabled` is set to `true`.

### E2E Tests

Run Playwright checks for neural-ROI failure handling:

```bash
npm run test:e2e
```

CI runs these tests on every pull request and on pushes to `main`.

## File Overview
- `index.html`: UI layout.
- `styles.css`: Styling.
- `app.js`: OCR + email draft logic.
- `backend/`: Optional FastAPI service for neural ROI detection + YOLO fine-tuning script.
- `AGENTS.md`: Contributor guide.
- `assets/`: Static assets and example uploads.

## Notes
- OCR runs fully in the browser using Tesseract.js.
- OCR now relies on neural ROI detection; if the backend is unavailable or ROI fails, the app asks for manual reading input.
- Digit decoding can optionally use a backend classifier (`src/ocr/config.js` -> `digitClassifier.enabled`) with automatic fallback to Tesseract.
- The Gmail flow opens a draft; you always review and send manually.

## Asset Naming (Meter Images)
- Use the EXIF `DateTimeOriginal` value as the source of truth for the acquisition date.
- Rename files to `meter_mmddyyyy` (zero-padded) and keep the original extension.
- If multiple images share the same date, keep one as-is and add suffixes to the rest (e.g., `_b`, `_c`).
- If EXIF is missing, prefer a known date from the filename or capture notes and document it.
