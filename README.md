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

2. Train/fine-tune a model (copies best checkpoint to `backend/models/roi.pt`):

```bash
python train_roi.py --data data/roi_dataset.yaml --base-model yolov8n.pt
```

3. Start the API:

```bash
uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```

By default, the frontend calls `http://127.0.0.1:8001/roi/detect` before heuristic ROI search.

## File Overview
- `index.html`: UI layout.
- `styles.css`: Styling.
- `app.js`: OCR + email draft logic.
- `backend/`: Optional FastAPI service for neural ROI detection + YOLO fine-tuning script.
- `AGENTS.md`: Contributor guide.
- `assets/`: Static assets and example uploads.

## Notes
- OCR runs fully in the browser using Tesseract.js.
- If the optional backend is running, neural ROI detection is used first, then JS OCR reads digits from the cropped region.
- The Gmail flow opens a draft; you always review and send manually.

## Asset Naming (Meter Images)
- Use the EXIF `DateTimeOriginal` value as the source of truth for the acquisition date.
- Rename files to `meter_mmddyyyy` (zero-padded) and keep the original extension.
- If multiple images share the same date, keep one as-is and add suffixes to the rest (e.g., `_b`, `_c`).
- If EXIF is missing, prefer a known date from the filename or capture notes and document it.
