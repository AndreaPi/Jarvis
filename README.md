# Jarvis

Jarvis is a lightweight personal assistant web app. The first module helps you read a water meter photo, review the detected value, and draft an email in Gmail.

## Documentation

- Docs index: [`docs/README.md`](./docs/README.md)
- OCR app logic flow: [`docs/app-logic.md`](./docs/app-logic.md)
- Backend API guide: [`docs/backend-api.md`](./docs/backend-api.md)
- OCR tuning playbook: [`docs/ocr-tuning-playbook.md`](./docs/ocr-tuning-playbook.md)

## Features
- Upload a meter photo and preview it.
- OCR from a neural-ROI crop with conservative acceptance (unsupported OCR guesses are rejected to manual input).
- Auto-fill an email draft with the current date in Italian format.
- Open a Gmail draft or use a mailto fallback.
- Run a built-in OCR test set table with `Detected`, `Absolute Error`, and `Failure Reason` columns plus MAE/exact-match/no-read summary stats.

## Local Development
1. Ensure Python 3 and Node.js are installed.
2. Run the dev server:

```bash
npm run serve
```

Then open `http://localhost:8000`.

If you also want to run Playwright checks, install JS dependencies once:

```bash
npm install
```

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
python train_roi.py \
  --data data/roi_dataset.yaml \
  --base-model yolov8n.pt \
  --rotation-angles 90,180,270,360 \
  --heavy-augment
```

The API default ROI checkpoint is pinned to `backend/models/roi-rotaug-e30-640.pt`.
To run with a newly trained checkpoint, set `ROI_MODEL_PATH` explicitly before starting the backend.
`train_roi.py` now enforces heavy augmentation + rotation expansion by default; weaker runs require explicit `--allow-no-augment-policy`.

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
Digit decoding is neural-classifier-only and calls `http://127.0.0.1:8001/digit/predict-cells`.
Check backend readiness with:

```bash
curl -s http://127.0.0.1:8001/health
```

### E2E Tests

Run Playwright checks for neural-ROI failure handling and OCR selection guard regressions:

```bash
npm run test:e2e
```

Generate a per-image ROI checkpoint comparison report (`roi-rotaug-e30-640.pt` vs `roi.pt`) with stage `5/6` debug snapshots:

```bash
npm run benchmark:roi-diff
```

Report artifacts are written under `output/roi-checkpoint-diff/<timestamp>/`.
Per-image diff tables include selected OCR metadata (`sourceLabel`, `method`, `preprocessMode`) and stage `6` exports use the last `6. OCR input candidate` frame from each debug session (the winning decode strip variant).

CI runs these tests on every pull request and on pushes to `master`.

## File Overview
- `index.html`: UI layout.
- `styles.css`: Styling.
- `app.js`: Thin entrypoint that imports `src/main.js`.
- `src/main.js`: UI orchestration and event wiring.
- `src/ocr/`: OCR pipeline and neural ROI integration.
- `src/testset/`: Manual OCR test-set runner.
- `backend/`: Optional FastAPI service for neural ROI and digit-classifier inference/training.
- `AGENTS.md`: Contributor guide.
- `assets/`: Static assets and example uploads.

## Notes
- OCR now relies on neural ROI detection; if the backend is unavailable or ROI fails, the app asks for manual reading input.
- Digit decoding uses the backend neural classifier endpoint (`/digit/predict-cells`) and is enabled by default.
- Edge-derived ROI strip candidates are enabled by default and can be toggled with `OCR_CONFIG.roiDeterministic.useEdgeCandidates`.
- The selection layer is fail-safe: isolated edge-only single hits are rejected unless independently corroborated.
- Use the UI `Run test set` action plus `npm run test:e2e` for OCR regressions before and after tuning.
- The Gmail flow opens a draft; you always review and send manually.

## Asset Naming (Meter Images)
- Use the EXIF `DateTimeOriginal` value as the source of truth for the acquisition date.
- Rename files to `meter_yyyymmdd` (zero-padded) and keep the original extension.
- If multiple images share the same date, keep one as-is and add numeric suffixes to the rest (e.g., `_1`, `_2`).
- If EXIF is missing, prefer a known date from the filename or capture notes and document it.
