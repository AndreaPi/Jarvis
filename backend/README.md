# ROI Backend

Python service for neural ROI detection (digit window) plus optional per-cell digit-classifier inference.

## 1) Install

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

## 2) Fine-tune on your dataset

Prepare a YOLO dataset and YAML file. A template is available at `data/roi_dataset.example.yaml`.

Expected labels: one class (`digit_window`) with normalized YOLO boxes.

```bash
cd backend
source .venv/bin/activate
python train_roi.py \
  --data data/roi_dataset.yaml \
  --base-model yolov8n.pt \
  --rotation-angles 90,180,270,360 \
  --heavy-augment
```

The training script enforces this augmentation policy by default (heavy online augmentation + 90/180/270/360 train rotations). You can bypass it only for explicit ablations with `--allow-no-augment-policy`.

```bash
python train_roi.py --data data/roi_dataset.yaml --base-model yolov8n.pt --allow-no-augment-policy --no-heavy-augment --rotation-angles 90,180
```

After training, best weights are copied to `backend/models/roi.pt`.
The API default is pinned to `backend/models/roi-rotaug-e30-640.pt`; use `ROI_MODEL_PATH` to explicitly test/use `roi.pt` or another checkpoint.

## Build a digit OCR dataset from ROI labels

This creates:
- strip crops + sequence labels (`data/digit_dataset/strips`, `data/digit_dataset/strip_labels`)
- per-cell crops grouped by digit class (`data/digit_dataset/cells`)
- manifests and QA previews (`data/digit_dataset/manifests`, `data/digit_dataset/qa_previews`)

```bash
cd backend
source .venv/bin/activate
python build_digit_dataset.py --clean
```

Validate manifest/file consistency and QA preview coverage:

```bash
python validate_digit_dataset.py
```

Generate a prioritized capture checklist for underrepresented digits:

```bash
python plan_digit_expansion.py --target-train-per-digit 12 --priority-digits 4,5,6,9
```

## Train a dedicated digit classifier

This trains a small per-cell CNN on `data/digit_dataset/cells` and writes:
- weights: `backend/models/digit_classifier.pt`
- training summary: `backend/runs/digit-classifier/digit_classifier_summary.json`

```bash
cd backend
source .venv/bin/activate
python train_digit_classifier.py --device cpu
```

## 3) Start the API

```bash
cd backend
source .venv/bin/activate
uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```

Readiness check:

```bash
curl -s http://127.0.0.1:8001/health
```

## 4) Endpoints

- `GET /health`: model readiness (`ready`, `roi_ready`, `digit_ready`) + effective model/device config.
- `POST /roi/detect`: multipart upload (`image`) and returns normalized bbox + confidence.
- `POST /digit/predict`: multipart upload (`image`) and returns the predicted digit + confidence.
- `POST /digit/predict-cells`: multipart upload (`images`, repeated field) for batch cell decoding.

Frontend integration defaults:
- ROI detection path is `http://127.0.0.1:8001/roi/detect` and is required for OCR.
- Digit classifier path is `http://127.0.0.1:8001/digit/predict-cells` and is only used when `OCR_CONFIG.digitClassifier.enabled=true`.
- Frontend ROI OCR currently uses conservative acceptance guards (`minWordPassHits`, `minRefinedHits`) to avoid high-confidence false positives.

## Environment Variables

- `ROI_MODEL_PATH`: path to `.pt` weights (default: `backend/models/roi-rotaug-e30-640.pt`)
- `ROI_DEFAULT_CONFIDENCE`: detection confidence threshold (default: `0.05`)
- `ROI_DEFAULT_IOU`: NMS IoU threshold (default: `0.5`)
- `ROI_DEFAULT_IMGSZ`: inference size (default: `960`)
- `ROI_CLASS_INDEX`: optional class id filter
- `ROI_DEVICE`: inference device (default: `cpu`).
  - Use `cpu` for CPU-only deploys (recommended on Vercel).
  - Use `auto` to let Ultralytics choose.
  - Use `0` or `cuda:0` to force GPU.
- `DIGIT_MODEL_PATH`: path to digit classifier checkpoint (default: `backend/models/digit_classifier.pt`)
- `DIGIT_DEVICE`: inference device for digit classifier (default follows `ROI_DEVICE`)
- `DIGIT_MIN_CONFIDENCE`: minimum accepted confidence for digit predictions (default: `0.0`)
- `DIGIT_TOP_K`: number of top classes returned by digit endpoints (default: `3`)

## CPU-only vs GPU

- CPU-only install (recommended for Vercel/serverless):
  - `pip install -r requirements-cpu.txt`
- GPU-capable install:
  - install a CUDA-enabled PyTorch build, then `pip install -r requirements.txt`.

Training can also be pinned with `--device`:

```bash
python train_roi.py --data data/roi_dataset.yaml --base-model yolov8n.pt --device cpu --rotation-angles 90,180,270,360 --heavy-augment
```
