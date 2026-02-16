# ROI Backend

Python service for neural ROI detection (digit window) using a fine-tuned pretrained YOLO model.

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
python train_roi.py --data data/roi_dataset.yaml --base-model yolov8n.pt
```

For an orientation-robust experiment with strong augmentation (including explicit 90/180/270/360 train rotations):

```bash
python train_roi.py \
  --data data/roi_dataset.yaml \
  --base-model yolov8n.pt \
  --rotation-angles 90,180,270,360 \
  --heavy-augment
```

After training, best weights are copied to `backend/models/roi.pt`.

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

## 4) Endpoints

- `GET /health`: model readiness.
- `POST /roi/detect`: multipart upload (`image`) and returns normalized bbox + confidence.
- `POST /digit/predict`: multipart upload (`image`) and returns the predicted digit + confidence.
- `POST /digit/predict-cells`: multipart upload (`images`, repeated field) for batch cell decoding.

## Environment Variables

- `ROI_MODEL_PATH`: path to `.pt` weights (default: `backend/models/roi.pt`)
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
python train_roi.py --data data/roi_dataset.yaml --base-model yolov8n.pt --device cpu
```
