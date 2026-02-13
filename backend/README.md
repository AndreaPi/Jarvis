# ROI Backend

Python service for neural ROI detection (digit window) using a fine-tuned pretrained YOLO model.

## 1) Install

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Fine-tune on your dataset

Prepare a YOLO dataset and YAML file. A template is available at `data/roi_dataset.example.yaml`.

Expected labels: one class (`digit_window`) with normalized YOLO boxes.

```bash
cd backend
source .venv/bin/activate
python train_roi.py --data data/roi_dataset.yaml --base-model yolov8n.pt
```

After training, best weights are copied to `backend/models/roi.pt`.

## 3) Start the API

```bash
cd backend
source .venv/bin/activate
uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```

## 4) Endpoints

- `GET /health`: model readiness.
- `POST /roi/detect`: multipart upload (`image`) and returns normalized bbox + confidence.

## Environment Variables

- `ROI_MODEL_PATH`: path to `.pt` weights (default: `backend/models/roi.pt`)
- `ROI_DEFAULT_CONFIDENCE`: detection confidence threshold (default: `0.25`)
- `ROI_DEFAULT_IOU`: NMS IoU threshold (default: `0.5`)
- `ROI_DEFAULT_IMGSZ`: inference size (default: `960`)
- `ROI_CLASS_INDEX`: optional class id filter
- `ROI_DEVICE`: optional device (for example `cpu`, `0`)
