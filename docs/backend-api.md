# Backend API

Jarvis includes an optional local FastAPI backend for ROI detection and digit classification.

## Run

From repo root:

```bash
cd backend
source .venv/bin/activate
uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```

In the Codex/DevTools environment, a backend started inside the sandbox may not be reachable from the page even if shell `curl` works. If browser fetches to `127.0.0.1:8001` fail, restart the service outside the sandbox and verify connectivity from the browser context.

Default local base URL: `http://127.0.0.1:8001`

## Endpoints

### `GET /health`

Reports backend readiness and resolved model/runtime configuration.

Example:

```bash
curl -s http://127.0.0.1:8001/health
```

Key fields:

- `ready` / `roi_ready`: ROI detector is loadable.
- `digit_ready`: digit classifier is loadable.
- `model_path`, `model_source`, `device`: ROI model/runtime selection.
- `digit_model_path`, `digit_device`: digit model/runtime selection.
- `default_confidence`, `default_iou`, `default_imgsz`: ROI inference defaults.
- `digit_min_confidence`, `digit_top_k`: classifier acceptance defaults.

### `POST /roi/detect`

Detects the ROI box for the 4-digit black register.

Request:

- Form-data field: `image` (required)

Example:

```bash
curl -s -X POST http://127.0.0.1:8001/roi/detect \
  -F "image=@assets/meter_02272026.JPEG"
```

Success response (shape):

```json
{
  "ok": true,
  "model": "roi-rotaug-e30-640.pt",
  "device": "cpu",
  "bbox_norm": { "x": 0.41, "y": 0.40, "width": 0.08, "height": 0.15 },
  "confidence": 0.49,
  "class_id": 0,
  "class_name": "digit_window",
  "image_size": { "width": 1536, "height": 2048 }
}
```

No-detection response:

```json
{
  "ok": false,
  "bbox_norm": null,
  "confidence": 0.0,
  "class_id": null,
  "class_name": null
}
```

### `POST /digit/predict`

Classifies one digit image.

Request:

- Form-data field: `image` (required)

Response includes:

- `predicted_digit`, `confidence`
- `accepted` / `ok` based on `DIGIT_MIN_CONFIDENCE`
- `digit` is null when prediction is not accepted

### `POST /digit/predict-cells`

Batch classifier endpoint for up to 16 images.

Request:

- Form-data field name: `images` (repeat for each file)

Example:

```bash
curl -s -X POST http://127.0.0.1:8001/digit/predict-cells \
  -F "images=@/tmp/cell0.jpg" \
  -F "images=@/tmp/cell1.jpg"
```

Response includes:

- `accepted_count`, `total`
- `predictions[]` with `predicted_digit`, `confidence`, `accepted`
- Per-item `error: "empty-upload"` when a file item is empty

## Error Semantics

- `400`: bad input (empty upload, unsupported image, too many images, etc.)
- `503`: model/dependency unavailable
- `200` with `ok: false` (ROI endpoint): inference ran but no ROI detected

## Environment Variables

ROI detector:

- `ROI_MODEL_PATH`
- `ROI_DEFAULT_CONFIDENCE` (default `0.05`)
- `ROI_DEFAULT_IOU` (default `0.5`)
- `ROI_DEFAULT_IMGSZ` (default `960`)
- `ROI_CLASS_INDEX` (optional class filter)
- `ROI_DEVICE` (`cpu`, `cuda`, `auto`)

Digit classifier:

- `DIGIT_MODEL_PATH`
- `DIGIT_MIN_CONFIDENCE` (default `0.0`)
- `DIGIT_TOP_K` (default `3`)
- `DIGIT_DEVICE` (`cpu`, `cuda`, `auto`)

## Source

- API entrypoint: `backend/app.py`
- ROI detector wrapper: `backend/detector.py`
- Digit classifier wrapper: `backend/digit_classifier.py`
