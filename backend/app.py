from __future__ import annotations

import os
from pathlib import Path
import io

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

try:
  from .detector import DetectorUnavailableError, RoiDetector
except ImportError:
  from detector import DetectorUnavailableError, RoiDetector


def _env_float(name: str, default: float) -> float:
  value = os.getenv(name)
  if value is None:
    return default
  try:
    return float(value)
  except ValueError:
    return default


def _env_int(name: str, default: int) -> int:
  value = os.getenv(name)
  if value is None:
    return default
  try:
    return int(value)
  except ValueError:
    return default


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "roi.pt"
MODEL_PATH = Path(os.getenv("ROI_MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser().resolve()
DEFAULT_CONFIDENCE = _env_float("ROI_DEFAULT_CONFIDENCE", 0.05)
DEFAULT_IOU = _env_float("ROI_DEFAULT_IOU", 0.5)
DEFAULT_IMGSZ = _env_int("ROI_DEFAULT_IMGSZ", 960)
CLASS_INDEX = os.getenv("ROI_CLASS_INDEX")
DEVICE_RAW = os.getenv("ROI_DEVICE", "cpu").strip()
if not DEVICE_RAW:
  DEVICE_RAW = "cpu"
DEVICE = None if DEVICE_RAW.lower() == "auto" else DEVICE_RAW

if CLASS_INDEX is None:
  CLASS_INDEX_VALUE = None
else:
  try:
    CLASS_INDEX_VALUE = int(CLASS_INDEX)
  except ValueError:
    CLASS_INDEX_VALUE = None

app = FastAPI(
  title="Jarvis ROI API",
  description="Detects water-meter digit window ROI using a fine-tuned pretrained detector.",
  version="0.1.0"
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=[
    "http://localhost:8000",
    "http://127.0.0.1:8000"
  ],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)

_detector: RoiDetector | None = None
_detector_error: str | None = None


def get_detector() -> RoiDetector:
  global _detector, _detector_error
  if _detector:
    return _detector
  try:
    _detector = RoiDetector(MODEL_PATH, class_index=CLASS_INDEX_VALUE, device=DEVICE)
    _detector_error = None
    return _detector
  except DetectorUnavailableError as error:
    _detector_error = str(error)
    raise


def _load_rgb_image(file_bytes: bytes) -> np.ndarray:
  try:
    with Image.open(io.BytesIO(file_bytes)) as image:
      rgb = image.convert("RGB")
      return np.array(rgb)
  except UnidentifiedImageError as error:
    raise HTTPException(status_code=400, detail="Unsupported image format.") from error

@app.get("/health")
def health() -> dict:
  model_exists = MODEL_PATH.exists()
  ready = False
  error = _detector_error
  try:
    get_detector()
    ready = True
    error = None
  except DetectorUnavailableError as detector_error:
    error = str(detector_error)

  return {
    "ok": True,
    "ready": ready,
    "model_path": str(MODEL_PATH),
    "model_exists": model_exists,
    "device": DEVICE_RAW if ready else (DEVICE or "auto"),
    "default_confidence": DEFAULT_CONFIDENCE,
    "default_iou": DEFAULT_IOU,
    "default_imgsz": DEFAULT_IMGSZ,
    "error": error
  }


@app.post("/roi/detect")
async def detect_roi(image: UploadFile = File(...)) -> dict:
  try:
    detector = get_detector()
  except DetectorUnavailableError as error:
    raise HTTPException(status_code=503, detail=str(error)) from error

  file_bytes = await image.read()
  if not file_bytes:
    raise HTTPException(status_code=400, detail="Empty upload.")

  image_rgb = _load_rgb_image(file_bytes)
  height, width = image_rgb.shape[:2]
  detection = detector.detect(
    image_rgb=image_rgb,
    conf=DEFAULT_CONFIDENCE,
    iou=DEFAULT_IOU,
    imgsz=DEFAULT_IMGSZ
  )

  if not detection:
    return {
      "ok": False,
      "model": detector.model_name,
      "device": detector.device_name,
      "bbox_norm": None,
      "confidence": 0.0,
      "class_id": None,
      "class_name": None,
      "image_size": {"width": width, "height": height}
    }

  return {
    "ok": True,
    "model": detector.model_name,
    "device": detector.device_name,
    "bbox_norm": detection.to_normalized_bbox(width=width, height=height),
    "confidence": detection.confidence,
    "class_id": detection.class_id,
    "class_name": detection.class_name,
    "image_size": {"width": width, "height": height}
  }
