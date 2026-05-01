from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

try:
  from .detector import DetectorUnavailableError, RoiDetector
except ImportError:
  from detector import DetectorUnavailableError, RoiDetector

try:
  from .digit_classifier import DigitClassifier, DigitClassifierUnavailableError
except ImportError:
  from digit_classifier import DigitClassifier, DigitClassifierUnavailableError

try:
  from .strip_digit_reader import StripDigitReader, StripDigitReaderUnavailableError
except ImportError:
  from strip_digit_reader import StripDigitReader, StripDigitReaderUnavailableError


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
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "roi-rotaug-e30-640.pt"
MODEL_PATH_ENV = os.getenv("ROI_MODEL_PATH")
MODEL_PATH = Path(MODEL_PATH_ENV or str(DEFAULT_MODEL_PATH)).expanduser().resolve()
MODEL_SOURCE = "env" if MODEL_PATH_ENV else "default"
DEFAULT_CONFIDENCE = _env_float("ROI_DEFAULT_CONFIDENCE", 0.05)
DEFAULT_IOU = _env_float("ROI_DEFAULT_IOU", 0.5)
DEFAULT_IMGSZ = _env_int("ROI_DEFAULT_IMGSZ", 960)
DEFAULT_DIGIT_MODEL_PATH = BASE_DIR / "models" / "digit_classifier.pt"
DIGIT_MODEL_PATH = Path(os.getenv("DIGIT_MODEL_PATH", str(DEFAULT_DIGIT_MODEL_PATH))).expanduser().resolve()
DIGIT_MIN_CONFIDENCE = _env_float("DIGIT_MIN_CONFIDENCE", 0.0)
DIGIT_TOP_K = _env_int("DIGIT_TOP_K", 3)
DEFAULT_STRIP_DIGIT_MODEL_PATH = BASE_DIR / "models" / "digit_strip_reader.pt"
STRIP_DIGIT_MODEL_PATH = Path(
  os.getenv("STRIP_DIGIT_MODEL_PATH", str(DEFAULT_STRIP_DIGIT_MODEL_PATH))
).expanduser().resolve()
STRIP_DIGIT_MIN_CONFIDENCE = _env_float("STRIP_DIGIT_MIN_CONFIDENCE", 0.0)
STRIP_DIGIT_TOP_K = _env_int("STRIP_DIGIT_TOP_K", 3)
CLASS_INDEX = os.getenv("ROI_CLASS_INDEX")
DEVICE_RAW = os.getenv("ROI_DEVICE", "cpu").strip()
if not DEVICE_RAW:
  DEVICE_RAW = "cpu"
DEVICE = None if DEVICE_RAW.lower() == "auto" else DEVICE_RAW
DIGIT_DEVICE_RAW = os.getenv("DIGIT_DEVICE", DEVICE_RAW).strip()
if not DIGIT_DEVICE_RAW:
  DIGIT_DEVICE_RAW = DEVICE_RAW
DIGIT_DEVICE = None if DIGIT_DEVICE_RAW.lower() == "auto" else DIGIT_DEVICE_RAW
STRIP_DIGIT_DEVICE_RAW = os.getenv("STRIP_DIGIT_DEVICE", DIGIT_DEVICE_RAW).strip()
if not STRIP_DIGIT_DEVICE_RAW:
  STRIP_DIGIT_DEVICE_RAW = DIGIT_DEVICE_RAW
STRIP_DIGIT_DEVICE = None if STRIP_DIGIT_DEVICE_RAW.lower() == "auto" else STRIP_DIGIT_DEVICE_RAW

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
_digit_classifier: DigitClassifier | None = None
_digit_classifier_error: str | None = None
_strip_digit_reader: StripDigitReader | None = None
_strip_digit_reader_error: str | None = None


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


def get_digit_classifier() -> DigitClassifier:
  global _digit_classifier, _digit_classifier_error
  if _digit_classifier:
    return _digit_classifier
  try:
    _digit_classifier = DigitClassifier(DIGIT_MODEL_PATH, device=DIGIT_DEVICE)
    _digit_classifier_error = None
    return _digit_classifier
  except DigitClassifierUnavailableError as error:
    _digit_classifier_error = str(error)
    raise


def get_strip_digit_reader() -> StripDigitReader:
  global _strip_digit_reader, _strip_digit_reader_error
  if _strip_digit_reader:
    return _strip_digit_reader
  try:
    _strip_digit_reader = StripDigitReader(STRIP_DIGIT_MODEL_PATH, device=STRIP_DIGIT_DEVICE)
    _strip_digit_reader_error = None
    return _strip_digit_reader
  except StripDigitReaderUnavailableError as error:
    _strip_digit_reader_error = str(error)
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
  roi_model_exists = MODEL_PATH.exists()
  digit_model_exists = DIGIT_MODEL_PATH.exists()
  strip_digit_model_exists = STRIP_DIGIT_MODEL_PATH.exists()
  roi_ready = False
  digit_ready = False
  strip_digit_ready = False
  roi_error = _detector_error
  digit_error = _digit_classifier_error
  strip_digit_error = _strip_digit_reader_error
  try:
    get_detector()
    roi_ready = True
    roi_error = None
  except DetectorUnavailableError as detector_error:
    roi_error = str(detector_error)
  try:
    get_digit_classifier()
    digit_ready = True
    digit_error = None
  except DigitClassifierUnavailableError as classifier_error:
    digit_error = str(classifier_error)
  try:
    get_strip_digit_reader()
    strip_digit_ready = True
    strip_digit_error = None
  except StripDigitReaderUnavailableError as reader_error:
    strip_digit_error = str(reader_error)

  return {
    "ok": True,
    "ready": roi_ready,
    "roi_ready": roi_ready,
    "digit_ready": digit_ready,
    "strip_digit_ready": strip_digit_ready,
    "model_path": str(MODEL_PATH),
    "model_source": MODEL_SOURCE,
    "default_model_path": str(DEFAULT_MODEL_PATH),
    "model_exists": roi_model_exists,
    "digit_model_path": str(DIGIT_MODEL_PATH),
    "digit_model_exists": digit_model_exists,
    "strip_digit_model_path": str(STRIP_DIGIT_MODEL_PATH),
    "strip_digit_model_exists": strip_digit_model_exists,
    "device": DEVICE_RAW if roi_ready else (DEVICE or "auto"),
    "digit_device": DIGIT_DEVICE_RAW if digit_ready else (DIGIT_DEVICE or "auto"),
    "strip_digit_device": STRIP_DIGIT_DEVICE_RAW if strip_digit_ready else (STRIP_DIGIT_DEVICE or "auto"),
    "default_confidence": DEFAULT_CONFIDENCE,
    "default_iou": DEFAULT_IOU,
    "default_imgsz": DEFAULT_IMGSZ,
    "digit_min_confidence": DIGIT_MIN_CONFIDENCE,
    "digit_top_k": DIGIT_TOP_K,
    "strip_digit_min_confidence": STRIP_DIGIT_MIN_CONFIDENCE,
    "strip_digit_top_k": STRIP_DIGIT_TOP_K,
    "error": roi_error,
    "digit_error": digit_error,
    "strip_digit_error": strip_digit_error
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


@app.post("/digit/predict")
async def predict_digit(image: UploadFile = File(...)) -> dict:
  try:
    classifier = get_digit_classifier()
  except DigitClassifierUnavailableError as error:
    raise HTTPException(status_code=503, detail=str(error)) from error

  file_bytes = await image.read()
  if not file_bytes:
    raise HTTPException(status_code=400, detail="Empty upload.")

  image_rgb = _load_rgb_image(file_bytes)
  prediction = classifier.predict(image_rgb=image_rgb, top_k=DIGIT_TOP_K)
  accepted = prediction.confidence >= DIGIT_MIN_CONFIDENCE

  return {
    "ok": accepted,
    "accepted": accepted,
    "model": classifier.model_name,
    "device": classifier.device_name,
    "digit": prediction.digit if accepted else None,
    "predicted_digit": prediction.digit,
    "confidence": prediction.confidence,
    "min_confidence": DIGIT_MIN_CONFIDENCE,
    "top_k": prediction.top_k
  }


@app.post("/digit/predict-cells")
async def predict_digit_cells(images: list[UploadFile] = File(...)) -> dict:
  try:
    classifier = get_digit_classifier()
  except DigitClassifierUnavailableError as error:
    raise HTTPException(status_code=503, detail=str(error)) from error

  if not images:
    raise HTTPException(status_code=400, detail="At least one image is required.")
  if len(images) > 16:
    raise HTTPException(status_code=400, detail="Too many images; limit is 16.")

  predictions = []
  accepted_count = 0
  for upload in images:
    file_bytes = await upload.read()
    if not file_bytes:
      predictions.append({
        "ok": False,
        "accepted": False,
        "digit": None,
        "predicted_digit": None,
        "confidence": 0.0,
        "error": "empty-upload"
      })
      continue

    image_rgb = _load_rgb_image(file_bytes)
    prediction = classifier.predict(image_rgb=image_rgb, top_k=DIGIT_TOP_K)
    accepted = prediction.confidence >= DIGIT_MIN_CONFIDENCE
    if accepted:
      accepted_count += 1
    predictions.append({
      "ok": accepted,
      "accepted": accepted,
      "digit": prediction.digit if accepted else None,
      "predicted_digit": prediction.digit,
      "confidence": prediction.confidence,
      "top_k": prediction.top_k
    })

  return {
    "ok": True,
    "model": classifier.model_name,
    "device": classifier.device_name,
    "min_confidence": DIGIT_MIN_CONFIDENCE,
    "accepted_count": accepted_count,
    "total": len(images),
    "predictions": predictions
  }


@app.post("/digit/predict-strip")
async def predict_digit_strip(image: UploadFile = File(...)) -> dict:
  try:
    reader = get_strip_digit_reader()
  except StripDigitReaderUnavailableError as error:
    raise HTTPException(status_code=503, detail=str(error)) from error

  file_bytes = await image.read()
  if not file_bytes:
    raise HTTPException(status_code=400, detail="Empty upload.")

  image_rgb = _load_rgb_image(file_bytes)
  prediction = reader.predict(image_rgb=image_rgb, top_k=STRIP_DIGIT_TOP_K)
  accepted = prediction.confidence >= STRIP_DIGIT_MIN_CONFIDENCE

  return {
    "ok": accepted,
    "accepted": accepted,
    "model": reader.model_name,
    "device": reader.device_name,
    "value": prediction.value if accepted else None,
    "predicted_value": prediction.value,
    "confidence": prediction.confidence,
    "min_confidence": STRIP_DIGIT_MIN_CONFIDENCE,
    "digits": prediction.digits if accepted else [],
    "predicted_digits": prediction.digits,
    "digit_confidences": prediction.digit_confidences,
    "top_k_by_position": prediction.top_k_by_position
  }
