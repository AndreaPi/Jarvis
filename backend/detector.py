from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class DetectorUnavailableError(RuntimeError):
  """Raised when detector dependencies or weights are not available."""


@dataclass
class RoiDetection:
  x1: float
  y1: float
  x2: float
  y2: float
  confidence: float
  class_id: int
  class_name: str

  def to_normalized_bbox(self, width: int, height: int) -> dict[str, float]:
    safe_width = max(1, int(width))
    safe_height = max(1, int(height))
    x = max(0.0, min(self.x1, float(safe_width - 1)))
    y = max(0.0, min(self.y1, float(safe_height - 1)))
    w = max(1.0, min(self.x2, float(safe_width)) - x)
    h = max(1.0, min(self.y2, float(safe_height)) - y)
    return {
      "x": x / safe_width,
      "y": y / safe_height,
      "width": w / safe_width,
      "height": h / safe_height
    }


class RoiDetector:
  def __init__(self, weights_path: Path, class_index: int | None = None, device: str | None = None) -> None:
    self.weights_path = Path(weights_path)
    self.class_index = class_index
    self.device = device or None
    if not self.weights_path.exists():
      raise DetectorUnavailableError(
        f"Model weights not found at {self.weights_path}. Train first or set ROI_MODEL_PATH."
      )

    try:
      from ultralytics import YOLO
    except ImportError as error:
      raise DetectorUnavailableError(
        "ultralytics is not installed. Install backend/requirements.txt first."
      ) from error

    self._model = YOLO(str(self.weights_path))
    self._names = self._model.names or {}

  @property
  def model_name(self) -> str:
    return self.weights_path.name

  def _resolve_name(self, class_id: int) -> str:
    if isinstance(self._names, dict):
      return str(self._names.get(class_id, class_id))
    if isinstance(self._names, list) and 0 <= class_id < len(self._names):
      return str(self._names[class_id])
    return str(class_id)

  def detect(self, image_rgb: np.ndarray, conf: float = 0.25, iou: float = 0.5, imgsz: int = 960) -> RoiDetection | None:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
      raise ValueError("Expected RGB image with shape HxWx3.")

    results = self._model.predict(
      source=image_rgb,
      conf=conf,
      iou=iou,
      imgsz=imgsz,
      device=self.device,
      verbose=False
    )
    if not results:
      return None

    result = results[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
      return None

    best: RoiDetection | None = None
    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)
    for coords, score, class_id in zip(xyxy, confidences, class_ids):
      if self.class_index is not None and class_id != self.class_index:
        continue
      x1, y1, x2, y2 = (float(value) for value in coords.tolist())
      detection = RoiDetection(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        confidence=float(score),
        class_id=int(class_id),
        class_name=self._resolve_name(int(class_id))
      )
      if not best or detection.confidence > best.confidence:
        best = detection

    return best
