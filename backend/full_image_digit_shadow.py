"""Shadow-only ROI-to-digit-detector inference for full meter images."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
  from .detector import RoiDetector
except ImportError:
  from detector import RoiDetector


class FullImageDigitShadowUnavailableError(RuntimeError):
  """Raised when the shadow detector cannot be loaded."""


RUNTIME_ROI_SANITY = {
  "min_center_x": 0.28,
  "max_center_x": 0.62,
  "min_center_y": 0.28,
  "max_center_y": 0.68,
  "min_area": 0.003,
  "max_area": 0.03,
  "min_aspect": 0.35,
  "max_aspect": 3.2,
}


def evaluate_roi_sanity(
  bbox: dict[str, float],
) -> tuple[bool, str, dict[str, float]]:
  center_x = bbox["x"] + bbox["width"] * 0.5
  center_y = bbox["y"] + bbox["height"] * 0.5
  area = bbox["width"] * bbox["height"]
  aspect = bbox["width"] / max(bbox["height"], 1e-6)
  geometry = {
    "center_x": center_x,
    "center_y": center_y,
    "area": area,
    "aspect": aspect,
  }
  checks = [
    ("center_x", "min_center_x", "max_center_x"),
    ("center_y", "min_center_y", "max_center_y"),
    ("area", "min_area", "max_area"),
    ("aspect", "min_aspect", "max_aspect"),
  ]
  for metric, minimum_key, maximum_key in checks:
    if not RUNTIME_ROI_SANITY[minimum_key] <= geometry[metric] <= RUNTIME_ROI_SANITY[maximum_key]:
      return False, f"invalid-{metric.replace('_', '-')}", geometry
  return True, "accepted", geometry


def expand_normalized_bbox(
  bbox: dict[str, float],
  expand_x: float,
  expand_y: float,
) -> dict[str, float]:
  left = max(0.0, bbox["x"] - bbox["width"] * expand_x)
  top = max(0.0, bbox["y"] - bbox["height"] * expand_y)
  right = min(1.0, bbox["x"] + bbox["width"] * (1 + expand_x))
  bottom = min(1.0, bbox["y"] + bbox["height"] * (1 + expand_y))
  return {
    "x": left,
    "y": top,
    "width": right - left,
    "height": bottom - top,
  }


def crop_image(
  image_rgb: np.ndarray,
  bbox: dict[str, float],
) -> np.ndarray:
  height, width = image_rgb.shape[:2]
  left = max(0, int(np.floor(bbox["x"] * width)))
  top = max(0, int(np.floor(bbox["y"] * height)))
  right = min(width, int(np.ceil((bbox["x"] + bbox["width"]) * width)))
  bottom = min(height, int(np.ceil((bbox["y"] + bbox["height"]) * height)))
  if right <= left or bottom <= top:
    raise ValueError("ROI expansion produced an empty crop.")
  return image_rgb[top:bottom, left:right].copy()


def sort_detections(
  detections: list[dict[str, float | int]],
  rotation: int,
) -> list[dict[str, float | int]]:
  if rotation == 0:
    return sorted(detections, key=lambda item: float(item["x_center"]))
  if rotation == 180:
    return sorted(detections, key=lambda item: float(item["x_center"]), reverse=True)
  if rotation == 90:
    return sorted(detections, key=lambda item: float(item["y_center"]), reverse=True)
  if rotation == 270:
    return sorted(detections, key=lambda item: float(item["y_center"]))
  raise ValueError(f"Unsupported reading-direction rotation: {rotation}")


def build_rotation_candidates(
  detections: list[dict[str, float | int]],
) -> list[dict[str, object]]:
  candidates = []
  for rotation in (0, 90, 180, 270):
    ordered = sort_detections(detections, rotation)
    candidates.append({
      "rotation": rotation,
      "value": (
        "".join(str(int(item["class_id"])) for item in ordered)
        if len(ordered) == 4
        else None
      ),
    })
  return candidates


def detections_from_result(result: object) -> list[dict[str, float | int]]:
  boxes = result.boxes
  if boxes is None or len(boxes) == 0:
    return []
  normalized = boxes.xywhn.cpu().numpy()
  confidences = boxes.conf.cpu().numpy()
  class_ids = boxes.cls.cpu().numpy().astype(int)
  detections = []
  for values, confidence, class_id in zip(normalized, confidences, class_ids):
    x_center, y_center, width, height = (float(value) for value in values.tolist())
    detections.append({
      "class_id": int(class_id),
      "confidence": float(confidence),
      "x_center": x_center,
      "y_center": y_center,
      "width": width,
      "height": height,
    })
  return detections


class FullImageDigitShadow:
  def __init__(self, weights_path: Path, device: str | None = None) -> None:
    self.weights_path = Path(weights_path)
    self.device = None if device is None or str(device).lower() == "auto" else str(device)
    if not self.weights_path.exists():
      raise FullImageDigitShadowUnavailableError(
        f"Shadow digit-detector weights not found at {self.weights_path}."
      )
    try:
      from ultralytics import YOLO
    except ImportError as error:
      raise FullImageDigitShadowUnavailableError(
        "ultralytics is required for the full-image digit shadow."
      ) from error
    self._model = YOLO(str(self.weights_path))

  @property
  def model_name(self) -> str:
    return self.weights_path.name

  @property
  def device_name(self) -> str:
    return self.device or "auto"

  def predict(
    self,
    image_rgb: np.ndarray,
    roi_detector: RoiDetector,
    *,
    roi_confidence: float = 0.05,
    roi_iou: float = 0.5,
    roi_imgsz: int = 960,
    roi_expand_x: float = 0.26,
    roi_expand_y: float = 0.16,
    confidence: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 1280,
    max_detections: int = 300,
  ) -> dict[str, object]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
      raise ValueError("Expected RGB image with shape HxWx3.")
    height, width = image_rgb.shape[:2]
    roi_detection = roi_detector.detect(
      image_rgb,
      conf=roi_confidence,
      iou=roi_iou,
      imgsz=roi_imgsz,
    )
    if roi_detection is None:
      return {
        "ok": False,
        "reason": "roi-no-detection",
        "model": self.model_name,
        "device": self.device_name,
        "roi_model": roi_detector.model_name,
        "detection_count": 0,
        "detections": [],
        "candidates": [],
      }
    roi_bbox = roi_detection.to_normalized_bbox(width, height)
    sane, roi_status, geometry = evaluate_roi_sanity(roi_bbox)
    roi_payload = {
      "status": roi_status,
      "confidence": roi_detection.confidence,
      "bbox_norm": roi_bbox,
      "geometry": geometry,
    }
    if not sane:
      return {
        "ok": False,
        "reason": f"roi-{roi_status}",
        "model": self.model_name,
        "device": self.device_name,
        "roi_model": roi_detector.model_name,
        "roi": roi_payload,
        "detection_count": 0,
        "detections": [],
        "candidates": [],
      }
    expanded_bbox = expand_normalized_bbox(roi_bbox, roi_expand_x, roi_expand_y)
    roi_payload["expanded_bbox_norm"] = expanded_bbox
    register_crop = crop_image(image_rgb, expanded_bbox)
    results = self._model.predict(
      source=register_crop,
      conf=confidence,
      iou=iou,
      imgsz=imgsz,
      max_det=max_detections,
      agnostic_nms=True,
      device=self.device,
      save=False,
      verbose=False,
    )
    detections = detections_from_result(results[0]) if results else []
    confidences = [float(item["confidence"]) for item in detections]
    return {
      "ok": len(detections) == 4,
      "reason": None if len(detections) == 4 else "digit-detection-count",
      "model": self.model_name,
      "device": self.device_name,
      "roi_model": roi_detector.model_name,
      "roi": roi_payload,
      "detection_count": len(detections),
      "confidence": min(confidences) if len(detections) == 4 else 0.0,
      "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
      "detections": detections,
      "candidates": build_rotation_candidates(detections),
    }
