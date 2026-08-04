"""Export a visual out-of-fold error audit for full-image digit detectors."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
  from backend.evaluate_full_image_digit_detector import (
    build_sequence_record,
    detections_from_result,
    summarize_sequence_records,
  )
  from backend.train_full_image_digit_detector import (
    annotation_box_pixels,
    digit_centered_crop_bounds,
    file_sha256,
    read_csv_rows,
    resolve_device,
    resolve_path,
  )
except ModuleNotFoundError:
  from evaluate_full_image_digit_detector import (
    build_sequence_record,
    detections_from_result,
    summarize_sequence_records,
  )
  from train_full_image_digit_detector import (
    annotation_box_pixels,
    digit_centered_crop_bounds,
    file_sha256,
    read_csv_rows,
    resolve_device,
    resolve_path,
  )


MATCH_IOU_THRESHOLD = 0.10
REGISTER_CONTEXT_RATIO = 0.75
RUNTIME_ROI_CONFIDENCE = 0.05
RUNTIME_ROI_IOU = 0.5
RUNTIME_ROI_IMGSZ = 960
RUNTIME_ROI_EXPAND_X = 0.26
RUNTIME_ROI_EXPAND_Y = 0.16
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
CONTACT_COLUMNS = 3
CONTACT_CELL_WIDTH = 420
CONTACT_IMAGE_HEIGHT = 260
CONTACT_LABEL_HEIGHT = 76


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--evaluation",
    action="append",
    default=[],
    help=(
      "Sequence-evaluation JSON to audit. Repeat for multiple folds. By "
      "default, discovers balanced48 fold 1-4 artifacts under runs/."
    ),
  )
  parser.add_argument(
    "--annotations",
    default="data/full_image_digit_dataset/manifests/annotations.csv",
  )
  parser.add_argument(
    "--source-images",
    default="data/roi_dataset/images",
  )
  parser.add_argument(
    "--output-root",
    default="../output/full-image-digit-error-audit",
  )
  parser.add_argument(
    "--timestamp",
    default="",
    help="Optional output-folder name. Defaults to the current UTC timestamp.",
  )
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--imgsz", type=int, default=1280)
  parser.add_argument("--confidence", type=float, default=0.25)
  parser.add_argument("--iou", type=float, default=0.7)
  parser.add_argument("--max-detections", type=int, default=300)
  parser.add_argument(
    "--roi-model",
    default="models/roi-rotaug-e30-640.pt",
    help="Production ROI checkpoint used by the end-to-end cascade.",
  )
  parser.add_argument("--roi-device", default="cpu")
  parser.add_argument("--roi-confidence", type=float, default=RUNTIME_ROI_CONFIDENCE)
  parser.add_argument("--roi-iou", type=float, default=RUNTIME_ROI_IOU)
  parser.add_argument("--roi-imgsz", type=int, default=RUNTIME_ROI_IMGSZ)
  parser.add_argument("--roi-expand-x", type=float, default=RUNTIME_ROI_EXPAND_X)
  parser.add_argument("--roi-expand-y", type=float, default=RUNTIME_ROI_EXPAND_Y)
  parser.add_argument(
    "--skip-roi-cascade",
    action="store_true",
    help="Skip production ROI detector to register-crop cascade inference.",
  )
  parser.add_argument(
    "--skip-oracles",
    action="store_true",
    help="Build the visual audit without register- and aperture-crop inference.",
  )
  return parser.parse_args()


def normalized_box(
  item: dict[str, object],
) -> tuple[float, float, float, float]:
  x_center = float(item["x_center"])
  y_center = float(item["y_center"])
  width = float(item["width"])
  height = float(item["height"])
  return (
    x_center - width * 0.5,
    y_center - height * 0.5,
    x_center + width * 0.5,
    y_center + height * 0.5,
  )


def intersection_over_union(
  first: tuple[float, float, float, float],
  second: tuple[float, float, float, float],
) -> float:
  left = max(first[0], second[0])
  top = max(first[1], second[1])
  right = min(first[2], second[2])
  bottom = min(first[3], second[3])
  intersection = max(0.0, right - left) * max(0.0, bottom - top)
  first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
  second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
  union = first_area + second_area - intersection
  return intersection / union if union > 0 else 0.0


def match_detections_by_iou(
  truth_rows: list[dict[str, str]],
  detections: list[dict[str, object]],
  threshold: float = MATCH_IOU_THRESHOLD,
) -> tuple[dict[int, tuple[int, float]], list[int], list[int]]:
  candidates = []
  for truth_index, detection_index in product(
    range(len(truth_rows)),
    range(len(detections)),
  ):
    score = intersection_over_union(
      normalized_box(truth_rows[truth_index]),
      normalized_box(detections[detection_index]),
    )
    if score >= threshold:
      candidates.append((score, truth_index, detection_index))

  matches: dict[int, tuple[int, float]] = {}
  used_detections: set[int] = set()
  for score, truth_index, detection_index in sorted(candidates, reverse=True):
    if truth_index in matches or detection_index in used_detections:
      continue
    matches[truth_index] = (detection_index, score)
    used_detections.add(detection_index)

  unmatched_truth = [index for index in range(len(truth_rows)) if index not in matches]
  unmatched_detections = [
    index for index in range(len(detections))
    if index not in used_detections
  ]
  return matches, unmatched_truth, unmatched_detections


def failure_bucket(
  record: dict[str, object],
  matched_truth_count: int,
) -> str:
  if bool(record["exact_match"]):
    return "exact"
  detection_count = int(record["detection_count"])
  if detection_count < 4:
    return "no-read-missing-detection"
  if detection_count > 4:
    return "no-read-extra-detection"
  if matched_truth_count < 4:
    return "readable-localization-error"
  return "readable-classification-error"


def choose_aperture_detection(
  detections: list[dict[str, object]],
) -> dict[str, object] | None:
  if not detections:
    return None

  def score(item: dict[str, object]) -> tuple[int, float, float]:
    x_center = float(item["x_center"])
    y_center = float(item["y_center"])
    central = int(0.2 <= x_center <= 0.8 and 0.2 <= y_center <= 0.8)
    center_distance = abs(x_center - 0.5) + abs(y_center - 0.5)
    return central, float(item["confidence"]), -center_distance

  return max(detections, key=score)


def evaluate_runtime_roi_sanity(
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
    ("center_x", RUNTIME_ROI_SANITY["min_center_x"], RUNTIME_ROI_SANITY["max_center_x"]),
    ("center_y", RUNTIME_ROI_SANITY["min_center_y"], RUNTIME_ROI_SANITY["max_center_y"]),
    ("area", RUNTIME_ROI_SANITY["min_area"], RUNTIME_ROI_SANITY["max_area"]),
    ("aspect", RUNTIME_ROI_SANITY["min_aspect"], RUNTIME_ROI_SANITY["max_aspect"]),
  ]
  for metric, minimum, maximum in checks:
    if geometry[metric] < minimum or geometry[metric] > maximum:
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


def normalized_bbox_crop_bounds(
  bbox: dict[str, float],
  image_width: int,
  image_height: int,
) -> tuple[int, int, int, int]:
  return (
    max(0, math.floor(bbox["x"] * image_width)),
    max(0, math.floor(bbox["y"] * image_height)),
    min(image_width, math.ceil((bbox["x"] + bbox["width"]) * image_width)),
    min(image_height, math.ceil((bbox["y"] + bbox["height"]) * image_height)),
  )


def map_crop_detection_to_full_image(
  detection: dict[str, object],
  crop_bbox: dict[str, float],
) -> dict[str, object]:
  return {
    **detection,
    "x_center": crop_bbox["x"] + float(detection["x_center"]) * crop_bbox["width"],
    "y_center": crop_bbox["y"] + float(detection["y_center"]) * crop_bbox["height"],
    "width": float(detection["width"]) * crop_bbox["width"],
    "height": float(detection["height"]) * crop_bbox["height"],
  }


def discover_evaluation_paths(
  base_dir: Path,
  values: list[str],
) -> list[Path]:
  if values:
    paths = [resolve_path(base_dir, value) for value in values]
  else:
    paths = sorted(base_dir.glob(
      "runs/full-image-digit-detector-balanced48-crops-fold*/"
      "sequence_evaluation_fold*.json"
    ))
  if not paths:
    raise FileNotFoundError("No sequence-evaluation JSON artifacts were found.")
  missing = [str(path) for path in paths if not path.exists()]
  if missing:
    raise FileNotFoundError("Missing evaluation artifacts: " + ", ".join(missing))
  return paths


def load_evaluations(paths: list[Path]) -> list[dict[str, object]]:
  payloads = []
  seen_folds: set[int] = set()
  seen_filenames: set[str] = set()
  annotation_hashes: set[str] = set()
  for path in paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    fold = int(payload["fold"])
    if fold in seen_folds:
      raise ValueError(f"Duplicate evaluation fold: {fold}")
    seen_folds.add(fold)
    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or not predictions:
      raise ValueError(f"Evaluation has no predictions: {path}")
    for record in predictions:
      filename = str(record["filename"])
      if filename in seen_filenames:
        raise ValueError(f"Evaluation filename appears in multiple folds: {filename}")
      seen_filenames.add(filename)
    annotation_hashes.add(str(payload.get("annotations_sha256") or ""))
    payload["audit_evaluation_path"] = str(path)
    payload["audit_evaluation_sha256"] = file_sha256(path)
    payloads.append(payload)
  if len(annotation_hashes) != 1 or "" in annotation_hashes:
    raise ValueError("Evaluation artifacts do not share one recorded annotation hash.")
  return sorted(payloads, key=lambda payload: int(payload["fold"]))


def reviewed_rows_for_predictions(
  annotation_rows: list[dict[str, str]],
  evaluations: list[dict[str, object]],
) -> dict[str, list[dict[str, str]]]:
  records = {
    str(record["filename"]): record
    for payload in evaluations
    for record in payload["predictions"]
  }
  grouped: dict[str, list[dict[str, str]]] = {filename: [] for filename in records}
  for row in annotation_rows:
    filename = row["filename"]
    if filename in grouped:
      grouped[filename].append(row)

  for filename, rows in grouped.items():
    rows.sort(key=lambda row: int(row["position"]))
    if len(rows) != 4:
      raise ValueError(f"Expected four reviewed rows for {filename}, got {len(rows)}")
    if any(row.get("review_status") != "reviewed" for row in rows):
      raise ValueError(f"Audited annotation is not reviewed: {filename}")
    reading = "".join(row["digit"] for row in rows)
    if reading != str(records[filename]["truth"]):
      raise ValueError(
        f"Current reviewed reading differs from frozen evaluation for {filename}: "
        f"{reading} != {records[filename]['truth']}"
      )
  return grouped


def register_crop_bounds(
  rows: list[dict[str, str]],
  image_width: int,
  image_height: int,
  context_ratio: float = REGISTER_CONTEXT_RATIO,
) -> tuple[int, int, int, int]:
  boxes = [annotation_box_pixels(row, image_width, image_height) for row in rows]
  left = min(box[0] for box in boxes)
  top = min(box[1] for box in boxes)
  right = max(box[2] for box in boxes)
  bottom = max(box[3] for box in boxes)
  longest_side = max(right - left, bottom - top)
  padding = longest_side * context_ratio
  return (
    max(0, math.floor(left - padding)),
    max(0, math.floor(top - padding)),
    min(image_width, math.ceil(right + padding)),
    min(image_height, math.ceil(bottom + padding)),
  )


def source_image_path(
  source_images_root: Path,
  rows: list[dict[str, str]],
) -> Path:
  path = source_images_root / rows[0]["split"] / rows[0]["filename"]
  if not path.exists():
    raise FileNotFoundError(f"Missing source image: {path}")
  return path


def prepare_oracle_images(
  records: list[dict[str, object]],
  annotation_groups: dict[str, list[dict[str, str]]],
  source_images_root: Path,
) -> tuple[list[Image.Image], list[Image.Image], list[tuple[str, int]]]:
  register_images: list[Image.Image] = []
  aperture_images: list[Image.Image] = []
  aperture_keys: list[tuple[str, int]] = []
  for record in records:
    filename = str(record["filename"])
    rows = annotation_groups[filename]
    path = source_image_path(source_images_root, rows)
    with Image.open(path) as source:
      source.load()
      image = source.convert("RGB")
      register_images.append(image.crop(register_crop_bounds(rows, *image.size)))
      for row in rows:
        bounds = digit_centered_crop_bounds(rows, row, *image.size, variant_index=0)
        aperture_images.append(image.crop(bounds))
        aperture_keys.append((filename, int(row["position"])))
  return register_images, aperture_images, aperture_keys


def truth_register_bbox(rows: list[dict[str, str]]) -> dict[str, float]:
  boxes = [normalized_box(row) for row in rows]
  left = min(box[0] for box in boxes)
  top = min(box[1] for box in boxes)
  right = max(box[2] for box in boxes)
  bottom = max(box[3] for box in boxes)
  return {
    "x": left,
    "y": top,
    "width": right - left,
    "height": bottom - top,
  }


def bbox_tuple(bbox: dict[str, float]) -> tuple[float, float, float, float]:
  return (
    bbox["x"],
    bbox["y"],
    bbox["x"] + bbox["width"],
    bbox["y"] + bbox["height"],
  )


def bbox_coverage(
  container: dict[str, float],
  target: dict[str, float],
) -> float:
  container_box = bbox_tuple(container)
  target_box = bbox_tuple(target)
  left = max(container_box[0], target_box[0])
  top = max(container_box[1], target_box[1])
  right = min(container_box[2], target_box[2])
  bottom = min(container_box[3], target_box[3])
  intersection = max(0.0, right - left) * max(0.0, bottom - top)
  target_area = target["width"] * target["height"]
  return intersection / target_area if target_area > 0 else 0.0


def prepare_roi_cascade_images(
  records: list[dict[str, object]],
  annotation_groups: dict[str, list[dict[str, str]]],
  source_images_root: Path,
  roi_detector: object,
  args: argparse.Namespace,
) -> tuple[list[Image.Image], list[str], dict[str, dict[str, object]]]:
  cascade_images: list[Image.Image] = []
  cascade_filenames: list[str] = []
  roi_metadata: dict[str, dict[str, object]] = {}
  for record in records:
    filename = str(record["filename"])
    rows = annotation_groups[filename]
    path = source_image_path(source_images_root, rows)
    with Image.open(path) as source:
      source.load()
      image = source.convert("RGB")
    detection = roi_detector.detect(
      np.asarray(image),
      conf=args.roi_confidence,
      iou=args.roi_iou,
      imgsz=args.roi_imgsz,
    )
    if detection is None:
      roi_metadata[filename] = {"status": "no-detection"}
      continue
    raw_bbox = detection.to_normalized_bbox(*image.size)
    sane, status, geometry = evaluate_runtime_roi_sanity(raw_bbox)
    metadata: dict[str, object] = {
      "status": status,
      "confidence": float(detection.confidence),
      "class_id": int(detection.class_id),
      "class_name": str(detection.class_name),
      "bbox_norm": raw_bbox,
      "geometry": geometry,
    }
    if not sane:
      roi_metadata[filename] = metadata
      continue
    crop_bbox = expand_normalized_bbox(
      raw_bbox,
      args.roi_expand_x,
      args.roi_expand_y,
    )
    truth_bbox = truth_register_bbox(rows)
    metadata.update({
      "crop_bbox_norm": crop_bbox,
      "truth_register_iou": intersection_over_union(
        bbox_tuple(raw_bbox),
        bbox_tuple(truth_bbox),
      ),
      "truth_register_coverage": bbox_coverage(crop_bbox, truth_bbox),
    })
    crop_bounds = normalized_bbox_crop_bounds(crop_bbox, *image.size)
    if crop_bounds[2] <= crop_bounds[0] or crop_bounds[3] <= crop_bounds[1]:
      metadata["status"] = "empty-crop"
      roi_metadata[filename] = metadata
      continue
    cascade_images.append(image.crop(crop_bounds))
    cascade_filenames.append(filename)
    roi_metadata[filename] = metadata
  return cascade_images, cascade_filenames, roi_metadata


def run_oracles(
  evaluations: list[dict[str, object]],
  annotation_groups: dict[str, list[dict[str, str]]],
  source_images_root: Path,
  args: argparse.Namespace,
) -> tuple[
  dict[str, dict[str, object]],
  dict[tuple[str, int], dict[str, object]],
  dict[str, dict[str, object]],
]:
  try:
    from ultralytics import YOLO
  except ImportError as error:
    raise RuntimeError("ultralytics is required for oracle inference.") from error

  matplotlib_cache = Path(tempfile.gettempdir()) / "jarvis-matplotlib-cache"
  matplotlib_cache.mkdir(parents=True, exist_ok=True)
  os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
  device = resolve_device(args.device)
  register_records: dict[str, dict[str, object]] = {}
  aperture_records: dict[tuple[str, int], dict[str, object]] = {}
  cascade_records: dict[str, dict[str, object]] = {}
  roi_detector = None
  if not args.skip_roi_cascade:
    try:
      from backend.detector import RoiDetector
    except ModuleNotFoundError:
      from detector import RoiDetector
    roi_model_path = resolve_path(base_dir=Path(__file__).resolve().parent, value=args.roi_model)
    roi_detector = RoiDetector(
      roi_model_path,
      device=resolve_device(args.roi_device),
    )

  for payload in evaluations:
    fold = int(payload["fold"])
    records = list(payload["predictions"])
    checkpoint_path = Path(str(payload["checkpoint"]))
    if not checkpoint_path.exists():
      raise FileNotFoundError(f"Missing fold {fold} checkpoint: {checkpoint_path}")
    model = YOLO(str(checkpoint_path))
    if not args.skip_oracles:
      register_images, aperture_images, aperture_keys = prepare_oracle_images(
        records,
        annotation_groups,
        source_images_root,
      )
      register_results = model.predict(
        source=register_images,
        imgsz=args.imgsz,
        device=device,
        conf=args.confidence,
        iou=args.iou,
        max_det=args.max_detections,
        agnostic_nms=True,
        save=False,
        verbose=False,
      )
      if len(register_results) != len(records):
        raise RuntimeError(f"Fold {fold} register-oracle result count differs.")
      for record, result in zip(records, register_results, strict=True):
        filename = str(record["filename"])
        rows = annotation_groups[filename]
        register_record = build_sequence_record(
          filename,
          str(record["truth"]),
          int(rows[0]["direction_rotation"]),
          detections_from_result(result),
        )
        register_records[filename] = register_record

      aperture_results = model.predict(
        source=aperture_images,
        imgsz=args.imgsz,
        device=device,
        conf=args.confidence,
        iou=args.iou,
        max_det=args.max_detections,
        agnostic_nms=True,
        save=False,
        verbose=False,
      )
      if len(aperture_results) != len(aperture_keys):
        raise RuntimeError(f"Fold {fold} aperture-oracle result count differs.")
      for key, result in zip(aperture_keys, aperture_results, strict=True):
        filename, position = key
        truth_digit = int(annotation_groups[filename][position]["digit"])
        detections = detections_from_result(result)
        selected = choose_aperture_detection(detections)
        predicted_digit = int(selected["class_id"]) if selected else None
        aperture_records[key] = {
          "truth_digit": truth_digit,
          "predicted_digit": predicted_digit,
          "correct": predicted_digit == truth_digit,
          "confidence": float(selected["confidence"]) if selected else None,
          "detection_count": len(detections),
        }

    if roi_detector is not None:
      cascade_images, cascade_filenames, roi_metadata = prepare_roi_cascade_images(
        records,
        annotation_groups,
        source_images_root,
        roi_detector,
        args,
      )
      cascade_results = predict_runtime_images_one_at_a_time(
        model,
        cascade_images,
        imgsz=args.imgsz,
        device=device,
        confidence=args.confidence,
        iou=args.iou,
        max_detections=args.max_detections,
      )
      if len(cascade_results) != len(cascade_filenames):
        raise RuntimeError(f"Fold {fold} ROI-cascade result count differs.")
      result_by_filename = dict(zip(cascade_filenames, cascade_results, strict=True))
      for record in records:
        filename = str(record["filename"])
        rows = annotation_groups[filename]
        result = result_by_filename.get(filename)
        detections = detections_from_result(result) if result is not None else []
        cascade_record = build_sequence_record(
          filename,
          str(record["truth"]),
          int(rows[0]["direction_rotation"]),
          detections,
        )
        cascade_record["roi"] = roi_metadata[filename]
        crop_bbox = roi_metadata[filename].get("crop_bbox_norm")
        cascade_record["detections_full_image"] = (
          [
            map_crop_detection_to_full_image(detection, crop_bbox)
            for detection in detections
          ]
          if isinstance(crop_bbox, dict)
          else []
        )
        cascade_records[filename] = cascade_record
  return register_records, aperture_records, cascade_records


def predict_runtime_images_one_at_a_time(
  model: object,
  images: list[Image.Image],
  *,
  imgsz: int,
  device: str | None,
  confidence: float,
  iou: float,
  max_detections: int,
) -> list[object]:
  """Mirror the deployed endpoint's one-upload-per-inference behavior."""
  results: list[object] = []
  for image in images:
    prediction = model.predict(
      source=np.asarray(image),
      imgsz=imgsz,
      device=device,
      conf=confidence,
      iou=iou,
      max_det=max_detections,
      agnostic_nms=True,
      save=False,
      verbose=False,
    )
    if len(prediction) != 1:
      raise RuntimeError("Single-image runtime inference returned an unexpected result count.")
    results.append(prediction[0])
  return results


def build_audit_rows(
  evaluations: list[dict[str, object]],
  annotation_groups: dict[str, list[dict[str, str]]],
  register_records: dict[str, dict[str, object]],
  aperture_records: dict[tuple[str, int], dict[str, object]],
  cascade_records: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
  audit_rows = []
  for payload in evaluations:
    fold = int(payload["fold"])
    for record in payload["predictions"]:
      filename = str(record["filename"])
      truth_rows = annotation_groups[filename]
      detections = list(record["detections"])
      matches, unmatched_truth, unmatched_detections = match_detections_by_iou(
        truth_rows,
        detections,
      )
      positions = []
      for truth_index, truth_row in enumerate(truth_rows):
        matched = matches.get(truth_index)
        detection = detections[matched[0]] if matched else None
        aperture = aperture_records.get((filename, truth_index))
        positions.append({
          "position": truth_index,
          "truth_digit": int(truth_row["digit"]),
          "transition_state": truth_row.get("transition_state") or "unknown",
          "full_image_predicted_digit": (
            int(detection["class_id"]) if detection is not None else None
          ),
          "full_image_confidence": (
            float(detection["confidence"]) if detection is not None else None
          ),
          "full_image_iou": matched[1] if matched else None,
          "aperture_oracle": aperture,
        })
      register_record = register_records.get(filename)
      cascade_record = cascade_records.get(filename)
      audit_rows.append({
        "fold": fold,
        "filename": filename,
        "truth": str(record["truth"]),
        "full_image_predicted": record["predicted"],
        "full_image_status": record["status"],
        "full_image_exact": bool(record["exact_match"]),
        "full_image_absolute_error": record["absolute_error"],
        "detection_count": int(record["detection_count"]),
        "matched_truth_count": len(matches),
        "unmatched_truth_positions": unmatched_truth,
        "unmatched_detection_indices": unmatched_detections,
        "failure_bucket": failure_bucket(record, len(matches)),
        "direction_rotation": int(record["direction_rotation"]),
        "detections": detections,
        "positions": positions,
        "register_oracle": register_record,
        "roi_cascade": cascade_record,
      })
  return sorted(
    audit_rows,
    key=lambda row: (
      row["failure_bucket"] == "exact",
      -int(row["full_image_absolute_error"] or 0),
      int(row["fold"]),
      str(row["filename"]),
    ),
  )


def count_confusions(
  rows: list[dict[str, object]],
) -> tuple[Counter[str], Counter[str]]:
  digit_confusions: Counter[str] = Counter()
  position_confusions: Counter[str] = Counter()
  for row in rows:
    for position in row["positions"]:
      predicted = position["full_image_predicted_digit"]
      truth = position["truth_digit"]
      if predicted is None or predicted == truth:
        continue
      digit_confusions[f"{truth}->{predicted}"] += 1
      position_confusions[
        f"p{position['position']}:{truth}->{predicted}"
      ] += 1
  return digit_confusions, position_confusions


def aggregate_summary(
  evaluations: list[dict[str, object]],
  rows: list[dict[str, object]],
  register_records: dict[str, dict[str, object]],
  aperture_records: dict[tuple[str, int], dict[str, object]],
  cascade_records: dict[str, dict[str, object]],
) -> dict[str, object]:
  full_records = [
    record
    for payload in evaluations
    for record in payload["predictions"]
  ]
  failure_counts = Counter(str(row["failure_bucket"]) for row in rows)
  no_read_detection_counts = Counter(
    str(row["detection_count"])
    for row in rows
    if row["full_image_predicted"] is None
  )
  digit_confusions, position_confusions = count_confusions(rows)
  transition_counts = Counter(
    str(position["transition_state"])
    for row in rows
    for position in row["positions"]
  )
  summary: dict[str, object] = {
    "full_image_sequence_metrics": summarize_sequence_records(full_records),
    "failure_bucket_counts": dict(sorted(failure_counts.items())),
    "no_read_detection_counts": dict(sorted(no_read_detection_counts.items())),
    "digit_confusions": dict(digit_confusions.most_common()),
    "position_confusions": dict(position_confusions.most_common()),
    "transition_state_counts": dict(sorted(transition_counts.items())),
    "fold_metrics": {
      str(int(payload["fold"])): {
        "full_image": summarize_sequence_records(list(payload["predictions"])),
      }
      for payload in evaluations
    },
  }
  if register_records:
    register_metrics = summarize_sequence_records(
      list(register_records.values())
    )
    summary["register_oracle_sequence_metrics"] = register_metrics
    for payload in evaluations:
      fold = str(int(payload["fold"]))
      fold_register_records = [
        register_records[str(record["filename"])]
        for record in payload["predictions"]
      ]
      summary["fold_metrics"][fold]["register_oracle"] = (
        summarize_sequence_records(fold_register_records)
      )
    absolute_errors = sorted(
      int(record["absolute_error"])
      for record in register_records.values()
      if record["absolute_error"] is not None
    )
    midpoint = len(absolute_errors) // 2
    median_absolute_error = (
      float(absolute_errors[midpoint])
      if len(absolute_errors) % 2
      else (absolute_errors[midpoint - 1] + absolute_errors[midpoint]) / 2
    )
    errors_without_worst = absolute_errors[:-1]
    rescued_exact = sum(
      not bool(row["full_image_exact"])
      and bool(register_records[str(row["filename"])]["exact_match"])
      for row in rows
    )
    worst_record = max(
      (
        record
        for record in register_records.values()
        if record["absolute_error"] is not None
      ),
      key=lambda record: int(record["absolute_error"]),
    )
    summary["register_oracle_diagnostics"] = {
      "rescued_exact_count": rescued_exact,
      "exact_match_rate_lift": (
        float(register_metrics["exact_match_rate"])
        - float(summary["full_image_sequence_metrics"]["exact_match_rate"])
      ),
      "no_read_rate_reduction": (
        float(summary["full_image_sequence_metrics"]["no_read_rate"])
        - float(register_metrics["no_read_rate"])
      ),
      "median_absolute_error": median_absolute_error,
      "mean_absolute_error_excluding_worst": (
        sum(errors_without_worst) / len(errors_without_worst)
        if errors_without_worst
        else None
      ),
      "worst_absolute_error": int(worst_record["absolute_error"]),
      "worst_filename": str(worst_record["filename"]),
      "worst_truth": str(worst_record["truth"]),
      "worst_prediction": str(worst_record["predicted"]),
    }
  if cascade_records:
    cascade_metrics = summarize_sequence_records(list(cascade_records.values()))
    summary["roi_cascade_sequence_metrics"] = cascade_metrics
    for payload in evaluations:
      fold = str(int(payload["fold"]))
      fold_cascade_records = [
        cascade_records[str(record["filename"])]
        for record in payload["predictions"]
      ]
      summary["fold_metrics"][fold]["roi_cascade"] = (
        summarize_sequence_records(fold_cascade_records)
      )
    roi_status_counts = Counter(
      str(record["roi"]["status"])
      for record in cascade_records.values()
    )
    coverages = sorted(
      float(record["roi"]["truth_register_coverage"])
      for record in cascade_records.values()
      if record["roi"].get("truth_register_coverage") is not None
    )
    coverage_midpoint = len(coverages) // 2
    median_coverage = (
      coverages[coverage_midpoint]
      if len(coverages) % 2
      else (coverages[coverage_midpoint - 1] + coverages[coverage_midpoint]) / 2
    ) if coverages else None
    cascade_rescued_exact = sum(
      not bool(row["full_image_exact"])
      and bool(cascade_records[str(row["filename"])]["exact_match"])
      for row in rows
    )
    cascade_absolute_errors = sorted(
      int(record["absolute_error"])
      for record in cascade_records.values()
      if record["absolute_error"] is not None
    )
    cascade_midpoint = len(cascade_absolute_errors) // 2
    cascade_median_absolute_error = (
      (
        float(cascade_absolute_errors[cascade_midpoint])
        if len(cascade_absolute_errors) % 2
        else (
          cascade_absolute_errors[cascade_midpoint - 1]
          + cascade_absolute_errors[cascade_midpoint]
        ) / 2
      )
      if cascade_absolute_errors
      else None
    )
    cascade_errors_without_worst = cascade_absolute_errors[:-1]
    cascade_worst_record = max(
      (
        record
        for record in cascade_records.values()
        if record["absolute_error"] is not None
      ),
      key=lambda record: int(record["absolute_error"]),
      default=None,
    )
    summary["roi_cascade_diagnostics"] = {
      "roi_status_counts": dict(sorted(roi_status_counts.items())),
      "median_truth_register_coverage": median_coverage,
      "minimum_truth_register_coverage": min(coverages) if coverages else None,
      "rescued_exact_count": cascade_rescued_exact,
      "median_absolute_error": cascade_median_absolute_error,
      "mean_absolute_error_excluding_worst": (
        sum(cascade_errors_without_worst) / len(cascade_errors_without_worst)
        if cascade_errors_without_worst
        else None
      ),
      "worst_absolute_error": (
        int(cascade_worst_record["absolute_error"])
        if cascade_worst_record
        else None
      ),
      "worst_filename": (
        str(cascade_worst_record["filename"])
        if cascade_worst_record
        else None
      ),
      "worst_truth": (
        str(cascade_worst_record["truth"])
        if cascade_worst_record
        else None
      ),
      "worst_prediction": (
        str(cascade_worst_record["predicted"])
        if cascade_worst_record
        else None
      ),
      "exact_match_rate_lift_over_full_image": (
        float(cascade_metrics["exact_match_rate"])
        - float(summary["full_image_sequence_metrics"]["exact_match_rate"])
      ),
      "exact_match_gap_to_register_oracle": (
        int(summary["register_oracle_sequence_metrics"]["exact_match_count"])
        - int(cascade_metrics["exact_match_count"])
      ) if register_records else None,
    }
  if aperture_records:
    predicted = [
      record
      for record in aperture_records.values()
      if record["predicted_digit"] is not None
    ]
    correct = sum(bool(record["correct"]) for record in aperture_records.values())
    summary["aperture_oracle_metrics"] = {
      "aperture_count": len(aperture_records),
      "predicted_count": len(predicted),
      "no_detection_count": len(aperture_records) - len(predicted),
      "correct_count": correct,
      "accuracy": correct / len(aperture_records) if aperture_records else None,
      "readable_accuracy": correct / len(predicted) if predicted else None,
    }
  if register_records and cascade_records:
    cascade_metrics = summary["roi_cascade_sequence_metrics"]
    cascade_diagnostics = summary["roi_cascade_diagnostics"]
    oracle_exact = int(register_metrics["exact_match_count"])
    cascade_exact = int(cascade_metrics["exact_match_count"])
    exact_gap = oracle_exact - cascade_exact
    minimum_coverage = float(cascade_diagnostics["minimum_truth_register_coverage"])
    if minimum_coverage >= 0.98 and int(cascade_metrics["no_read_count"]) > 3:
      finding = (
        "The production ROI crops fully cover every reviewed register, but "
        "single-image runtime inference still has a material no-read gap."
      )
      recommended_next_step = (
        "Audit and standardize the detector's single-image padding/canvas "
        "behavior; do not tune ROI expansion when register coverage is already complete."
      )
    elif exact_gap <= 2 and int(cascade_metrics["no_read_count"]) <= 3:
      finding = (
        "The production ROI cascade preserves most of the register-context "
        "oracle gain; full-image scale and localization were the main bottleneck."
      )
      recommended_next_step = (
        "Run npm run qa:full-image-digit-shadow with one explicitly configured "
        "checkpoint, then judge its leakage-safe validation-fold slice before "
        "considering any promotion."
      )
    else:
      finding = (
        "The production ROI cascade improves scale but leaves a material gap to "
        "the ground-truth register-context oracle."
      )
      recommended_next_step = (
        "Inspect the cascade overlays with the lowest register coverage, then "
        "tune ROI expansion or crop geometry before any retraining."
      )
    summary["decision"] = {
      "finding": finding,
      "supporting_findings": [
        (
          f"The production ROI cascade reads {cascade_metrics['readable_count']}/"
          f"{cascade_metrics['image_count']} images and is exact on "
          f"{cascade_metrics['exact_match_count']}, rescuing "
          f"{cascade_diagnostics['rescued_exact_count']} full-image failures."
        ),
        (
          f"Register-context inference reads {register_metrics['readable_count']}/"
          f"{register_metrics['image_count']} images and is exact on "
          f"{register_metrics['exact_match_count']}, rescuing {rescued_exact} "
          "previous failures."
        ),
        (
          "The production-expanded ROI crops cover at least "
          f"{minimum_coverage:.1%} of every reviewed register."
        ),
        (
          "Single-aperture inference is correct on only "
          f"{correct}/{len(aperture_records)} crops, so the next runtime design "
          "should preserve whole-register context."
        ) if aperture_records else "Single-aperture inference was skipped.",
        (
          f"All {sum(transition_counts.values())} audited transition states remain "
          "unknown; the supplied worksheet is the next label-quality check."
        ),
      ],
      "recommended_next_step": recommended_next_step,
      "promotion_status": (
        "Do not promote yet: transition states remain unreviewed and this is a "
        "small development cross-validation scope, not a locked external test."
      ),
    }
  elif register_records:
    summary["decision"] = {
      "finding": (
        "Digit scale and full-image localization are the dominant bottleneck; "
        "a ground-truth digit-box-derived register-context crop rescues most "
        "sequence failures without retraining."
      ),
      "supporting_findings": [],
      "recommended_next_step": "Run the production ROI cascade evaluation.",
      "promotion_status": "Do not promote from oracle geometry alone.",
    }
  return summary


def load_font(size: int = 22) -> ImageFont.ImageFont:
  try:
    return ImageFont.load_default(size=size)
  except TypeError:
    return ImageFont.load_default()


def draw_label(
  draw: ImageDraw.ImageDraw,
  xy: tuple[int, int],
  text: str,
  color: tuple[int, int, int],
  font: ImageFont.ImageFont,
) -> None:
  left, top, right, bottom = draw.textbbox(xy, text, font=font)
  draw.rectangle((left - 3, top - 2, right + 3, bottom + 2), fill=(15, 23, 42))
  draw.text(xy, text, fill=color, font=font)


def display_oriented_crop(image: Image.Image, rotation: int) -> Image.Image:
  if rotation not in {0, 90, 180, 270}:
    raise ValueError(f"Unsupported display rotation: {rotation}")
  return image.rotate(rotation, expand=True) if rotation else image.copy()


def resize_max(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
  scale = min(max_width / image.width, max_height / image.height, 1.0)
  if scale >= 1:
    return image.copy()
  return image.resize(
    (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
    Image.Resampling.LANCZOS,
  )


def render_row_assets(
  row: dict[str, object],
  truth_rows: list[dict[str, str]],
  source_images_root: Path,
  assets_dir: Path,
) -> None:
  filename = str(row["filename"])
  stem = Path(filename).stem
  path = source_image_path(source_images_root, truth_rows)
  with Image.open(path) as source:
    source.load()
    image = source.convert("RGB")
  width, height = image.size
  draw = ImageDraw.Draw(image)
  font = load_font()
  gt_boxes = []
  for truth_row in truth_rows:
    box = annotation_box_pixels(truth_row, width, height)
    gt_boxes.append(box)
    draw.rectangle(box, outline=(34, 211, 238), width=max(3, round(width / 700)))
    draw_label(
      draw,
      (round(box[0]), max(0, round(box[1]) - 24)),
      f"GT p{truth_row['position']}={truth_row['digit']}",
      (34, 211, 238),
      font,
    )
  prediction_boxes = []
  for detection in row["detections"]:
    normalized = normalized_box(detection)
    box = (
      normalized[0] * width,
      normalized[1] * height,
      normalized[2] * width,
      normalized[3] * height,
    )
    prediction_boxes.append(box)
    draw.rectangle(box, outline=(244, 63, 94), width=max(3, round(width / 700)))
    draw_label(
      draw,
      (round(box[0]), min(height - 28, round(box[3]) + 3)),
      f"P {detection['class_id']} {float(detection['confidence']):.2f}",
      (251, 113, 133),
      font,
    )

  cascade = row.get("roi_cascade") or {}
  roi = cascade.get("roi") or {}
  crop_bbox = roi.get("crop_bbox_norm")
  if isinstance(crop_bbox, dict):
    roi_box = (
      float(crop_bbox["x"]) * width,
      float(crop_bbox["y"]) * height,
      float(crop_bbox["x"] + crop_bbox["width"]) * width,
      float(crop_bbox["y"] + crop_bbox["height"]) * height,
    )
    prediction_boxes.append(roi_box)
    draw.rectangle(roi_box, outline=(34, 197, 94), width=max(3, round(width / 700)))
    draw_label(
      draw,
      (round(roi_box[0]), max(0, round(roi_box[1]) - 50)),
      f"ROI crop {float(roi.get('confidence', 0)):.2f}",
      (74, 222, 128),
      font,
    )
  for detection in cascade.get("detections_full_image", []):
    normalized = normalized_box(detection)
    box = (
      normalized[0] * width,
      normalized[1] * height,
      normalized[2] * width,
      normalized[3] * height,
    )
    prediction_boxes.append(box)
    draw.rectangle(box, outline=(168, 85, 247), width=max(3, round(width / 700)))
    draw_label(
      draw,
      (round(box[0]), min(height - 28, round(box[3]) + 28)),
      f"C {detection['class_id']} {float(detection['confidence']):.2f}",
      (196, 139, 253),
      font,
    )

  all_boxes = gt_boxes + prediction_boxes
  left = min(box[0] for box in all_boxes)
  top = min(box[1] for box in all_boxes)
  right = max(box[2] for box in all_boxes)
  bottom = max(box[3] for box in all_boxes)
  padding = max(right - left, bottom - top) * 0.55
  zoom_bounds = (
    max(0, math.floor(left - padding)),
    max(0, math.floor(top - padding)),
    min(width, math.ceil(right + padding)),
    min(height, math.ceil(bottom + padding)),
  )
  zoom = resize_max(image.crop(zoom_bounds), 1100, 620)
  overlay_dir = assets_dir / "overlays"
  overlay_dir.mkdir(parents=True, exist_ok=True)
  overlay_path = overlay_dir / f"{stem}.jpg"
  zoom.save(overlay_path, "JPEG", quality=92)
  row["overlay_href"] = f"assets/overlays/{overlay_path.name}"

  crop_dir = assets_dir / "crops"
  crop_dir.mkdir(parents=True, exist_ok=True)
  crop_hrefs = []
  rotation = int(row["direction_rotation"])
  with Image.open(path) as source:
    source.load()
    source_rgb = source.convert("RGB")
    for truth_row in truth_rows:
      bounds = digit_centered_crop_bounds(
        truth_rows,
        truth_row,
        *source_rgb.size,
        variant_index=0,
      )
      crop = display_oriented_crop(source_rgb.crop(bounds), rotation)
      if crop.height < 240:
        scale = 240 / crop.height
        crop = crop.resize(
          (max(1, round(crop.width * scale)), 240),
          Image.Resampling.LANCZOS,
        )
      crop_path = crop_dir / f"{stem}__p{truth_row['position']}.jpg"
      crop.save(crop_path, "JPEG", quality=95)
      crop_hrefs.append(f"assets/crops/{crop_path.name}")
  row["crop_hrefs"] = crop_hrefs


def render_contact_sheet(rows: list[dict[str, object]], output_path: Path) -> None:
  row_count = math.ceil(len(rows) / CONTACT_COLUMNS)
  sheet = Image.new(
    "RGB",
    (
      CONTACT_COLUMNS * CONTACT_CELL_WIDTH,
      row_count * (CONTACT_IMAGE_HEIGHT + CONTACT_LABEL_HEIGHT),
    ),
    "#0f172a",
  )
  draw = ImageDraw.Draw(sheet)
  font = load_font(18)
  for index, row in enumerate(rows):
    column = index % CONTACT_COLUMNS
    grid_row = index // CONTACT_COLUMNS
    left = column * CONTACT_CELL_WIDTH
    top = grid_row * (CONTACT_IMAGE_HEIGHT + CONTACT_LABEL_HEIGHT)
    overlay_path = output_path.parent / str(row["overlay_href"])
    with Image.open(overlay_path) as overlay:
      thumb = resize_max(
        overlay.convert("RGB"),
        CONTACT_CELL_WIDTH - 20,
        CONTACT_IMAGE_HEIGHT - 20,
      )
    image_x = left + (CONTACT_CELL_WIDTH - thumb.width) // 2
    image_y = top + 10 + (CONTACT_IMAGE_HEIGHT - 20 - thumb.height) // 2
    sheet.paste(thumb, (image_x, image_y))
    register = row.get("register_oracle") or {}
    cascade = row.get("roi_cascade") or {}
    lines = [
      f"fold {row['fold']} | {row['filename']}",
      f"truth {row['truth']} | full {row['full_image_predicted'] or 'NO-READ'}",
      f"cascade {cascade.get('predicted') or 'NO-READ'} | register {register.get('predicted') or 'n/a'}",
    ]
    for line_index, line in enumerate(lines):
      draw.text(
        (left + 12, top + CONTACT_IMAGE_HEIGHT + line_index * 20),
        line,
        fill=(226, 232, 240),
        font=font,
      )
  sheet.save(output_path, "JPEG", quality=90)


def write_transition_review(
  rows: list[dict[str, object]],
  output_path: Path,
) -> None:
  fieldnames = [
    "filename",
    "fold",
    "position",
    "truth_digit",
    "current_transition_state",
    "reviewed_transition_state",
    "review_notes",
    "full_image_predicted_digit",
    "full_image_confidence",
    "full_image_iou",
    "aperture_oracle_digit",
    "aperture_oracle_confidence",
  ]
  with output_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in sorted(rows, key=lambda item: (int(item["fold"]), str(item["filename"]))):
      for position in row["positions"]:
        aperture = position.get("aperture_oracle") or {}
        writer.writerow({
          "filename": row["filename"],
          "fold": row["fold"],
          "position": position["position"],
          "truth_digit": position["truth_digit"],
          "current_transition_state": position["transition_state"],
          "reviewed_transition_state": "",
          "review_notes": "",
          "full_image_predicted_digit": position["full_image_predicted_digit"],
          "full_image_confidence": position["full_image_confidence"],
          "full_image_iou": position["full_image_iou"],
          "aperture_oracle_digit": aperture.get("predicted_digit"),
          "aperture_oracle_confidence": aperture.get("confidence"),
        })


def metric_badges(metrics: dict[str, object], prefix: str) -> str:
  values = [
    ("images", metrics.get("image_count")),
    ("readable", metrics.get("readable_count")),
    ("no-read", metrics.get("no_read_count")),
    ("exact", metrics.get("exact_match_count")),
    ("digit accuracy", metrics.get("readable_digit_accuracy")),
    ("MAE", metrics.get("readable_mae")),
  ]
  badges = []
  for label, value in values:
    if isinstance(value, float):
      formatted = f"{value:.3f}"
    else:
      formatted = str(value)
    badges.append(f"<code>{html.escape(prefix)} {label}: {html.escape(formatted)}</code>")
  return " ".join(badges)


def write_html_report(
  rows: list[dict[str, object]],
  summary: dict[str, object],
  output_path: Path,
  generated_at: str,
) -> None:
  confusion_rows = "".join(
    f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>"
    for label, count in summary["digit_confusions"].items()
  )
  cards = []
  for row in rows:
    positions = []
    for position, crop_href in zip(row["positions"], row["crop_hrefs"], strict=True):
      aperture = position.get("aperture_oracle") or {}
      full_prediction = position["full_image_predicted_digit"]
      positions.append(f"""
        <figure>
          <img src="{html.escape(crop_href)}" alt="{html.escape(str(row['filename']))} position {position['position']}">
          <figcaption>
            p{position['position']} truth <strong>{position['truth_digit']}</strong><br>
            full <strong>{full_prediction if full_prediction is not None else 'missing'}</strong>
            {f"@ {float(position['full_image_confidence']):.2f}" if position['full_image_confidence'] is not None else ""}<br>
            aperture oracle <strong>{aperture.get('predicted_digit', 'n/a')}</strong>
            {f"@ {float(aperture['confidence']):.2f}" if aperture.get('confidence') is not None else ""}<br>
            transition <strong>{html.escape(str(position['transition_state']))}</strong>
          </figcaption>
        </figure>
      """)
    register = row.get("register_oracle") or {}
    cascade = row.get("roi_cascade") or {}
    roi = cascade.get("roi") or {}
    cards.append(f"""
      <article class="card {html.escape(str(row['failure_bucket']))}">
        <h2>Fold {row['fold']} - {html.escape(str(row['filename']))}</h2>
        <p>
          <code>truth {row['truth']}</code>
          <code>full {row['full_image_predicted'] or 'NO-READ'}</code>
          <code>ROI cascade {cascade.get('predicted') or 'NO-READ'}</code>
          <code>register oracle {register.get('predicted') or 'n/a'}</code>
          <code>ROI {html.escape(str(roi.get('status', 'skipped')))} {float(roi.get('confidence', 0)):.2f}</code>
          <code>detections {row['detection_count']}</code>
          <code>{html.escape(str(row['failure_bucket']))}</code>
        </p>
        <img class="overlay" src="{html.escape(str(row['overlay_href']))}" alt="prediction and reviewed boxes">
        <div class="positions">{''.join(positions)}</div>
      </article>
    """)

  full_metrics = summary["full_image_sequence_metrics"]
  register_metrics = summary.get("register_oracle_sequence_metrics")
  cascade_metrics = summary.get("roi_cascade_sequence_metrics")
  aperture_metrics = summary.get("aperture_oracle_metrics")
  diagnostics = summary.get("register_oracle_diagnostics") or {}
  cascade_diagnostics = summary.get("roi_cascade_diagnostics") or {}
  decision = summary.get("decision") or {}
  supporting_findings = "".join(
    f"<li>{html.escape(str(item))}</li>"
    for item in decision.get("supporting_findings", [])
  )
  oracle_badges = (
    metric_badges(register_metrics, "register")
    if isinstance(register_metrics, dict)
    else "<code>register oracle: skipped</code>"
  )
  cascade_badges = (
    metric_badges(cascade_metrics, "ROI cascade")
    if isinstance(cascade_metrics, dict)
    else "<code>ROI cascade: skipped</code>"
  )
  aperture_badges = "<code>aperture oracle: skipped</code>"
  if isinstance(aperture_metrics, dict):
    aperture_badges = " ".join([
      f"<code>apertures: {aperture_metrics['aperture_count']}</code>",
      f"<code>predicted: {aperture_metrics['predicted_count']}</code>",
      f"<code>correct: {aperture_metrics['correct_count']}</code>",
      f"<code>accuracy: {float(aperture_metrics['accuracy']):.3f}</code>",
    ])
  if diagnostics:
    diagnostic_summary = (
      f"The ground-truth digit-box-derived register-context crop rescued <strong>"
      f"{diagnostics['rescued_exact_count']}</strong> previously failed images. "
      f"Its median absolute error is <strong>"
      f"{diagnostics['median_absolute_error']}</strong>; excluding the single "
      f"worst case, mean absolute error is <strong>"
      f"{float(diagnostics['mean_absolute_error_excluding_worst']):.2f}</strong>."
    )
    worst_summary = (
      f"The overall register-oracle MAE is distorted by <code>"
      f"{html.escape(str(diagnostics['worst_filename']))}</code>: truth <code>"
      f"{diagnostics['worst_truth']}</code>, prediction <code>"
      f"{diagnostics['worst_prediction']}</code>, absolute error <code>"
      f"{diagnostics['worst_absolute_error']}</code>."
    )
  else:
    diagnostic_summary = "Oracle inference was skipped."
    worst_summary = ""
  cascade_diagnostic_summary = ""
  if cascade_diagnostics:
    cascade_diagnostic_summary = (
      f"The ROI cascade median absolute error is <strong>"
      f"{cascade_diagnostics['median_absolute_error']}</strong>; excluding its "
      f"single worst case, mean absolute error is <strong>"
      f"{float(cascade_diagnostics['mean_absolute_error_excluding_worst']):.2f}"
      f"</strong>. The worst case is <code>"
      f"{html.escape(str(cascade_diagnostics['worst_filename']))}</code>: "
      f"<code>{cascade_diagnostics['worst_truth']}</code> to "
      f"<code>{cascade_diagnostics['worst_prediction']}</code>."
    )

  output_path.write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Jarvis Full-Image Digit Error Audit</title>
  <style>
    :root {{ color: #172033; background: #f5efe4; font-family: Georgia, 'Times New Roman', serif; }}
    body {{ margin: 24px; }}
    h1 {{ font-size: 42px; margin-bottom: 8px; }}
    h2 {{ margin-top: 0; }}
    a {{ color: #075985; }}
    .summary, .card {{ background: rgba(255,255,255,.86); border: 1px solid #cbd5e1; border-radius: 16px; padding: 18px; margin: 18px 0; box-shadow: 0 10px 25px rgba(15,23,42,.08); }}
    code {{ display: inline-block; padding: 4px 9px; margin: 3px; border-radius: 999px; background: #eef2f7; border: 1px solid #cbd5e1; }}
    .overlay {{ width: min(100%, 1100px); border-radius: 10px; background: #0f172a; }}
    .positions {{ display: grid; grid-template-columns: repeat(4,minmax(120px,1fr)); gap: 12px; margin-top: 14px; }}
    figure {{ margin: 0; }}
    figure img {{ width: 100%; max-height: 280px; object-fit: contain; background: #111827; border-radius: 8px; }}
    figcaption {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; line-height: 1.45; }}
    .no-read-missing-detection, .no-read-extra-detection {{ border-left: 7px solid #f59e0b; }}
    .readable-classification-error, .readable-localization-error {{ border-left: 7px solid #ef4444; }}
    .exact {{ border-left: 7px solid #22c55e; }}
    table {{ border-collapse: collapse; background: white; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 7px 12px; }}
    @media (max-width: 900px) {{ .positions {{ grid-template-columns: repeat(2,1fr); }} }}
  </style>
</head>
<body>
  <h1>Jarvis Full-Image Digit Error Audit</h1>
  <p>Generated {html.escape(generated_at)} from frozen out-of-fold sequence-evaluation artifacts. Cyan boxes are reviewed truth; pink boxes are full-image predictions; green is the production-expanded ROI crop; purple boxes are cascade predictions.</p>
  <section class="summary">
    <h2>Sequence summary</h2>
    <p>{metric_badges(full_metrics, 'full')}</p>
    <p>{cascade_badges}</p>
    <p>{oracle_badges}</p>
    <p>{aperture_badges}</p>
    <p><a href="contact-sheet.jpg">Open contact sheet</a> | <a href="summary.json">Open JSON</a> | <a href="transition-review.csv">Open transition worksheet</a></p>
    <p>The worksheet is intentionally non-mutating. Review each aperture as <code>stable</code>, <code>transitioning</code>, or <code>uncertain</code> before changing canonical annotations.</p>
  </section>
  <section class="summary">
    <h2>Finding and decision</h2>
    <p><strong>{html.escape(str(decision.get('finding', 'No oracle decision available.')))}</strong></p>
    <ul>{supporting_findings}</ul>
    <p>{cascade_diagnostic_summary}</p>
    <p>{diagnostic_summary}</p>
    <p>{worst_summary}</p>
    <p><strong>Next:</strong> {html.escape(str(decision.get('recommended_next_step', '')))}</p>
    <p><strong>Gate:</strong> {html.escape(str(decision.get('promotion_status', '')))}</p>
  </section>
  <section class="summary">
    <h2>Interpretation gates</h2>
    <ul>
      <li>If register/aperture oracles remove most failures, investigate scale, tiling, or localization before retraining.</li>
      <li>If stable reviewed apertures remain wrong under oracle geometry, collect independent real sources before another model recipe.</li>
      <li>If errors concentrate on transitioning wheels, redesign labels or sequence decoding instead of treating them as ordinary ten-class glyphs.</li>
    </ul>
    <h3>Matched full-image digit confusions</h3>
    <table><thead><tr><th>Truth to prediction</th><th>Count</th></tr></thead><tbody>{confusion_rows}</tbody></table>
  </section>
  {''.join(cards)}
</body>
</html>
""", encoding="utf-8")


def write_markdown_summary(
  summary: dict[str, object],
  output_path: Path,
  generated_at: str,
) -> None:
  full_metrics = summary["full_image_sequence_metrics"]
  register_metrics = summary.get("register_oracle_sequence_metrics") or {}
  cascade_metrics = summary.get("roi_cascade_sequence_metrics") or {}
  aperture_metrics = summary.get("aperture_oracle_metrics") or {}
  diagnostics = summary.get("register_oracle_diagnostics") or {}
  cascade_diagnostics = summary.get("roi_cascade_diagnostics") or {}
  decision = summary.get("decision") or {}
  lines = [
    "# Jarvis Full-Image Digit Error Audit",
    "",
    f"Generated {generated_at} from frozen out-of-fold sequence evaluations.",
    "",
    "## Summary",
    "",
    f"- Full images: {full_metrics.get('readable_count')}/{full_metrics.get('image_count')} readable, {full_metrics.get('exact_match_count')} exact, {full_metrics.get('no_read_count')} no-read, digit accuracy {full_metrics.get('readable_digit_accuracy')}, MAE {full_metrics.get('readable_mae')}.",
    f"- Production ROI cascade: {cascade_metrics.get('readable_count', 'skipped')}/{cascade_metrics.get('image_count', 'skipped')} readable, {cascade_metrics.get('exact_match_count', 'skipped')} exact, {cascade_metrics.get('no_read_count', 'skipped')} no-read, MAE {cascade_metrics.get('readable_mae', 'skipped')}.",
    f"- ROI cascade robust errors: median {cascade_diagnostics.get('median_absolute_error', 'skipped')}; mean excluding the worst case {cascade_diagnostics.get('mean_absolute_error_excluding_worst', 'skipped')}.",
    f"- Register oracle: {register_metrics.get('readable_count', 'skipped')}/{register_metrics.get('image_count', 'skipped')} readable, {register_metrics.get('exact_match_count', 'skipped')} exact, MAE {register_metrics.get('readable_mae', 'skipped')}.",
    f"- Aperture oracle: {aperture_metrics.get('correct_count', 'skipped')}/{aperture_metrics.get('aperture_count', 'skipped')} correct.",
    f"- Register crop rescued {diagnostics.get('rescued_exact_count', 'n/a')} previously failed images; median absolute error {diagnostics.get('median_absolute_error', 'n/a')}; mean excluding the worst case {diagnostics.get('mean_absolute_error_excluding_worst', 'n/a')}.",
    "",
    "## Decision",
    "",
    str(decision.get("finding", "No oracle decision available.")),
    "",
    *[
      f"- {item}"
      for item in decision.get("supporting_findings", [])
    ],
    "",
    f"**Recommended next step:** {decision.get('recommended_next_step', '')}",
    "",
    f"**Promotion gate:** {decision.get('promotion_status', '')}",
    "",
    "## Artifacts",
    "",
    "- [Visual HTML report](full-image-digit-error-audit.html)",
    "- [Contact sheet](contact-sheet.jpg)",
    "- [Machine-readable summary](summary.json)",
    "- [Transition review worksheet](transition-review.csv)",
    "",
    "Do not import transition-review.csv automatically. Review its blank transition fields before updating the canonical annotation manifest.",
    "",
  ]
  output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
  args = parse_args()
  if not 0 <= args.confidence <= 1:
    raise ValueError("--confidence must be in [0, 1].")
  if not 0 < args.iou <= 1:
    raise ValueError("--iou must be in (0, 1].")
  if not 0 <= args.roi_confidence <= 1:
    raise ValueError("--roi-confidence must be in [0, 1].")
  if not 0 < args.roi_iou <= 1:
    raise ValueError("--roi-iou must be in (0, 1].")
  if args.roi_expand_x < 0 or args.roi_expand_y < 0:
    raise ValueError("ROI expansion ratios must be non-negative.")
  base_dir = Path(__file__).resolve().parent
  evaluation_paths = discover_evaluation_paths(base_dir, args.evaluation)
  evaluations = load_evaluations(evaluation_paths)
  annotations_path = resolve_path(base_dir, args.annotations)
  source_images_root = resolve_path(base_dir, args.source_images)
  output_root = resolve_path(base_dir, args.output_root)
  timestamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
  output_dir = output_root / timestamp
  output_dir.mkdir(parents=True, exist_ok=False)
  assets_dir = output_dir / "assets"
  assets_dir.mkdir()

  annotation_rows = read_csv_rows(annotations_path)
  annotation_groups = reviewed_rows_for_predictions(annotation_rows, evaluations)
  if args.skip_oracles and args.skip_roi_cascade:
    register_records: dict[str, dict[str, object]] = {}
    aperture_records: dict[tuple[str, int], dict[str, object]] = {}
    cascade_records: dict[str, dict[str, object]] = {}
  else:
    register_records, aperture_records, cascade_records = run_oracles(
      evaluations,
      annotation_groups,
      source_images_root,
      args,
    )
  rows = build_audit_rows(
    evaluations,
    annotation_groups,
    register_records,
    aperture_records,
    cascade_records,
  )
  for row in rows:
    render_row_assets(
      row,
      annotation_groups[str(row["filename"])],
      source_images_root,
      assets_dir,
    )

  generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
  aggregate = aggregate_summary(
    evaluations,
    rows,
    register_records,
    aperture_records,
    cascade_records,
  )
  payload = {
    "version": 1,
    "generated_at": generated_at,
    "evaluation_scope": "frozen_out_of_fold_sequence_artifacts",
    "recorded_annotations_sha256": evaluations[0]["annotations_sha256"],
    "current_annotations_path": str(annotations_path),
    "current_annotations_sha256": file_sha256(annotations_path),
    "evaluation_inputs": [
      {
        "fold": int(item["fold"]),
        "path": item["audit_evaluation_path"],
        "sha256": item["audit_evaluation_sha256"],
        "checkpoint": item["checkpoint"],
        "checkpoint_sha256": item.get("checkpoint_sha256"),
      }
      for item in evaluations
    ],
    "settings": {
      "match_iou_threshold": MATCH_IOU_THRESHOLD,
      "register_context_ratio": REGISTER_CONTEXT_RATIO,
      "oracles_enabled": not args.skip_oracles,
      "roi_cascade_enabled": not args.skip_roi_cascade,
      "device": args.device,
      "imgsz": args.imgsz,
      "confidence": args.confidence,
      "iou": args.iou,
      "max_detections": args.max_detections,
      "roi_model": str(resolve_path(base_dir, args.roi_model)),
      "roi_model_sha256": (
        file_sha256(resolve_path(base_dir, args.roi_model))
        if not args.skip_roi_cascade
        else None
      ),
      "roi_device": args.roi_device,
      "roi_confidence": args.roi_confidence,
      "roi_iou": args.roi_iou,
      "roi_imgsz": args.roi_imgsz,
      "roi_expand_x": args.roi_expand_x,
      "roi_expand_y": args.roi_expand_y,
      "roi_sanity": RUNTIME_ROI_SANITY,
    },
    "summary": aggregate,
    "rows": rows,
  }
  summary_path = output_dir / "summary.json"
  summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  write_transition_review(rows, output_dir / "transition-review.csv")
  render_contact_sheet(rows, output_dir / "contact-sheet.jpg")
  write_html_report(
    rows,
    aggregate,
    output_dir / "full-image-digit-error-audit.html",
    generated_at,
  )
  write_markdown_summary(aggregate, output_dir / "README.md", generated_at)

  print(json.dumps({
    "output": str(output_dir),
    "summary": aggregate,
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
