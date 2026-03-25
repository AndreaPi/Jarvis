from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
  from .detector import RoiDetector
  from .digit_classifier import DigitClassifier
except ImportError:
  from detector import RoiDetector
  from digit_classifier import DigitClassifier


EXPAND_X = 0.26
EXPAND_Y = 0.16
NORMALIZE_WIDTH = 520
PRIMARY_ANGLES = (90, 270)
EDGE_ROW_MAX_RATIO = 0.65
CELL_COUNT = 4
CELL_OVERLAP = 0.03
MIN_CANDIDATE_WIDTH = 120
MIN_CANDIDATE_HEIGHT = 28
MIN_CANDIDATE_ASPECT = 0.12
MAX_CANDIDATE_ASPECT = 18.0
MIN_STRIP_ASPECT = 1.45
MAX_STRIP_ASPECT = 8.2
DESKEW_MAX_ANGLE = 8
DESKEW_STEP = 2
TIGHTEN_INK_RATIO = 0.08
HARD_STRIP_MIN_FACTOR = 0.96
HARD_STRIP_MAX_FACTOR = 1.06


@dataclass
class Candidate:
  label: str
  image: Image.Image


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Export a train-only runtime failure set from live OCR candidate crops."
  )
  parser.add_argument(
    "--assets-root",
    default="../assets",
    help="Directory containing meter images."
  )
  parser.add_argument(
    "--readings-csv",
    default="../assets/meter_readings.csv",
    help="CSV with filename,value rows."
  )
  parser.add_argument(
    "--output-root",
    default="data/runtime_failure_dataset",
    help="Output root for exported runtime strips/cells and manifests."
  )
  parser.add_argument(
    "--roi-model",
    default="models/roi-rotaug-e30-640.pt",
    help="ROI detector checkpoint path."
  )
  parser.add_argument(
    "--digit-model",
    default="models/digit_classifier.pt",
    help="Digit classifier checkpoint path used to identify failures."
  )
  parser.add_argument(
    "--digit-min-confidence",
    type=float,
    default=0.18,
    help="Frontend-equivalent per-cell confidence threshold for considering a runtime candidate successful."
  )
  parser.add_argument(
    "--roi-conf",
    type=float,
    default=0.05
  )
  parser.add_argument(
    "--roi-iou",
    type=float,
    default=0.5
  )
  parser.add_argument(
    "--roi-imgsz",
    type=int,
    default=960
  )
  parser.add_argument(
    "--filename",
    action="append",
    default=[],
    help="Optional filename filter. Repeat to export only specific benchmark rows."
  )
  return parser.parse_args()


def resolve_path(base_dir: Path, value: str) -> Path:
  path = Path(value)
  if path.is_absolute():
    return path
  return (base_dir / path).resolve()


def load_readings(csv_path: Path) -> list[dict[str, str]]:
  rows: list[dict[str, str]] = []
  with csv_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      filename = (row.get("filename") or "").strip()
      value = (row.get("value") or "").strip()
      if not filename or not value:
        continue
      rows.append({"filename": filename, "value": value})
  return rows


def clamp(value: float, minimum: float, maximum: float) -> float:
  return min(maximum, max(minimum, value))


def normalize_angle(angle: float) -> int:
  return int(((angle % 360) + 360) % 360)


def scale_up_to_width(image: Image.Image, target_width: int) -> Image.Image:
  if image.width >= target_width:
    return image
  scale = target_width / max(1, image.width)
  new_width = max(1, int(round(image.width * scale)))
  new_height = max(1, int(round(image.height * scale)))
  return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


def crop_image(image: Image.Image, rect: dict[str, float]) -> Image.Image:
  x = clamp(rect["x"], 0, image.width - 1)
  y = clamp(rect["y"], 0, image.height - 1)
  width = clamp(rect["width"], 1, image.width - x)
  height = clamp(rect["height"], 1, image.height - y)
  left = int(round(x))
  top = int(round(y))
  right = int(round(x + width))
  bottom = int(round(y + height))
  right = max(left + 1, min(image.width, right))
  bottom = max(top + 1, min(image.height, bottom))
  return image.crop((left, top, right, bottom))


def rotate_image(image: Image.Image, angle: int) -> Image.Image:
  normalized = normalize_angle(angle)
  if normalized == 0:
    return image
  # Match the browser canvas rotation convention used by rotateCanvas():
  # positive angles rotate clockwise on the HTML canvas, while Pillow's
  # ROTATE_90/270 constants are counter-clockwise.
  if normalized == 90:
    return image.transpose(Image.Transpose.ROTATE_270)
  if normalized == 180:
    return image.transpose(Image.Transpose.ROTATE_180)
  if normalized == 270:
    return image.transpose(Image.Transpose.ROTATE_90)
  return image.rotate(normalized, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=255)


def has_valid_candidate_geometry(image: Image.Image) -> bool:
  width = image.width
  height = image.height
  aspect = width / max(1, height)
  return (
    width >= MIN_CANDIDATE_WIDTH
    and height >= MIN_CANDIDATE_HEIGHT
    and MIN_CANDIDATE_ASPECT <= aspect <= MAX_CANDIDATE_ASPECT
  )


def to_grayscale_array(image: Image.Image) -> np.ndarray:
  return np.asarray(image.convert("L"), dtype=np.float32)


def tighten_crop_by_ink(image: Image.Image, min_area_ratio: float = 0.15) -> Image.Image:
  gray = to_grayscale_array(image)
  dark = 255.0 - gray
  cols = dark.sum(axis=0)
  rows = dark.sum(axis=1)
  mean_cols = float(cols.mean())
  mean_rows = float(rows.mean())
  max_cols = float(cols.max())
  max_rows = float(rows.max())
  col_threshold = mean_cols + (max_cols - mean_cols) * 0.25
  row_threshold = mean_rows + (max_rows - mean_rows) * 0.25

  col_indices = np.flatnonzero(cols > col_threshold)
  row_indices = np.flatnonzero(rows > row_threshold)
  if not len(col_indices) or not len(row_indices):
    return image

  left = int(col_indices[0])
  right = int(col_indices[-1])
  top = int(row_indices[0])
  bottom = int(row_indices[-1])
  if right <= left or bottom <= top:
    return image

  padding_x = int(round((right - left) * 0.08))
  padding_y = int(round((bottom - top) * 0.15))
  left = int(clamp(left - padding_x, 0, image.width - 1))
  right = int(clamp(right + padding_x, 1, image.width))
  top = int(clamp(top - padding_y, 0, image.height - 1))
  bottom = int(clamp(bottom + padding_y, 1, image.height))
  crop_width = right - left
  crop_height = bottom - top
  area_ratio = (crop_width * crop_height) / max(1, image.width * image.height)
  if area_ratio < min_area_ratio or area_ratio > 0.95:
    return image

  return image.crop((left, top, right, bottom))


def find_digit_window_by_edges(image: Image.Image) -> dict[str, int] | None:
  gray = to_grayscale_array(image)
  height, width = gray.shape
  max_row = max(2, int(height * EDGE_ROW_MAX_RATIO))
  cols = np.zeros(width, dtype=np.float32)
  rows = np.zeros(height, dtype=np.float32)

  for y in range(1, max_row - 1):
    left = gray[y, :-2]
    right = gray[y, 2:]
    edge = np.abs(right - left)
    cols[1:-1] += edge
    rows[y] += float(edge.sum())

  cols_mean = float(cols.mean())
  rows_mean = float(rows.mean())
  cols_max = float(cols.max())
  rows_max = float(rows.max())
  col_threshold = cols_mean + (cols_max - cols_mean) * 0.35
  row_threshold = rows_mean + (rows_max - rows_mean) * 0.35

  col_indices = np.flatnonzero(cols > col_threshold)
  row_indices = np.flatnonzero(rows > row_threshold)
  if not len(col_indices) or not len(row_indices):
    return None

  left_index = int(col_indices[0])
  right_index = int(col_indices[-1])
  top_index = int(row_indices[0])
  bottom_index = int(row_indices[-1])
  if right_index <= left_index or bottom_index <= top_index:
    return None

  crop_width = right_index - left_index
  crop_height = bottom_index - top_index
  pad_x = int(round(crop_width * 0.08))
  pad_y = int(round(crop_height * 0.24))
  x = int(clamp(left_index - pad_x, 0, width - 1))
  y = int(clamp(top_index - pad_y, 0, height - 1))
  right = int(clamp(right_index + pad_x, 1, width))
  bottom = int(clamp(bottom_index + pad_y, 1, height))
  width_out = right - x
  height_out = bottom - y
  if width_out <= 0 or height_out <= 0:
    return None
  return {"x": x, "y": y, "width": width_out, "height": height_out}


def score_deskew_candidate(image: Image.Image, angle: int) -> tuple[Image.Image, float] | None:
  rotated = image if angle == 0 else rotate_image(image, angle)
  tightened = tighten_crop_by_ink(rotated, TIGHTEN_INK_RATIO)
  aspect = tightened.width / max(1, tightened.height)
  area_ratio = (tightened.width * tightened.height) / max(1, rotated.width * rotated.height)
  score = aspect - max(0.0, 0.14 - area_ratio) * 3.5
  return tightened, score


def normalize_roi_strip(image: Image.Image) -> Image.Image | None:
  best_image, best_score = score_deskew_candidate(image, 0)
  for delta in range(DESKEW_STEP, DESKEW_MAX_ANGLE + 1, DESKEW_STEP):
    for angle in (delta, -delta):
      candidate = score_deskew_candidate(image, angle)
      if candidate is None:
        continue
      candidate_image, candidate_score = candidate
      if candidate_score > best_score + 0.02:
        best_image = candidate_image
        best_score = candidate_score

  normalized = best_image
  if normalized.height > normalized.width:
    normalized = rotate_image(normalized, 90)
  if not has_valid_candidate_geometry(normalized):
    return None

  normalized = scale_up_to_width(normalized, NORMALIZE_WIDTH)
  if normalized.height > normalized.width:
    normalized = rotate_image(normalized, 90)
  if not has_valid_candidate_geometry(normalized):
    return None

  aspect = normalized.width / max(1, normalized.height)
  hard_min = MIN_STRIP_ASPECT * HARD_STRIP_MIN_FACTOR
  hard_max = MAX_STRIP_ASPECT * HARD_STRIP_MAX_FACTOR
  if aspect < hard_min or aspect > hard_max:
    return None
  return normalized


def split_into_cells(image: Image.Image, count: int, overlap_ratio: float) -> list[Image.Image]:
  cells: list[Image.Image] = []
  cell_width = image.width / count
  overlap = cell_width * overlap_ratio
  for index in range(count):
    x = cell_width * index - overlap
    width = cell_width + overlap * 2
    cells.append(crop_image(image, {
      "x": x,
      "y": 0,
      "width": width,
      "height": image.height
    }))
  return cells


def resolve_roi_rect(image: Image.Image, detection) -> dict[str, float]:
  raw_x = detection.x1
  raw_y = detection.y1
  raw_width = max(1.0, detection.x2 - detection.x1)
  raw_height = max(1.0, detection.y2 - detection.y1)
  return {
    "x": raw_x - raw_width * EXPAND_X,
    "y": raw_y - raw_height * EXPAND_Y,
    "width": raw_width * (1 + EXPAND_X * 2),
    "height": raw_height * (1 + EXPAND_Y * 2)
  }


def predict_cells(classifier: DigitClassifier, cells: list[Image.Image], min_confidence: float) -> tuple[str, list[float]]:
  digits: list[str] = []
  confidences: list[float] = []
  for cell in cells:
    rgb = np.asarray(cell.convert("RGB"), dtype=np.uint8)
    prediction = classifier.predict(rgb, top_k=3)
    confidences.append(float(prediction.confidence))
    digits.append(prediction.digit if prediction.confidence >= min_confidence else "")
  return "".join(digits), confidences


def export_runtime_failure_set() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent
  assets_root = resolve_path(base_dir, args.assets_root)
  readings_csv = resolve_path(base_dir, args.readings_csv)
  output_root = resolve_path(base_dir, args.output_root)
  roi_model = resolve_path(base_dir, args.roi_model)
  digit_model = resolve_path(base_dir, args.digit_model)

  rows = load_readings(readings_csv)
  filename_filter = {value.strip() for value in args.filename if value and value.strip()}
  if filename_filter:
    rows = [row for row in rows if row["filename"] in filename_filter]
  detector = RoiDetector(roi_model, device="cpu")
  classifier = DigitClassifier(digit_model, device="cpu")

  if output_root.exists():
    shutil.rmtree(output_root)
  failure_root = output_root / "sections_labeled" / "train"
  strips_root = output_root / "strips"
  manifests_root = output_root / "manifests"
  for digit in range(10):
    (failure_root / str(digit)).mkdir(parents=True, exist_ok=True)
  strips_root.mkdir(parents=True, exist_ok=True)
  manifests_root.mkdir(parents=True, exist_ok=True)

  manifest_rows: list[dict[str, object]] = []
  summary = {
    "images_total": len(rows),
    "images_with_detection": 0,
    "failure_candidates": 0,
    "exported_cells": 0
  }

  for row in rows:
    filename = row["filename"]
    expected = "".join(ch for ch in row["value"] if ch.isdigit())
    image_path = assets_root / filename
    if not image_path.exists() or len(expected) != CELL_COUNT:
      manifest_rows.append({
        "filename": filename,
        "candidate": "",
        "status": "skipped",
        "reason": "missing-image-or-invalid-reading",
        "expected": expected
      })
      continue

    with Image.open(image_path) as source:
      base_image = source.convert("RGB")

    detection = detector.detect(
      image_rgb=np.asarray(base_image, dtype=np.uint8),
      conf=args.roi_conf,
      iou=args.roi_iou,
      imgsz=args.roi_imgsz
    )
    if detection is None:
      manifest_rows.append({
        "filename": filename,
        "candidate": "",
        "status": "no-detection",
        "reason": "roi-miss",
        "expected": expected
      })
      continue

    summary["images_with_detection"] += 1
    roi_rect = resolve_roi_rect(base_image, detection)
    roi_crop = crop_image(base_image, roi_rect)

    candidates: list[Candidate] = [Candidate(label="scan-roi", image=roi_crop)]
    base_fallback: Candidate | None = None
    for angle in PRIMARY_ANGLES:
      rotated = rotate_image(roi_crop, angle)
      edge_rect = find_digit_window_by_edges(rotated)
      if edge_rect is not None:
        edge_crop = crop_image(rotated, edge_rect)
        edge_scaled = scale_up_to_width(edge_crop, NORMALIZE_WIDTH)
        if has_valid_candidate_geometry(edge_scaled):
          candidates.append(Candidate(label=f"roi-{angle}-edge-roi", image=edge_crop))

      if base_fallback is None:
        base_scaled = scale_up_to_width(rotated, NORMALIZE_WIDTH)
        if has_valid_candidate_geometry(base_scaled):
          base_fallback = Candidate(label=f"roi-{angle}-base-roi", image=rotated)
    if base_fallback is not None:
      candidates.append(base_fallback)

    for candidate in candidates:
      normalized = normalize_roi_strip(candidate.image)
      if normalized is None:
        manifest_rows.append({
          "filename": filename,
          "candidate": candidate.label,
          "status": "invalid-strip",
          "reason": "normalize-roi-strip-failed",
          "expected": expected
        })
        continue

      cells = split_into_cells(normalized, CELL_COUNT, CELL_OVERLAP)
      predicted_value, confidences = predict_cells(classifier, cells, args.digit_min_confidence)
      avg_confidence = sum(confidences) / max(1, len(confidences))
      is_failure = predicted_value != expected
      status = "failure" if is_failure else "match"
      strip_stem = f"{Path(filename).stem}__{candidate.label}"
      normalized.save(strips_root / f"{strip_stem}.png")

      manifest_rows.append({
        "filename": filename,
        "candidate": candidate.label,
        "status": status,
        "reason": "" if is_failure else "match",
        "expected": expected,
        "predicted": predicted_value,
        "avg_confidence": round(avg_confidence, 4),
        "cell_confidences": ";".join(f"{value:.4f}" for value in confidences)
      })

      if not is_failure:
        continue

      summary["failure_candidates"] += 1
      for index, cell in enumerate(cells):
        digit = expected[index]
        cell_path = failure_root / digit / f"{strip_stem}__c{index}.png"
        cell.convert("L").save(cell_path)
        summary["exported_cells"] += 1

  manifest_path = manifests_root / "runtime_failure_candidates.csv"
  with manifest_path.open("w", encoding="utf-8", newline="") as handle:
    fieldnames = [
      "filename",
      "candidate",
      "status",
      "reason",
      "expected",
      "predicted",
      "avg_confidence",
      "cell_confidences"
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(manifest_rows)

  summary_path = manifests_root / "summary.json"
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps({
    "output_root": str(output_root),
    "manifest": str(manifest_path),
    "summary": summary
  }, indent=2))


if __name__ == "__main__":
  export_runtime_failure_set()
