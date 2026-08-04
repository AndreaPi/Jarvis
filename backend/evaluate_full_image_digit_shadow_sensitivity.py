"""Sweep bounded runtime confidence/NMS settings on one full-image CV fold."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
  from backend.detector import RoiDetector
  from backend.evaluate_full_image_digit_detector import (
    build_sequence_record,
    summarize_sequence_records,
    validation_annotation_groups,
  )
  from backend.full_image_digit_shadow import FullImageDigitShadow, crop_image
  from backend.train_full_image_digit_detector import (
    CV_FOLD_COUNT,
    file_sha256,
    group_annotations,
    read_csv_rows,
    read_fold_assignments,
    resolve_device,
    resolve_path,
  )
except ModuleNotFoundError:
  from detector import RoiDetector
  from evaluate_full_image_digit_detector import (
    build_sequence_record,
    summarize_sequence_records,
    validation_annotation_groups,
  )
  from full_image_digit_shadow import FullImageDigitShadow, crop_image
  from train_full_image_digit_detector import (
    CV_FOLD_COUNT,
    file_sha256,
    group_annotations,
    read_csv_rows,
    read_fold_assignments,
    resolve_device,
    resolve_path,
  )


DEFAULT_CONFIDENCES = (0.10, 0.15, 0.20, 0.25, 0.30)
DEFAULT_IOUS = (0.50, 0.70, 0.90)
DEFAULT_TARGET = "meter_20260423.JPEG"
BASELINE_CONFIDENCE = 0.25
BASELINE_IOU = 0.70


def parse_float_grid(raw: str, *, minimum: float, maximum: float) -> list[float]:
  values = sorted({float(token.strip()) for token in raw.split(",") if token.strip()})
  if not values:
    raise ValueError("Sensitivity grid cannot be empty.")
  if any(value < minimum or value > maximum for value in values):
    raise ValueError(f"Sensitivity values must be in [{minimum}, {maximum}].")
  return values


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Run exact single-image ROI-cascade inference across a bounded confidence/"
      "NMS grid on one persistent CV fold."
    )
  )
  parser.add_argument(
    "--checkpoint",
    default="runs/full-image-digit-detector-balanced48-crops-fold4/weights/best.pt",
  )
  parser.add_argument("--fold", type=int, default=4, choices=range(CV_FOLD_COUNT))
  parser.add_argument(
    "--confidences",
    default=",".join(str(value) for value in DEFAULT_CONFIDENCES),
  )
  parser.add_argument(
    "--ious",
    default=",".join(str(value) for value in DEFAULT_IOUS),
  )
  parser.add_argument("--target", default=DEFAULT_TARGET)
  parser.add_argument(
    "--annotations",
    default="data/full_image_digit_dataset/manifests/annotations.csv",
  )
  parser.add_argument(
    "--folds",
    default="data/full_image_digit_dataset/manifests/cv_folds.csv",
  )
  parser.add_argument("--source-images", default="data/roi_dataset/images")
  parser.add_argument("--roi-model", default="models/roi-rotaug-e30-640.pt")
  parser.add_argument("--device", default="cpu")
  parser.add_argument("--roi-device", default="cpu")
  parser.add_argument("--imgsz", type=int, default=1280)
  parser.add_argument("--max-detections", type=int, default=300)
  parser.add_argument("--roi-confidence", type=float, default=0.05)
  parser.add_argument("--roi-iou", type=float, default=0.5)
  parser.add_argument("--roi-imgsz", type=int, default=960)
  parser.add_argument("--roi-expand-x", type=float, default=0.26)
  parser.add_argument("--roi-expand-y", type=float, default=0.16)
  parser.add_argument(
    "--output-root",
    default="../output/full-image-digit-shadow-sensitivity",
  )
  return parser.parse_args()


def rank_setting(
  payload: dict[str, object],
) -> tuple[float, float, float, float, float]:
  metrics = payload["sequence_metrics"]
  assert isinstance(metrics, dict)
  readable_mae = metrics["readable_mae"]
  return (
    -float(metrics["no_read_count"]),
    float(metrics["exact_match_count"]),
    -float(readable_mae) if readable_mae is not None else float("-inf"),
    -abs(float(payload["iou"]) - BASELINE_IOU),
    float(payload["confidence"]),
  )


def render_target_grid(
  output_path: Path,
  target_image: np.ndarray,
  target_results: list[dict[str, object]],
) -> None:
  font = ImageFont.load_default()
  tile_width = 360
  tile_height = 520
  header_height = 54
  columns = 3
  rows = (len(target_results) + columns - 1) // columns
  canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")

  for index, result in enumerate(target_results):
    crop_bbox = result.get("crop_bbox_norm")
    if not isinstance(crop_bbox, dict):
      continue
    crop = crop_image(target_image, crop_bbox)
    crop_image_pil = Image.fromarray(crop)
    scale = min(
      tile_width / max(1, crop_image_pil.width),
      (tile_height - header_height) / max(1, crop_image_pil.height),
    )
    resized = crop_image_pil.resize((
      max(1, int(crop_image_pil.width * scale)),
      max(1, int(crop_image_pil.height * scale)),
    ))
    draw = ImageDraw.Draw(resized)
    for detection in result["detections"]:
      left = (float(detection["x_center"]) - float(detection["width"]) / 2) * resized.width
      top = (float(detection["y_center"]) - float(detection["height"]) / 2) * resized.height
      right = (float(detection["x_center"]) + float(detection["width"]) / 2) * resized.width
      bottom = (float(detection["y_center"]) + float(detection["height"]) / 2) * resized.height
      draw.rectangle((left, top, right, bottom), outline="#ff2f92", width=3)
      draw.text(
        (left + 2, max(0, top - 12)),
        f"{detection['class_id']} {float(detection['confidence']):.2f}",
        fill="#ff2f92",
        font=font,
      )
    x = (index % columns) * tile_width
    y = (index // columns) * tile_height
    title = (
      f"conf {float(result['confidence']):.2f} / IoU {float(result['iou']):.2f} | "
      f"{result['predicted'] or 'NO-READ'} | {result['detection_count']} boxes"
    )
    canvas_draw = ImageDraw.Draw(canvas)
    canvas_draw.text((x + 6, y + 6), title, fill="black", font=font)
    canvas.paste(resized, (x + (tile_width - resized.width) // 2, y + header_height))

  output_path.parent.mkdir(parents=True, exist_ok=True)
  canvas.save(output_path, quality=92)


def main() -> None:
  args = parse_args()
  confidences = parse_float_grid(args.confidences, minimum=0.0, maximum=1.0)
  ious = parse_float_grid(args.ious, minimum=0.01, maximum=1.0)
  if args.max_detections < 4:
    raise ValueError("--max-detections must be at least 4.")

  base_dir = Path(__file__).resolve().parent
  checkpoint_path = resolve_path(base_dir, args.checkpoint)
  annotations_path = resolve_path(base_dir, args.annotations)
  folds_path = resolve_path(base_dir, args.folds)
  source_images_root = resolve_path(base_dir, args.source_images)
  roi_model_path = resolve_path(base_dir, args.roi_model)
  output_root = resolve_path(base_dir, args.output_root)
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

  grouped = group_annotations(read_csv_rows(annotations_path))
  assignments = read_fold_assignments(folds_path)
  validation_groups = validation_annotation_groups(grouped, assignments, args.fold)
  if args.target not in validation_groups:
    raise ValueError(f"Target is not assigned to fold {args.fold}: {args.target}")

  images: dict[str, np.ndarray] = {}
  for filename, rows in validation_groups.items():
    image_path = source_images_root / rows[0]["split"] / filename
    with Image.open(image_path) as source:
      images[filename] = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))

  roi_detector = RoiDetector(
    roi_model_path,
    device=resolve_device(args.roi_device),
  )
  shadow = FullImageDigitShadow(
    checkpoint_path,
    device=resolve_device(args.device),
  )
  settings: list[dict[str, object]] = []
  target_results: list[dict[str, object]] = []
  for confidence in confidences:
    for iou in ious:
      records: list[dict[str, object]] = []
      for filename, rows in validation_groups.items():
        payload = shadow.predict(
          images[filename],
          roi_detector,
          roi_confidence=args.roi_confidence,
          roi_iou=args.roi_iou,
          roi_imgsz=args.roi_imgsz,
          roi_expand_x=args.roi_expand_x,
          roi_expand_y=args.roi_expand_y,
          confidence=confidence,
          iou=iou,
          imgsz=args.imgsz,
          max_detections=args.max_detections,
        )
        record = build_sequence_record(
          filename,
          rows[0]["reading"],
          int(rows[0]["direction_rotation"]),
          list(payload["detections"]),
        )
        records.append(record)
        if filename == args.target:
          target_results.append({
            "confidence": confidence,
            "iou": iou,
            "predicted": record["predicted"],
            "detection_count": record["detection_count"],
            "detections": record["detections"],
            "crop_bbox_norm": (
              payload.get("roi", {}).get("expanded_bbox_norm")
              if isinstance(payload.get("roi"), dict)
              else None
            ),
          })
      settings.append({
        "confidence": confidence,
        "iou": iou,
        "sequence_metrics": summarize_sequence_records(records),
        "predictions": records,
      })

  best = max(settings, key=rank_setting)
  timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  output_dir = output_root / timestamp
  output_dir.mkdir(parents=True, exist_ok=False)
  target_grid_path = output_dir / f"{Path(args.target).stem}-sensitivity.jpg"
  render_target_grid(target_grid_path, images[args.target], target_results)
  payload = {
    "version": 1,
    "generated_at": datetime.now().astimezone().isoformat(),
    "evaluation_scope": "single_image_runtime_roi_cascade_selected_cv_fold",
    "fold": args.fold,
    "target": args.target,
    "checkpoint": str(checkpoint_path),
    "checkpoint_sha256": file_sha256(checkpoint_path),
    "roi_model": str(roi_model_path),
    "roi_model_sha256": file_sha256(roi_model_path),
    "grid": {"confidences": confidences, "ious": ious},
    "settings": settings,
    "best_setting_by_no_read_exact_mae_minimal_change": {
      "confidence": best["confidence"],
      "iou": best["iou"],
      "sequence_metrics": best["sequence_metrics"],
    },
    "target_grid": target_grid_path.name,
    "decision_boundary": (
      "This fold is now a tuning surface. A selected setting must be verified on "
      "different unseen images before promotion."
    ),
  }
  (output_dir / "summary.json").write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
  )
  lines = [
    "# Full-Image Digit Shadow Sensitivity",
    "",
    f"Fold {args.fold}; exact single-image runtime path; target `{args.target}`.",
    "",
    "| Confidence | NMS IoU | Exact | No-read | Readable MAE | Target | Boxes |",
    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for setting in settings:
    metrics = setting["sequence_metrics"]
    target = next(
      record for record in setting["predictions"]
      if record["filename"] == args.target
    )
    lines.append(
      f"| {float(setting['confidence']):.2f} | {float(setting['iou']):.2f} | "
      f"{metrics['exact_match_count']}/{metrics['image_count']} | "
      f"{metrics['no_read_count']} | {metrics['readable_mae']} | "
      f"{target['predicted'] or 'NO-READ'} | {target['detection_count']} |"
    )
  lines.extend([
    "",
    "The ranking minimizes no-read first, maximizes exact match, minimizes readable MAE, then prefers the baseline NMS IoU and highest safe confidence.",
    "This fold is now a tuning surface; verify any selected setting on different unseen images before promotion.",
    "",
    f"![Target sensitivity]({target_grid_path.name})",
    "",
  ])
  (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
  print(json.dumps({
    "output": str(output_dir),
    "best_setting": payload["best_setting_by_no_read_exact_mae_minimal_change"],
  }, indent=2))


if __name__ == "__main__":
  main()
