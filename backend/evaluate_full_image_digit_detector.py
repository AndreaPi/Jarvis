from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from importlib import metadata
from pathlib import Path

try:
  from backend.build_full_image_digit_dataset import (
    filter_excluded_annotations,
    read_source_exclusions,
  )
except ModuleNotFoundError:
  from build_full_image_digit_dataset import (
    filter_excluded_annotations,
    read_source_exclusions,
  )

try:
  from backend.train_full_image_digit_detector import (
    CV_FOLD_COUNT,
    file_sha256,
    group_annotations,
    materialize_fold_dataset,
    read_csv_rows,
    read_fold_assignments,
    resolve_device,
    resolve_path,
  )
except ModuleNotFoundError:
  from train_full_image_digit_detector import (
    CV_FOLD_COUNT,
    file_sha256,
    group_annotations,
    materialize_fold_dataset,
    read_csv_rows,
    read_fold_assignments,
    resolve_device,
    resolve_path,
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Evaluate one full-image digit-detector checkpoint on its selected "
      "cross-validation fold, including complete four-digit readings."
    )
  )
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument(
    "--annotations",
    default="data/full_image_digit_dataset/manifests/annotations.csv",
  )
  parser.add_argument(
    "--folds",
    default="data/full_image_digit_dataset/manifests/cv_folds.csv",
  )
  parser.add_argument(
    "--source-exclusions",
    default="data/full_image_digit_dataset/manifests/source_exclusions.csv",
    help="Legacy-stress sources excluded from the active evaluation scope.",
  )
  parser.add_argument(
    "--source-images",
    default="data/roi_dataset/images",
  )
  parser.add_argument(
    "--labels",
    default="data/full_image_digit_dataset/labels",
  )
  parser.add_argument("--fold", type=int, default=0, choices=range(CV_FOLD_COUNT))
  parser.add_argument("--imgsz", type=int, default=1280)
  parser.add_argument("--batch", type=int, default=4)
  parser.add_argument("--workers", type=int, default=0)
  parser.add_argument(
    "--device",
    default="cpu",
    help="Evaluation device. CPU is the safe default for the small CV fold.",
  )
  parser.add_argument(
    "--confidence",
    type=float,
    default=0.25,
    help="Fixed confidence threshold for complete-reading predictions.",
  )
  parser.add_argument(
    "--iou",
    type=float,
    default=0.7,
    help="Class-agnostic NMS IoU threshold for complete-reading predictions.",
  )
  parser.add_argument("--max-detections", type=int, default=300)
  parser.add_argument(
    "--output",
    default="",
    help=(
      "JSON output path. Defaults to sequence_evaluation_foldN.json beside "
      "the checkpoint run's weights directory."
    ),
  )
  return parser.parse_args()


def validation_annotation_groups(
  grouped_annotations: dict[str, list[dict[str, str]]],
  fold_assignments: dict[str, int],
  selected_fold: int,
) -> dict[str, list[dict[str, str]]]:
  if selected_fold not in range(CV_FOLD_COUNT):
    raise ValueError(f"Selected fold must be in 0..{CV_FOLD_COUNT - 1}")

  selected: dict[str, list[dict[str, str]]] = {}
  for filename, fold in sorted(fold_assignments.items()):
    if fold != selected_fold:
      continue
    rows = grouped_annotations.get(filename)
    if not rows:
      raise ValueError(f"Fold assignment has no annotations: {filename}")
    ordered_rows = sorted(rows, key=lambda row: int(row["position"]))
    if ordered_rows[0]["split"] != "train":
      raise ValueError(f"CV validation source must come from train: {filename}")
    selected[filename] = ordered_rows

  if not selected:
    raise ValueError(f"Fold {selected_fold} has no validation images.")
  return selected


def sort_detections_in_reading_order(
  detections: list[dict[str, float | int]],
  rotation: int,
) -> list[dict[str, float | int]]:
  if rotation == 0:
    return sorted(detections, key=lambda item: float(item["x_center"]))
  if rotation == 180:
    return sorted(
      detections,
      key=lambda item: float(item["x_center"]),
      reverse=True,
    )
  if rotation == 90:
    return sorted(
      detections,
      key=lambda item: float(item["y_center"]),
      reverse=True,
    )
  if rotation == 270:
    return sorted(detections, key=lambda item: float(item["y_center"]))
  raise ValueError(f"Unsupported reading-direction rotation: {rotation}")


def build_sequence_record(
  filename: str,
  truth: str,
  rotation: int,
  detections: list[dict[str, float | int]],
) -> dict[str, object]:
  if len(truth) != 4 or not truth.isdigit():
    raise ValueError(f"Expected a four-digit truth for {filename}: {truth!r}")

  ordered = sort_detections_in_reading_order(detections, rotation)
  predicted = (
    "".join(str(int(detection["class_id"])) for detection in ordered)
    if len(ordered) == 4
    else None
  )
  exact = predicted == truth
  digit_correct = (
    sum(expected == actual for expected, actual in zip(truth, predicted, strict=True))
    if predicted is not None
    else None
  )
  absolute_error = (
    abs(int(predicted) - int(truth))
    if predicted is not None
    else None
  )
  return {
    "filename": filename,
    "truth": truth,
    "predicted": predicted,
    "status": "read" if predicted is not None else "no-read",
    "exact_match": exact,
    "digit_correct": digit_correct,
    "absolute_error": absolute_error,
    "direction_rotation": rotation,
    "detection_count": len(ordered),
    "detections": ordered,
  }


def summarize_sequence_records(
  records: list[dict[str, object]],
) -> dict[str, int | float | None]:
  image_count = len(records)
  if image_count == 0:
    raise ValueError("Cannot summarize an empty sequence evaluation.")

  readable = [record for record in records if record["predicted"] is not None]
  exact_match_count = sum(bool(record["exact_match"]) for record in records)
  no_read_count = image_count - len(readable)
  readable_exact_count = sum(bool(record["exact_match"]) for record in readable)
  absolute_errors = [int(record["absolute_error"]) for record in readable]
  digit_correct = sum(int(record["digit_correct"]) for record in readable)
  readable_digit_count = len(readable) * 4

  return {
    "image_count": image_count,
    "readable_count": len(readable),
    "no_read_count": no_read_count,
    "no_read_rate": no_read_count / image_count,
    "exact_match_count": exact_match_count,
    "exact_match_rate": exact_match_count / image_count,
    "readable_exact_match_rate": (
      readable_exact_count / len(readable)
      if readable
      else None
    ),
    "readable_digit_accuracy": (
      digit_correct / readable_digit_count
      if readable_digit_count
      else None
    ),
    "readable_mae": (
      sum(absolute_errors) / len(absolute_errors)
      if absolute_errors
      else None
    ),
  }


def detections_from_result(result: object) -> list[dict[str, float | int]]:
  boxes = getattr(result, "boxes", None)
  if boxes is None:
    return []

  detections: list[dict[str, float | int]] = []
  for class_id, confidence, xywh in zip(
    boxes.cls.tolist(),
    boxes.conf.tolist(),
    boxes.xywhn.tolist(),
    strict=True,
  ):
    detections.append({
      "class_id": int(class_id),
      "confidence": round(float(confidence), 8),
      "x_center": round(float(xywh[0]), 8),
      "y_center": round(float(xywh[1]), 8),
      "width": round(float(xywh[2]), 8),
      "height": round(float(xywh[3]), 8),
    })
  return detections


def default_output_path(checkpoint_path: Path, fold: int) -> Path:
  parent = checkpoint_path.parent
  run_dir = parent.parent if parent.name == "weights" else parent
  return run_dir / f"sequence_evaluation_fold{fold}.json"


def main() -> None:
  args = parse_args()
  if not 0 <= args.confidence <= 1:
    raise ValueError("--confidence must be in [0, 1].")
  if not 0 < args.iou <= 1:
    raise ValueError("--iou must be in (0, 1].")
  if args.max_detections < 4:
    raise ValueError("--max-detections must be at least 4.")

  base_dir = Path(__file__).resolve().parent
  checkpoint_path = resolve_path(base_dir, args.checkpoint)
  annotations_path = resolve_path(base_dir, args.annotations)
  folds_path = resolve_path(base_dir, args.folds)
  source_exclusions_path = resolve_path(base_dir, args.source_exclusions)
  source_images_root = resolve_path(base_dir, args.source_images)
  labels_root = resolve_path(base_dir, args.labels)
  output_path = (
    resolve_path(base_dir, args.output)
    if args.output
    else default_output_path(checkpoint_path, args.fold)
  )
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

  source_exclusions = read_source_exclusions(source_exclusions_path)
  annotation_rows = filter_excluded_annotations(
    read_csv_rows(annotations_path),
    set(source_exclusions),
  )
  fold_assignments = read_fold_assignments(folds_path)
  grouped_annotations = group_annotations(annotation_rows)
  validation_groups = validation_annotation_groups(
    grouped_annotations,
    fold_assignments,
    args.fold,
  )

  temporary_parent = Path(tempfile.mkdtemp(prefix="jarvis_full_digit_evaluation_"))
  try:
    dataset_yaml, dataset_summary = materialize_fold_dataset(
      temporary_parent / "dataset",
      annotation_rows,
      fold_assignments,
      source_images_root,
      labels_root,
      args.fold,
    )
    expected_validation_count = len(validation_groups)
    if dataset_summary["split_images"]["val"] != expected_validation_count:
      raise RuntimeError("Materialized validation count differs from fold manifest.")

    matplotlib_cache = Path(tempfile.gettempdir()) / "jarvis-matplotlib-cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    try:
      from ultralytics import YOLO
    except ImportError as error:
      raise RuntimeError("ultralytics is required. Install backend/requirements.txt.") from error

    model = YOLO(str(checkpoint_path))
    device = resolve_device(args.device)
    validation_metrics = model.val(
      data=str(dataset_yaml),
      split="val",
      imgsz=args.imgsz,
      batch=args.batch,
      workers=args.workers,
      device=device,
      iou=args.iou,
      max_det=args.max_detections,
      plots=False,
      verbose=False,
      project=str(temporary_parent / "validation"),
      name="metrics",
    )

    filenames = sorted(validation_groups)
    image_paths = [
      source_images_root
      / validation_groups[filename][0]["split"]
      / filename
      for filename in filenames
    ]
    prediction_results = model.predict(
      source=[str(path) for path in image_paths],
      imgsz=args.imgsz,
      device=device,
      conf=args.confidence,
      iou=args.iou,
      max_det=args.max_detections,
      agnostic_nms=True,
      save=False,
      verbose=False,
    )
    if len(prediction_results) != len(filenames):
      raise RuntimeError("Prediction count differs from validation image count.")

    records = []
    for filename, result in zip(filenames, prediction_results, strict=True):
      rows = validation_groups[filename]
      reading_values = {row["reading"] for row in rows}
      rotation_values = {int(row["direction_rotation"]) for row in rows}
      if len(reading_values) != 1 or len(rotation_values) != 1:
        raise ValueError(f"Inconsistent reading metadata for {filename}")
      records.append(build_sequence_record(
        filename,
        next(iter(reading_values)),
        next(iter(rotation_values)),
        detections_from_result(result),
      ))

    box_metrics = validation_metrics.box
    map50 = float(box_metrics.map50)
    map50_95 = float(box_metrics.map)
    payload = {
      "evaluation_scope": "selected_cv_validation_fold_active_full_images_only",
      "fold": args.fold,
      "checkpoint": str(checkpoint_path),
      "checkpoint_sha256": file_sha256(checkpoint_path),
      "annotations_sha256": file_sha256(annotations_path),
      "cv_folds_sha256": file_sha256(folds_path),
      "source_exclusions_sha256": (
        file_sha256(source_exclusions_path)
        if source_exclusions_path.exists()
        else None
      ),
      "source_exclusions": [
        source_exclusions[filename]
        for filename in sorted(source_exclusions)
      ],
      "ultralytics_version": metadata.version("ultralytics"),
      "settings": {
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "confidence": args.confidence,
        "iou": args.iou,
        "max_detections": args.max_detections,
        "agnostic_nms": True,
        "required_detection_count": 4,
      },
      "detection_metrics": {
        "precision": float(box_metrics.mp),
        "recall": float(box_metrics.mr),
        "map50": map50,
        "map50_95": map50_95,
        "fitness": 0.1 * map50 + 0.9 * map50_95,
      },
      "sequence_metrics": summarize_sequence_records(records),
      "predictions": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
      json.dumps(payload, indent=2) + "\n",
      encoding="utf-8",
    )
    print(json.dumps({
      "output": str(output_path),
      "detection_metrics": payload["detection_metrics"],
      "sequence_metrics": payload["sequence_metrics"],
    }, indent=2))
  finally:
    shutil.rmtree(temporary_parent, ignore_errors=True)


if __name__ == "__main__":
  main()
