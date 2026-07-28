from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from importlib import metadata
from pathlib import Path

import yaml
from PIL import Image

try:
  from backend.build_full_image_digit_dataset import (
    CLASS_NAMES,
    filter_excluded_annotations,
    read_source_exclusions,
    validate_annotations,
  )
except ModuleNotFoundError:
  from build_full_image_digit_dataset import (
    CLASS_NAMES,
    filter_excluded_annotations,
    read_source_exclusions,
    validate_annotations,
  )


CV_FOLD_COUNT = 5
DIGIT_CROP_AXIS_CONTEXT_SCALES = (0.55, 0.75, 0.95)
DIGIT_CROP_PERPENDICULAR_CONTEXT_RATIOS = (0.75, 1.25, 1.75, 2.25)
TRAIN_AUGMENT_KWARGS = {
  "degrees": 180.0,
  "translate": 0.15,
  "scale": 0.5,
  "shear": 10.0,
  "perspective": 0.0005,
  "flipud": 0.5,
  "fliplr": 0.5,
  "mosaic": 1.0,
  "mixup": 0.1,
  "close_mosaic": 10,
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Train a ten-class YOLO detector on reviewed full-image digit-wheel boxes "
      "with an image-level cross-validation fold."
    )
  )
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
    help="Legacy-stress sources excluded from active training and CV.",
  )
  parser.add_argument(
    "--source-images",
    default="data/roi_dataset/images",
    help="Existing DVC-managed full images grouped by source split.",
  )
  parser.add_argument(
    "--labels",
    default="data/full_image_digit_dataset/labels",
    help="Reviewed YOLO labels grouped by source split.",
  )
  parser.add_argument("--fold", type=int, default=0, choices=range(CV_FOLD_COUNT))
  parser.add_argument(
    "--base-model",
    default="yolov8n.pt",
    help="Ultralytics detection checkpoint or model YAML.",
  )
  parser.add_argument(
    "--pretrained-model",
    default="",
    help=(
      "Optional checkpoint to load into a model YAML, for example "
      "--base-model yolov8n-p2.yaml --pretrained-model yolov8n.pt."
    ),
  )
  parser.add_argument("--epochs", type=int, default=120)
  parser.add_argument("--imgsz", type=int, default=1280)
  parser.add_argument("--batch", type=int, default=4)
  parser.add_argument("--patience", type=int, default=25)
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
    "--train-register-crops",
    action="store_true",
    help=(
      "Add one register-context crop for each training image. Validation and "
      "test images always remain full-image only."
    ),
  )
  parser.add_argument(
    "--register-crop-context",
    type=float,
    default=0.75,
    help=(
      "Padding around the union of the four digit boxes, as a fraction of the "
      "register's longest side."
    ),
  )
  parser.add_argument(
    "--train-balanced-digit-target",
    type=int,
    default=0,
    help=(
      "Minimum total training-box exposure per class after full images and "
      "optional register crops. Rare classes are raised to this floor with "
      "digit-centred crops from training-fold images only; 0 disables it."
    ),
  )
  parser.add_argument(
    "--device",
    default="auto",
    help="Training device: auto, cpu, mps, 0, or cuda:0.",
  )
  parser.add_argument("--project", default="runs")
  parser.add_argument(
    "--name",
    default="",
    help="Run name. Defaults to full-image-digit-detector-foldN.",
  )
  parser.add_argument(
    "--copy-to",
    default="",
    help=(
      "Optional checkpoint destination. Leave empty during CV; use "
      "models/full_image_digit_detector.pt only after promotion approval."
    ),
  )
  parser.add_argument(
    "--resume-from",
    default="",
    help=(
      "Resume an interrupted run from its unstripped weights/last.pt. The "
      "selected fold and crop recipe are rematerialized before Ultralytics "
      "restores the epoch, optimizer, scheduler, and early-stopping state."
    ),
  )
  parser.add_argument(
    "--validate-only",
    action="store_true",
    help="Validate and materialize the selected fold without training.",
  )
  return parser.parse_args()


def resolve_path(base_dir: Path, value: str) -> Path:
  path = Path(value)
  if path.is_absolute():
    return path
  return (base_dir / path).resolve()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
  with path.open("r", encoding="utf-8") as handle:
    return [
      {key: (value or "") for key, value in row.items()}
      for row in csv.DictReader(handle)
    ]


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def read_fold_assignments(path: Path) -> dict[str, int]:
  assignments: dict[str, int] = {}
  for row in read_csv_rows(path):
    filename = row.get("filename", "")
    if not filename or filename in assignments:
      raise ValueError(f"Invalid duplicate CV fold row: {filename!r}")
    try:
      fold = int(row.get("fold", ""))
    except ValueError as error:
      raise ValueError(f"Invalid fold for {filename}: {row.get('fold')!r}") from error
    if fold not in range(CV_FOLD_COUNT):
      raise ValueError(f"Fold for {filename} must be in 0..{CV_FOLD_COUNT - 1}")
    assignments[filename] = fold
  return assignments


def parse_label_rows(
  path: Path,
  expected_count: int = 4,
) -> list[tuple[int, float, float, float, float]]:
  labels: list[tuple[int, float, float, float, float]] = []
  for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
    parts = line.strip().split()
    if not parts:
      continue
    if len(parts) != 5:
      raise ValueError(f"{path}:{line_number}: expected five YOLO columns")
    try:
      class_id = int(parts[0])
      x_center, y_center, width, height = (float(value) for value in parts[1:])
    except ValueError as error:
      raise ValueError(f"{path}:{line_number}: invalid YOLO row") from error
    if class_id not in range(10):
      raise ValueError(f"{path}:{line_number}: class must be in 0..9")
    if width <= 0 or height <= 0:
      raise ValueError(f"{path}:{line_number}: box size must be positive")
    if x_center - width * 0.5 < 0 or x_center + width * 0.5 > 1:
      raise ValueError(f"{path}:{line_number}: horizontal bounds exceed image")
    if y_center - height * 0.5 < 0 or y_center + height * 0.5 > 1:
      raise ValueError(f"{path}:{line_number}: vertical bounds exceed image")
    labels.append((class_id, x_center, y_center, width, height))
  if len(labels) != expected_count:
    raise ValueError(
      f"{path}: expected exactly {expected_count} digit boxes, got {len(labels)}"
    )
  return labels


def parse_label_file(path: Path) -> list[int]:
  return [class_id for class_id, *_ in parse_label_rows(path)]


def write_register_crop(
  source_image_path: Path,
  target_image_path: Path,
  target_label_path: Path,
  labels: list[tuple[int, float, float, float, float]],
  context_ratio: float,
) -> None:
  if context_ratio < 0:
    raise ValueError("Register crop context must be non-negative.")

  with Image.open(source_image_path) as source_image:
    source_image.load()
    image_width, image_height = source_image.size
    left = min((x_center - width * 0.5) * image_width for _, x_center, _, width, _ in labels)
    top = min((y_center - height * 0.5) * image_height for _, _, y_center, _, height in labels)
    right = max((x_center + width * 0.5) * image_width for _, x_center, _, width, _ in labels)
    bottom = max((y_center + height * 0.5) * image_height for _, _, y_center, _, height in labels)
    longest_side = max(right - left, bottom - top)
    padding = longest_side * context_ratio
    crop_left = max(0, math.floor(left - padding))
    crop_top = max(0, math.floor(top - padding))
    crop_right = min(image_width, math.ceil(right + padding))
    crop_bottom = min(image_height, math.ceil(bottom + padding))
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    if crop_width <= 0 or crop_height <= 0:
      raise ValueError(f"Register crop is empty for {source_image_path.name}")

    cropped_image = source_image.crop(
      (crop_left, crop_top, crop_right, crop_bottom)
    ).convert("RGB")
    cropped_image.save(target_image_path, "JPEG", quality=95)

  transformed_lines = []
  for class_id, x_center, y_center, width, height in labels:
    transformed_x_center = (x_center * image_width - crop_left) / crop_width
    transformed_y_center = (y_center * image_height - crop_top) / crop_height
    transformed_width = width * image_width / crop_width
    transformed_height = height * image_height / crop_height
    transformed_lines.append(
      f"{class_id} {transformed_x_center:.8f} {transformed_y_center:.8f} "
      f"{transformed_width:.8f} {transformed_height:.8f}"
    )
  target_label_path.write_text(
    "\n".join(transformed_lines) + "\n",
    encoding="utf-8",
  )
  parse_label_rows(target_label_path)


def annotation_box_pixels(
  row: dict[str, str],
  image_width: int,
  image_height: int,
) -> tuple[float, float, float, float]:
  x_center = float(row["x_center"]) * image_width
  y_center = float(row["y_center"]) * image_height
  width = float(row["width"]) * image_width
  height = float(row["height"]) * image_height
  return (
    x_center - width * 0.5,
    y_center - height * 0.5,
    x_center + width * 0.5,
    y_center + height * 0.5,
  )


def digit_centered_crop_bounds(
  rows: list[dict[str, str]],
  target_row: dict[str, str],
  image_width: int,
  image_height: int,
  variant_index: int,
) -> tuple[int, int, int, int]:
  rotation_values = {int(row["direction_rotation"]) for row in rows}
  if len(rotation_values) != 1:
    raise ValueError(f"Conflicting direction rotations for {target_row['filename']}")
  rotation = next(iter(rotation_values))
  if rotation not in {0, 90, 180, 270}:
    raise ValueError(f"Unsupported direction rotation: {rotation}")

  horizontal = rotation in {0, 180}
  box_by_position = {
    int(row["position"]): annotation_box_pixels(row, image_width, image_height)
    for row in rows
  }
  target_position = int(target_row["position"])
  target_box = box_by_position[target_position]
  physical_positions = sorted(
    box_by_position,
    key=lambda position: (
      (box_by_position[position][0] + box_by_position[position][2]) * 0.5
      if horizontal
      else (box_by_position[position][1] + box_by_position[position][3]) * 0.5
    ),
  )
  physical_index = physical_positions.index(target_position)

  if horizontal:
    target_axis_min, target_axis_max = target_box[0], target_box[2]
    target_perpendicular_min, target_perpendicular_max = target_box[1], target_box[3]
    axis_limit = image_width
    perpendicular_limit = image_height
  else:
    target_axis_min, target_axis_max = target_box[1], target_box[3]
    target_perpendicular_min, target_perpendicular_max = target_box[0], target_box[2]
    axis_limit = image_height
    perpendicular_limit = image_width

  target_axis_size = target_axis_max - target_axis_min
  target_perpendicular_size = target_perpendicular_max - target_perpendicular_min
  if target_axis_size <= 0 or target_perpendicular_size <= 0:
    raise ValueError(f"Degenerate target digit box for {target_row['filename']}")

  if physical_index > 0:
    previous_box = box_by_position[physical_positions[physical_index - 1]]
    previous_axis_max = previous_box[2] if horizontal else previous_box[3]
    cell_axis_min = (
      (previous_axis_max + target_axis_min) * 0.5
      if previous_axis_max < target_axis_min
      else target_axis_min
    )
  else:
    cell_axis_min = target_axis_min - target_axis_size * 0.5

  if physical_index + 1 < len(physical_positions):
    next_box = box_by_position[physical_positions[physical_index + 1]]
    next_axis_min = next_box[0] if horizontal else next_box[1]
    cell_axis_max = (
      (target_axis_max + next_axis_min) * 0.5
      if target_axis_max < next_axis_min
      else target_axis_max
    )
  else:
    cell_axis_max = target_axis_max + target_axis_size * 0.5

  axis_scale = DIGIT_CROP_AXIS_CONTEXT_SCALES[
    variant_index % len(DIGIT_CROP_AXIS_CONTEXT_SCALES)
  ]
  perpendicular_context = DIGIT_CROP_PERPENDICULAR_CONTEXT_RATIOS[
    (variant_index // len(DIGIT_CROP_AXIS_CONTEXT_SCALES))
    % len(DIGIT_CROP_PERPENDICULAR_CONTEXT_RATIOS)
  ]
  crop_axis_min = target_axis_min - (target_axis_min - cell_axis_min) * axis_scale
  crop_axis_max = target_axis_max + (cell_axis_max - target_axis_max) * axis_scale
  crop_perpendicular_min = (
    target_perpendicular_min - target_perpendicular_size * perpendicular_context
  )
  crop_perpendicular_max = (
    target_perpendicular_max + target_perpendicular_size * perpendicular_context
  )

  crop_axis_min = max(0, math.floor(crop_axis_min))
  crop_axis_max = min(axis_limit, math.ceil(crop_axis_max))
  crop_perpendicular_min = max(0, math.floor(crop_perpendicular_min))
  crop_perpendicular_max = min(
    perpendicular_limit,
    math.ceil(crop_perpendicular_max),
  )
  if horizontal:
    bounds = (
      crop_axis_min,
      crop_perpendicular_min,
      crop_axis_max,
      crop_perpendicular_max,
    )
  else:
    bounds = (
      crop_perpendicular_min,
      crop_axis_min,
      crop_perpendicular_max,
      crop_axis_max,
    )
  if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
    raise ValueError(f"Digit-centred crop is empty for {target_row['filename']}")
  return bounds


def write_digit_centered_crop(
  source_image_path: Path,
  target_image_path: Path,
  target_label_path: Path,
  rows: list[dict[str, str]],
  target_row: dict[str, str],
  variant_index: int,
) -> None:
  with Image.open(source_image_path) as source_image:
    source_image.load()
    image_width, image_height = source_image.size
    crop_left, crop_top, crop_right, crop_bottom = digit_centered_crop_bounds(
      rows,
      target_row,
      image_width,
      image_height,
      variant_index,
    )
    cropped_image = source_image.crop(
      (crop_left, crop_top, crop_right, crop_bottom)
    ).convert("RGB")
    cropped_image.save(target_image_path, "JPEG", quality=95)

  crop_width = crop_right - crop_left
  crop_height = crop_bottom - crop_top
  target_left, target_top, target_right, target_bottom = annotation_box_pixels(
    target_row,
    image_width,
    image_height,
  )
  x_center = ((target_left + target_right) * 0.5 - crop_left) / crop_width
  y_center = ((target_top + target_bottom) * 0.5 - crop_top) / crop_height
  width = (target_right - target_left) / crop_width
  height = (target_bottom - target_top) / crop_height
  target_label_path.write_text(
    f"{int(target_row['class_id'])} {x_center:.8f} {y_center:.8f} "
    f"{width:.8f} {height:.8f}\n",
    encoding="utf-8",
  )
  parse_label_rows(target_label_path, expected_count=1)


def group_annotations(
  annotation_rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
  grouped: dict[str, list[dict[str, str]]] = {}
  for row in annotation_rows:
    grouped.setdefault(row["filename"], []).append(row)
  validate_annotations(annotation_rows, set(grouped))
  pending = sorted({
    row["filename"]
    for row in annotation_rows
    if row.get("review_status") != "reviewed"
  })
  if pending:
    raise ValueError(
      "Training requires every annotation to be reviewed. Pending images: "
      + ", ".join(pending)
    )
  return grouped


def materialize_fold_dataset(
  destination: Path,
  annotation_rows: list[dict[str, str]],
  fold_assignments: dict[str, int],
  source_images_root: Path,
  labels_root: Path,
  selected_fold: int,
  train_register_crops: bool = False,
  register_crop_context: float = 0.75,
  train_balanced_digit_target: int = 0,
) -> tuple[Path, dict[str, object]]:
  if selected_fold not in range(CV_FOLD_COUNT):
    raise ValueError(f"Selected fold must be in 0..{CV_FOLD_COUNT - 1}")
  if register_crop_context < 0:
    raise ValueError("Register crop context must be non-negative.")
  if train_balanced_digit_target < 0:
    raise ValueError("Balanced digit target must be non-negative.")
  grouped = group_annotations(annotation_rows)
  train_filenames = {
    filename
    for filename, rows in grouped.items()
    if rows[0]["split"] == "train"
  }
  test_filenames = {
    filename
    for filename, rows in grouped.items()
    if rows[0]["split"] == "test"
  }
  unsupported = sorted({
    rows[0]["split"]
    for rows in grouped.values()
    if rows[0]["split"] not in {"train", "test"}
  })
  if unsupported:
    raise ValueError(f"Unexpected source splits: {unsupported}")
  if set(fold_assignments) != train_filenames:
    missing = sorted(train_filenames - set(fold_assignments))
    extra = sorted(set(fold_assignments) - train_filenames)
    raise ValueError(f"CV assignment mismatch: missing={missing}, extra={extra}")
  if not test_filenames:
    raise ValueError("A fixed test holdout is required.")

  destination.mkdir(parents=True, exist_ok=False)
  split_image_counts: Counter[str] = Counter()
  split_class_counts: dict[str, Counter[str]] = {
    "train": Counter(),
    "val": Counter(),
    "test": Counter(),
  }
  generated_register_crops = 0
  digit_crop_sources: dict[
    str,
    list[tuple[str, Path, list[dict[str, str]], dict[str, str]]],
  ] = {
    digit: []
    for digit in CLASS_NAMES
  }

  for filename in sorted(grouped):
    rows = sorted(grouped[filename], key=lambda row: int(row["position"]))
    source_split = rows[0]["split"]
    if source_split == "test":
      target_split = "test"
    elif fold_assignments[filename] == selected_fold:
      target_split = "val"
    else:
      target_split = "train"

    source_image = source_images_root / source_split / filename
    source_label = labels_root / source_split / f"{Path(filename).stem}.txt"
    if not source_image.exists():
      raise FileNotFoundError(f"Missing source image: {source_image}")
    if not source_label.exists():
      raise FileNotFoundError(f"Missing source label: {source_label}")
    label_rows = parse_label_rows(source_label)
    label_classes = [class_id for class_id, *_ in label_rows]
    expected_classes = sorted(int(row["class_id"]) for row in rows)
    if sorted(label_classes) != expected_classes:
      raise ValueError(f"Label classes differ from reviewed annotations for {filename}")

    target_image_dir = destination / "images" / target_split
    target_label_dir = destination / "labels" / target_split
    target_image_dir.mkdir(parents=True, exist_ok=True)
    target_label_dir.mkdir(parents=True, exist_ok=True)
    (target_image_dir / filename).symlink_to(source_image.resolve())
    shutil.copy2(source_label, target_label_dir / source_label.name)
    split_image_counts[target_split] += 1
    split_class_counts[target_split].update(str(class_id) for class_id in label_classes)

    if target_split == "train":
      for row in rows:
        digit_crop_sources[row["class_id"]].append(
          (filename, source_image, rows, row)
        )

    if train_register_crops and target_split == "train":
      crop_stem = f"{Path(filename).stem}__register_crop"
      crop_image_path = target_image_dir / f"{crop_stem}.JPEG"
      crop_label_path = target_label_dir / f"{crop_stem}.txt"
      write_register_crop(
        source_image,
        crop_image_path,
        crop_label_path,
        label_rows,
        register_crop_context,
      )
      generated_register_crops += 1
      split_image_counts["train"] += 1
      split_class_counts["train"].update(str(class_id) for class_id in label_classes)

  if split_image_counts["train"] == 0 or split_image_counts["val"] == 0:
    raise ValueError("Selected fold must produce non-empty train and validation splits.")

  generated_digit_class_counts: Counter[str] = Counter()
  generated_digit_source_names: dict[str, set[str]] = {
    digit: set()
    for digit in CLASS_NAMES
  }
  if train_balanced_digit_target:
    target_image_dir = destination / "images" / "train"
    target_label_dir = destination / "labels" / "train"
    for digit in CLASS_NAMES:
      required = max(
        0,
        train_balanced_digit_target - split_class_counts["train"][digit],
      )
      candidates = digit_crop_sources[digit]
      if required and not candidates:
        raise ValueError(
          f"Cannot balance digit {digit}: no source exists in this fold's training images."
        )
      candidate_usage: Counter[tuple[str, int]] = Counter()
      for generated_index in range(required):
        filename, source_image, rows, target_row = candidates[
          generated_index % len(candidates)
        ]
        source_key = (filename, int(target_row["position"]))
        variant_index = candidate_usage[source_key]
        candidate_usage[source_key] += 1
        crop_stem = (
          f"{Path(filename).stem}__digit_p{target_row['position']}"
          f"_d{digit}_r{variant_index:02d}"
        )
        write_digit_centered_crop(
          source_image,
          target_image_dir / f"{crop_stem}.JPEG",
          target_label_dir / f"{crop_stem}.txt",
          rows,
          target_row,
          variant_index,
        )
        generated_digit_class_counts[digit] += 1
        generated_digit_source_names[digit].add(filename)
        split_image_counts["train"] += 1
        split_class_counts["train"][digit] += 1

  payload = {
    "path": str(destination.resolve()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 10,
    "names": {
      class_id: class_name
      for class_id, class_name in enumerate(CLASS_NAMES)
    },
  }
  yaml_path = destination / "dataset.yaml"
  yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
  summary = {
    "selected_fold": selected_fold,
    "train_register_crops": {
      "enabled": train_register_crops,
      "context_ratio": register_crop_context,
      "generated_images": generated_register_crops,
    },
    "train_balanced_digit_crops": {
      "enabled": train_balanced_digit_target > 0,
      "target_class_count": train_balanced_digit_target,
      "axis_context_scales": list(DIGIT_CROP_AXIS_CONTEXT_SCALES),
      "perpendicular_context_ratios": list(
        DIGIT_CROP_PERPENDICULAR_CONTEXT_RATIOS
      ),
      "generated_images": sum(generated_digit_class_counts.values()),
      "generated_class_counts": {
        digit: generated_digit_class_counts[digit]
        for digit in CLASS_NAMES
      },
      "source_image_counts": {
        digit: len(generated_digit_source_names[digit])
        for digit in CLASS_NAMES
      },
    },
    "split_images": {
      split: split_image_counts[split]
      for split in ("train", "val", "test")
    },
    "split_class_counts": {
      split: {
        digit: split_class_counts[split][digit]
        for digit in CLASS_NAMES
      }
      for split in ("train", "val", "test")
    },
  }
  return yaml_path, summary


def resolve_device(value: str) -> str | None:
  normalized = value.strip()
  if not normalized or normalized.lower() == "auto":
    return None
  return normalized


def validate_resume_checkpoint(
  checkpoint_path: Path,
  checkpoint: dict,
  args: argparse.Namespace,
  expected_run_dir: Path,
) -> int:
  if checkpoint_path.name != "last.pt":
    raise ValueError("--resume-from must point to an interrupted run's weights/last.pt")
  actual_run_dir = checkpoint_path.parent.parent.resolve()
  if actual_run_dir != expected_run_dir.resolve():
    raise ValueError(
      "Resume checkpoint run directory does not match --project/--name: "
      f"{actual_run_dir} != {expected_run_dir.resolve()}"
    )

  epoch = int(checkpoint.get("epoch", -1))
  if epoch < 0:
    raise ValueError("Resume checkpoint does not contain a completed epoch")
  if checkpoint.get("optimizer") is None:
    raise ValueError(
      "Resume checkpoint has no optimizer state; use an unstripped interrupted "
      "weights/last.pt checkpoint"
    )

  checkpoint_args = checkpoint.get("train_args") or {}
  expected_args = {
    "epochs": args.epochs,
    "imgsz": args.imgsz,
    "batch": args.batch,
    "seed": args.seed,
  }
  mismatches = {
    key: (checkpoint_args.get(key), value)
    for key, value in expected_args.items()
    if checkpoint_args.get(key) != value
  }
  if mismatches:
    details = ", ".join(
      f"{key}: checkpoint={actual!r}, requested={expected!r}"
      for key, (actual, expected) in mismatches.items()
    )
    raise ValueError(f"Resume arguments differ from the checkpoint: {details}")
  if epoch + 1 >= args.epochs:
    raise ValueError(
      f"Checkpoint already completed {epoch + 1} of {args.epochs} epochs"
    )
  return epoch + 1


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent
  annotations_path = resolve_path(base_dir, args.annotations)
  folds_path = resolve_path(base_dir, args.folds)
  source_exclusions_path = resolve_path(base_dir, args.source_exclusions)
  source_images_root = resolve_path(base_dir, args.source_images)
  labels_root = resolve_path(base_dir, args.labels)
  project_path = resolve_path(base_dir, args.project)
  copy_to_path = resolve_path(base_dir, args.copy_to) if args.copy_to else None
  resume_from_path = (
    resolve_path(base_dir, args.resume_from) if args.resume_from else None
  )
  run_name = args.name.strip() or f"full-image-digit-detector-fold{args.fold}"
  run_dir = project_path / run_name
  if resume_from_path is not None and args.pretrained_model:
    raise ValueError("--resume-from cannot be combined with --pretrained-model")
  if resume_from_path is not None and args.validate_only:
    raise ValueError("--resume-from cannot be combined with --validate-only")

  source_exclusions = read_source_exclusions(source_exclusions_path)
  annotation_rows = filter_excluded_annotations(
    read_csv_rows(annotations_path),
    set(source_exclusions),
  )
  fold_assignments = read_fold_assignments(folds_path)
  temporary_parent = Path(tempfile.mkdtemp(prefix="jarvis_full_digit_detector_"))
  dataset_root = temporary_parent / "dataset"
  try:
    dataset_yaml, dataset_summary = materialize_fold_dataset(
      dataset_root,
      annotation_rows,
      fold_assignments,
      source_images_root,
      labels_root,
      args.fold,
      train_register_crops=args.train_register_crops,
      register_crop_context=args.register_crop_context,
      train_balanced_digit_target=args.train_balanced_digit_target,
    )
    provenance = {
      **dataset_summary,
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
      "base_model": args.base_model,
      "pretrained_model": args.pretrained_model or None,
      "epochs": args.epochs,
      "imgsz": args.imgsz,
      "batch": args.batch,
      "patience": args.patience,
      "seed": args.seed,
      "device": args.device,
      "resume_from": str(resume_from_path) if resume_from_path else None,
      "ultralytics_version": metadata.version("ultralytics"),
      "augmentation": TRAIN_AUGMENT_KWARGS,
    }
    print(json.dumps(provenance, indent=2))
    if args.validate_only:
      print(f"Validated temporary dataset: {dataset_yaml}")
      return

    matplotlib_cache = Path(tempfile.gettempdir()) / "jarvis-matplotlib-cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    try:
      from ultralytics import YOLO
    except ImportError as error:
      raise RuntimeError("ultralytics is required. Install backend/requirements.txt.") from error

    resume = resume_from_path is not None
    if resume:
      if not resume_from_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_from_path}")
      model = YOLO(str(resume_from_path))
      completed_epochs = validate_resume_checkpoint(
        resume_from_path,
        model.ckpt or {},
        args,
        run_dir,
      )
      print(
        f"Resuming {run_name} after {completed_epochs} completed epochs "
        f"from {resume_from_path}"
      )
    else:
      local_base_model = resolve_path(base_dir, args.base_model)
      model_source = (
        str(local_base_model) if local_base_model.exists() else args.base_model
      )
      model = YOLO(model_source)
      if args.pretrained_model:
        local_pretrained_model = resolve_path(base_dir, args.pretrained_model)
        pretrained_source = (
          str(local_pretrained_model)
          if local_pretrained_model.exists()
          else args.pretrained_model
        )
        model.load(pretrained_source)
    train_kwargs = dict(
      data=str(dataset_yaml),
      epochs=args.epochs,
      imgsz=args.imgsz,
      batch=args.batch,
      patience=args.patience,
      workers=args.workers,
      project=str(project_path),
      name=run_name,
      device=resolve_device(args.device),
      seed=args.seed,
      deterministic=True,
      **TRAIN_AUGMENT_KWARGS,
    )
    if resume:
      train_kwargs["resume"] = True
    model.train(**train_kwargs)

    trainer = getattr(model, "trainer", None)
    save_dir_value = getattr(trainer, "save_dir", None) if trainer is not None else None
    if not save_dir_value:
      raise RuntimeError("Training completed without an available run directory.")
    save_dir = Path(save_dir_value)
    best_path = Path(getattr(trainer, "best", "") or save_dir / "weights" / "best.pt")
    if not best_path.exists():
      raise FileNotFoundError(f"Training completed but best checkpoint is missing: {best_path}")
    (save_dir / "dataset_provenance.json").write_text(
      json.dumps(provenance, indent=2) + "\n",
      encoding="utf-8",
    )
    print(f"Best checkpoint: {best_path}")

    if copy_to_path is not None:
      copy_to_path.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(best_path, copy_to_path)
      print(f"Copied checkpoint to: {copy_to_path}")
  finally:
    shutil.rmtree(temporary_parent, ignore_errors=True)


if __name__ == "__main__":
  main()
