from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

try:
  from .runtime_digit_pipeline import rotate_image
except ImportError:
  from runtime_digit_pipeline import rotate_image


VALID_SPLITS = ("train", "val", "test")
CLASS_NAMES = tuple(str(value) for value in range(10))
ANNOTATION_HEADERS = [
  "split",
  "filename",
  "reading",
  "position",
  "digit",
  "class_id",
  "x_center",
  "y_center",
  "width",
  "height",
  "x0_px",
  "y0_px",
  "x1_px",
  "y1_px",
  "image_width",
  "image_height",
  "direction_rotation",
  "annotation_source",
  "review_status",
  "transition_state",
  "notes",
]
BOOTSTRAP_HEADERS = [
  *ANNOTATION_HEADERS,
  "roi_x_center",
  "roi_y_center",
  "roi_width",
  "roi_height",
  "major_inset_ratio",
  "minor_inset_ratio",
]
CV_FOLD_HEADERS = ["filename", "reading", "fold"]
SOURCE_EXCLUSION_HEADERS = [
  "filename",
  "scope",
  "reason",
  "retention",
  "notes",
]
FULL_IMAGE_DIGIT_SCOPE = "full_image_digit_detection"
BOX_COLORS = (
  (0, 210, 255),
  (255, 170, 0),
  (60, 220, 120),
  (255, 90, 150),
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Bootstrap one full-image YOLO digit-wheel box per reading position from "
      "the existing register ROI, verified reading, and canonical orientation."
    )
  )
  parser.add_argument(
    "--readings",
    default="../assets/meter_readings.csv",
    help="CSV containing filename,value rows (default resolves from backend/).",
  )
  parser.add_argument(
    "--roi-dataset",
    default="data/roi_dataset",
    help="ROI dataset containing images/<split> and labels/<split>.",
  )
  parser.add_argument(
    "--canonical-manifest",
    default="data/digit_dataset/manifests/canonical_windows.csv",
    help="Canonical window manifest containing reading orientation metadata.",
  )
  parser.add_argument(
    "--out-dir",
    default="data/full_image_digit_dataset",
    help="Versioned output root for manifests and YOLO labels.",
  )
  parser.add_argument(
    "--source-exclusions",
    default="",
    help=(
      "CSV of sources retained for diagnostics but excluded from active "
      "training and CV. Defaults to manifests/source_exclusions.csv under "
      "--out-dir."
    ),
  )
  parser.add_argument(
    "--review-dir",
    default="../output/full-image-digit-review",
    help="Disposable Make Sense review package and visual QA output.",
  )
  parser.add_argument(
    "--major-inset-ratio",
    type=float,
    default=0.08,
    help="Inset each digit cell along the register axis.",
  )
  parser.add_argument(
    "--minor-inset-ratio",
    type=float,
    default=0.15,
    help="Inset each digit cell across the register axis.",
  )
  parser.add_argument(
    "--skip-review-package",
    action="store_true",
    help="Build manifests and labels without copying images or rendering previews.",
  )
  parser.add_argument(
    "--rebuild-cv-folds",
    action="store_true",
    help=(
      "Discard and recompute generated CV assignments. "
      "Use only before experiments or after a deliberate fold-policy change."
    ),
  )
  return parser.parse_args()


def resolve(base_dir: Path, value: str) -> Path:
  path = Path(value)
  if path.is_absolute():
    return path
  return (base_dir / path).resolve()


def parse_float(raw: str | None, field: str, filename: str) -> float:
  try:
    return float((raw or "").strip())
  except ValueError as error:
    raise ValueError(f"Invalid {field} for {filename}: {raw!r}") from error


def read_readings(path: Path) -> dict[str, str]:
  readings: dict[str, str] = {}
  with path.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      filename = (row.get("filename") or "").strip()
      reading = (row.get("value") or "").strip()
      if not filename:
        continue
      if len(reading) != 4 or not reading.isdigit():
        raise ValueError(f"Expected a four-digit reading for {filename}: {reading!r}")
      readings[filename] = reading
  return readings


def read_canonical_rows(path: Path) -> dict[str, dict[str, str]]:
  rows: dict[str, dict[str, str]] = {}
  with path.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      filename = (row.get("filename") or "").strip()
      if filename:
        rows[filename] = {key: (value or "") for key, value in row.items()}
  return rows


def load_roi_label(path: Path) -> tuple[float, float, float, float]:
  for line in path.read_text(encoding="utf-8").splitlines():
    parts = line.strip().split()
    if len(parts) != 5:
      continue
    _, x_center, y_center, width, height = parts
    values = tuple(float(value) for value in (x_center, y_center, width, height))
    if values[2] <= 0 or values[3] <= 0:
      break
    return values
  raise ValueError(f"No valid YOLO ROI row in {path}")


def direction_rotation(row: dict[str, str], filename: str) -> int:
  rotation = int(round(parse_float(row.get("applied_rotation"), "applied_rotation", filename))) % 360
  canonical_rotation = parse_float(
    row.get("canonical_rotate_degrees"),
    "canonical_rotate_degrees",
    filename,
  )
  if abs(canonical_rotation) >= 135:
    rotation = (rotation + 180) % 360
  if rotation not in {0, 90, 180, 270}:
    raise ValueError(f"Unsupported reading-direction rotation for {filename}: {rotation}")
  return rotation


def clamp01(value: float) -> float:
  return max(0.0, min(1.0, value))


def build_digit_boxes(
  reading: str,
  roi: tuple[float, float, float, float],
  rotation: int,
  major_inset_ratio: float,
  minor_inset_ratio: float,
) -> list[dict[str, float | int | str]]:
  if len(reading) != 4 or not reading.isdigit():
    raise ValueError(f"Expected a four-digit reading, got {reading!r}")
  if not 0 <= major_inset_ratio < 0.5:
    raise ValueError("--major-inset-ratio must be in [0, 0.5).")
  if not 0 <= minor_inset_ratio < 0.5:
    raise ValueError("--minor-inset-ratio must be in [0, 0.5).")
  if rotation not in {0, 90, 180, 270}:
    raise ValueError(f"Unsupported rotation: {rotation}")

  roi_x_center, roi_y_center, roi_width, roi_height = roi
  roi_x0 = roi_x_center - roi_width * 0.5
  roi_y0 = roi_y_center - roi_height * 0.5
  horizontal = rotation in {0, 180}
  boxes: list[dict[str, float | int | str]] = []

  for position, digit in enumerate(reading):
    if horizontal:
      physical_index = position if rotation == 0 else 3 - position
      cell_width = roi_width / 4
      x0 = roi_x0 + (physical_index + major_inset_ratio) * cell_width
      x1 = roi_x0 + (physical_index + 1 - major_inset_ratio) * cell_width
      y0 = roi_y0 + minor_inset_ratio * roi_height
      y1 = roi_y0 + (1 - minor_inset_ratio) * roi_height
    else:
      # Runtime positive 90-degree rotation is clockwise and maps source
      # bottom-to-top into canonical left-to-right reading order.
      physical_index = 3 - position if rotation == 90 else position
      cell_height = roi_height / 4
      x0 = roi_x0 + minor_inset_ratio * roi_width
      x1 = roi_x0 + (1 - minor_inset_ratio) * roi_width
      y0 = roi_y0 + (physical_index + major_inset_ratio) * cell_height
      y1 = roi_y0 + (physical_index + 1 - major_inset_ratio) * cell_height

    x0 = clamp01(x0)
    y0 = clamp01(y0)
    x1 = clamp01(x1)
    y1 = clamp01(y1)
    if x1 <= x0 or y1 <= y0:
      raise ValueError(f"Degenerate digit box for position {position}")

    boxes.append({
      "position": position,
      "digit": digit,
      "class_id": int(digit),
      "x_center": (x0 + x1) * 0.5,
      "y_center": (y0 + y1) * 0.5,
      "width": x1 - x0,
      "height": y1 - y0,
    })
  return boxes


def write_csv(path: Path, rows: list[dict[str, str]], headers: list[str]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=headers, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
  if not path.exists():
    return []
  with path.open("r", encoding="utf-8") as handle:
    return [
      {key: (value or "") for key, value in row.items()}
      for row in csv.DictReader(handle)
    ]


def read_source_exclusions(path: Path) -> dict[str, dict[str, str]]:
  rows = read_csv_rows(path)
  exclusions: dict[str, dict[str, str]] = {}
  for row in rows:
    normalized = {
      header: (row.get(header) or "").strip()
      for header in SOURCE_EXCLUSION_HEADERS
    }
    filename = normalized["filename"]
    if not filename:
      raise ValueError(f"Source exclusion row is missing filename: {path}")
    if filename in exclusions:
      raise ValueError(f"Duplicate source exclusion for {filename}: {path}")
    if normalized["scope"] != FULL_IMAGE_DIGIT_SCOPE:
      raise ValueError(
        f"Unsupported source exclusion scope for {filename}: "
        f"{normalized['scope']!r}"
      )
    if not normalized["reason"]:
      raise ValueError(f"Source exclusion is missing a reason for {filename}")
    if normalized["retention"] != "legacy_stress":
      raise ValueError(
        f"Source exclusion retention for {filename} must be 'legacy_stress'"
      )
    exclusions[filename] = normalized
  return exclusions


def filter_excluded_annotations(
  rows: list[dict[str, str]],
  excluded_filenames: set[str],
) -> list[dict[str, str]]:
  return [
    row
    for row in rows
    if row["filename"] not in excluded_filenames
  ]


def annotation_key(row: dict[str, str]) -> tuple[str, int]:
  return row["filename"], int(row["position"])


def seed_or_preserve_annotations(
  path: Path,
  bootstrap_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
  existing_rows = read_csv_rows(path)
  if not existing_rows:
    seeded = [
      {header: row.get(header, "") for header in ANNOTATION_HEADERS}
      for row in bootstrap_rows
    ]
    write_csv(path, seeded, ANNOTATION_HEADERS)
    return seeded, len(seeded)

  existing_by_key = {annotation_key(row): row for row in existing_rows}
  bootstrap_keys = {annotation_key(row) for row in bootstrap_rows}
  stale_keys = sorted(set(existing_by_key) - bootstrap_keys)
  if stale_keys:
    raise ValueError(
      "Refusing to discard existing human-review annotations for missing sources: "
      + ", ".join(f"{filename}:p{position}" for filename, position in stale_keys)
    )

  merged: list[dict[str, str]] = []
  seeded_count = 0
  for bootstrap in bootstrap_rows:
    key = annotation_key(bootstrap)
    existing = existing_by_key.get(key)
    if existing is None:
      merged.append({header: bootstrap.get(header, "") for header in ANNOTATION_HEADERS})
      seeded_count += 1
    else:
      merged.append({header: existing.get(header, "") for header in ANNOTATION_HEADERS})
  write_csv(path, merged, ANNOTATION_HEADERS)
  return merged, seeded_count


def digit_counts(reading: str) -> Counter[str]:
  return Counter(reading)


def seed_or_preserve_cv_folds(
  path: Path,
  annotation_rows: list[dict[str, str]],
  fold_count: int = 5,
  excluded_filenames: set[str] | None = None,
) -> tuple[list[dict[str, str]], int]:
  if fold_count < 2:
    raise ValueError("Cross-validation requires at least two folds.")

  train_readings: dict[str, str] = {}
  for row in annotation_rows:
    if row["split"] != "train":
      continue
    filename = row["filename"]
    reading = row["reading"]
    existing_reading = train_readings.get(filename)
    if existing_reading is not None and existing_reading != reading:
      raise ValueError(f"Conflicting train readings for {filename}")
    train_readings[filename] = reading

  existing_rows = read_csv_rows(path)
  existing_by_filename: dict[str, dict[str, str]] = {}
  for row in existing_rows:
    filename = row.get("filename", "")
    if not filename or filename in existing_by_filename:
      raise ValueError(f"Invalid duplicate CV fold row: {filename!r}")
    try:
      fold = int(row.get("fold", ""))
    except ValueError as error:
      raise ValueError(f"Invalid CV fold for {filename}: {row.get('fold')!r}") from error
    if fold not in range(fold_count):
      raise ValueError(f"CV fold for {filename} must be in 0..{fold_count - 1}")
    existing_by_filename[filename] = {
      "filename": filename,
      "reading": train_readings.get(filename, row.get("reading", "")),
      "fold": str(fold),
    }

  excluded_filenames = excluded_filenames or set()
  explicitly_excluded = set(existing_by_filename) & excluded_filenames
  for filename in explicitly_excluded:
    del existing_by_filename[filename]

  stale = sorted(set(existing_by_filename) - set(train_readings))
  if stale:
    raise ValueError(
      "Refusing to discard persistent CV assignments for missing train sources: "
      + ", ".join(stale)
    )

  total_class_counts: Counter[str] = Counter()
  for reading in train_readings.values():
    total_class_counts.update(reading)
  class_targets = {
    digit: total_class_counts[digit] / fold_count
    for digit in CLASS_NAMES
  }

  fold_sizes = [0] * fold_count
  fold_class_counts = [Counter() for _ in range(fold_count)]
  for row in existing_by_filename.values():
    fold = int(row["fold"])
    fold_sizes[fold] += 1
    fold_class_counts[fold].update(train_readings[row["filename"]])

  total_images = len(train_readings)
  base_size, remainder = divmod(total_images, fold_count)
  fold_capacities = [
    base_size + (1 if fold < remainder else 0)
    for fold in range(fold_count)
  ]

  new_filenames = sorted(
    set(train_readings) - set(existing_by_filename),
    key=lambda filename: (
      -sum(
        1 / max(total_class_counts[digit], 1)
        for digit in train_readings[filename]
      ),
      filename,
    ),
  )
  for filename in new_filenames:
    reading = train_readings[filename]
    image_counts = digit_counts(reading)
    candidates = [
      fold
      for fold in range(fold_count)
      if fold_sizes[fold] < fold_capacities[fold]
    ]
    if not candidates:
      candidates = list(range(fold_count))

    def assignment_score(fold: int) -> tuple[float, int, int]:
      projected = fold_class_counts[fold] + image_counts
      current_error = sum(
        (
          (fold_class_counts[fold][digit] - class_targets[digit])
          / max(class_targets[digit], 1.0)
        ) ** 2
        for digit in CLASS_NAMES
      )
      projected_error = sum(
        (
          (projected[digit] - class_targets[digit])
          / max(class_targets[digit], 1.0)
        ) ** 2
        for digit in CLASS_NAMES
      )
      return projected_error - current_error, fold_sizes[fold], fold

    chosen_fold = min(candidates, key=assignment_score)
    existing_by_filename[filename] = {
      "filename": filename,
      "reading": reading,
      "fold": str(chosen_fold),
    }
    fold_sizes[chosen_fold] += 1
    fold_class_counts[chosen_fold].update(reading)

  rows = [
    existing_by_filename[filename]
    for filename in sorted(existing_by_filename)
  ]
  write_csv(path, rows, CV_FOLD_HEADERS)
  return rows, len(new_filenames)


def summarize_cv_folds(
  rows: list[dict[str, str]],
  fold_count: int = 5,
) -> dict[str, object]:
  image_counts = Counter(int(row["fold"]) for row in rows)
  class_counts = [Counter() for _ in range(fold_count)]
  for row in rows:
    class_counts[int(row["fold"])].update(row["reading"])
  return {
    "fold_count": fold_count,
    "image_counts": {
      str(fold): image_counts[fold]
      for fold in range(fold_count)
    },
    "class_counts": {
      str(fold): {
        digit: class_counts[fold][digit]
        for digit in CLASS_NAMES
      }
      for fold in range(fold_count)
    },
  }


def validate_annotations(
  rows: list[dict[str, str]],
  expected_images: set[str],
) -> dict[str, object]:
  grouped: dict[str, list[dict[str, str]]] = {}
  class_counts: Counter[str] = Counter()
  split_counts: Counter[str] = Counter()
  review_counts: Counter[str] = Counter()

  for row in rows:
    filename = row["filename"]
    split = row["split"]
    if split not in VALID_SPLITS:
      raise ValueError(f"Invalid split for {filename}: {split}")
    position = int(row["position"])
    digit = row["digit"]
    class_id = int(row["class_id"])
    if position not in {0, 1, 2, 3}:
      raise ValueError(f"Invalid position for {filename}: {position}")
    if digit not in CLASS_NAMES or class_id != int(digit):
      raise ValueError(f"Digit/class mismatch for {filename} position {position}")
    for field in ("x_center", "y_center", "width", "height"):
      value = float(row[field])
      if not 0 < value <= 1:
        raise ValueError(f"Invalid {field} for {filename} position {position}: {value}")
    x_center = float(row["x_center"])
    y_center = float(row["y_center"])
    width = float(row["width"])
    height = float(row["height"])
    if x_center - width * 0.5 < -1e-6 or x_center + width * 0.5 > 1 + 1e-6:
      raise ValueError(f"Horizontal bounds exceed image for {filename} position {position}")
    if y_center - height * 0.5 < -1e-6 or y_center + height * 0.5 > 1 + 1e-6:
      raise ValueError(f"Vertical bounds exceed image for {filename} position {position}")
    grouped.setdefault(filename, []).append(row)
    class_counts[digit] += 1
    review_counts[row.get("review_status") or "unspecified"] += 1

  if set(grouped) != expected_images:
    missing = sorted(expected_images - set(grouped))
    extra = sorted(set(grouped) - expected_images)
    raise ValueError(f"Annotation image mismatch: missing={missing}, extra={extra}")

  for filename, image_rows in grouped.items():
    positions = sorted(int(row["position"]) for row in image_rows)
    if positions != [0, 1, 2, 3]:
      raise ValueError(f"Expected positions 0..3 for {filename}, got {positions}")
    readings = {row["reading"] for row in image_rows}
    if len(readings) != 1:
      raise ValueError(f"Conflicting readings for {filename}: {sorted(readings)}")
    ordered_digits = "".join(
      row["digit"]
      for row in sorted(image_rows, key=lambda item: int(item["position"]))
    )
    if ordered_digits != next(iter(readings)):
      raise ValueError(
        f"Reading/annotation mismatch for {filename}: {next(iter(readings))} != {ordered_digits}"
      )
    split_counts[image_rows[0]["split"]] += 1

  return {
    "images": len(grouped),
    "annotations": len(rows),
    "split_images": dict(sorted(split_counts.items())),
    "class_counts": {digit: class_counts.get(digit, 0) for digit in CLASS_NAMES},
    "review_counts": dict(sorted(review_counts.items())),
  }


def write_yolo_labels(root: Path, rows: list[dict[str, str]]) -> None:
  if root.exists():
    shutil.rmtree(root)
  grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
  for row in rows:
    grouped.setdefault((row["split"], row["filename"]), []).append(row)

  for (split, filename), image_rows in grouped.items():
    label_path = root / split / f"{Path(filename).stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in sorted(image_rows, key=lambda item: int(item["position"])):
      lines.append(
        f"{int(row['class_id'])} "
        f"{float(row['x_center']):.6f} "
        f"{float(row['y_center']):.6f} "
        f"{float(row['width']):.6f} "
        f"{float(row['height']):.6f}"
      )
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pixel_rect(row: dict[str, str], image_width: int, image_height: int) -> tuple[int, int, int, int]:
  x_center = float(row["x_center"]) * image_width
  y_center = float(row["y_center"]) * image_height
  width = float(row["width"]) * image_width
  height = float(row["height"]) * image_height
  return (
    int(round(x_center - width * 0.5)),
    int(round(y_center - height * 0.5)),
    int(round(x_center + width * 0.5)),
    int(round(y_center + height * 0.5)),
  )


def draw_annotations(image: Image.Image, rows: list[dict[str, str]]) -> Image.Image:
  rendered = image.convert("RGB").copy()
  draw = ImageDraw.Draw(rendered)
  stroke = max(2, int(round(max(rendered.size) / 700)))
  for row in sorted(rows, key=lambda item: int(item["position"])):
    position = int(row["position"])
    color = BOX_COLORS[position]
    rect = pixel_rect(row, rendered.width, rendered.height)
    draw.rectangle(rect, outline=color, width=stroke)
    label = f"p{position}:{row['digit']}"
    text_box = draw.textbbox((0, 0), label)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    text_x = rect[0]
    text_y = max(0, rect[1] - text_height - stroke * 2)
    draw.rectangle(
      (text_x, text_y, text_x + text_width + stroke * 2, text_y + text_height + stroke * 2),
      fill=color,
    )
    draw.text((text_x + stroke, text_y + stroke), label, fill=(0, 0, 0))
  return rendered


def fit_within(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
  scale = min(max_width / image.width, max_height / image.height, 1.0)
  if scale >= 1:
    return image
  target = (
    max(1, int(round(image.width * scale))),
    max(1, int(round(image.height * scale))),
  )
  return image.resize(target, Image.Resampling.LANCZOS)


def crop_around_annotations(
  image: Image.Image,
  rows: list[dict[str, str]],
  context_ratio: float = 0.45,
) -> Image.Image:
  rects = [pixel_rect(row, image.width, image.height) for row in rows]
  x0 = min(rect[0] for rect in rects)
  y0 = min(rect[1] for rect in rects)
  x1 = max(rect[2] for rect in rects)
  y1 = max(rect[3] for rect in rects)
  width = x1 - x0
  height = y1 - y0
  pad_x = max(12, int(round(width * context_ratio)))
  pad_y = max(12, int(round(height * context_ratio)))
  return image.crop((
    max(0, x0 - pad_x),
    max(0, y0 - pad_y),
    min(image.width, x1 + pad_x),
    min(image.height, y1 + pad_y),
  ))


def orient_review_crop(image: Image.Image, rotation: int) -> Image.Image:
  return rotate_image(image, rotation)


def build_contact_sheet(
  preview_paths: list[Path],
  destination: Path,
  columns: int = 4,
) -> None:
  tile_width = 520
  tile_height = 300
  gap = 12
  rows = (len(preview_paths) + columns - 1) // columns
  sheet = Image.new(
    "RGB",
    (
      columns * tile_width + (columns + 1) * gap,
      rows * tile_height + (rows + 1) * gap,
    ),
    color=(28, 30, 34),
  )
  draw = ImageDraw.Draw(sheet)
  for index, path in enumerate(preview_paths):
    with Image.open(path) as source:
      preview = fit_within(source.convert("RGB"), tile_width - 20, tile_height - 40)
    column = index % columns
    row = index // columns
    tile_x = gap + column * (tile_width + gap)
    tile_y = gap + row * (tile_height + gap)
    image_x = tile_x + (tile_width - preview.width) // 2
    image_y = tile_y + 24 + (tile_height - 34 - preview.height) // 2
    sheet.paste(preview, (image_x, image_y))
    draw.text((tile_x + 8, tile_y + 6), path.stem, fill=(240, 240, 240))
  destination.parent.mkdir(parents=True, exist_ok=True)
  sheet.save(destination, "JPEG", quality=90)


def build_review_package(
  review_dir: Path,
  image_paths: dict[str, Path],
  rows: list[dict[str, str]],
) -> None:
  if review_dir.exists():
    shutil.rmtree(review_dir)
  image_dir = review_dir / "images"
  annotation_dir = review_dir / "annotations"
  full_preview_dir = review_dir / "previews" / "full"
  crop_preview_dir = review_dir / "previews" / "crops"
  for directory in (image_dir, annotation_dir, full_preview_dir, crop_preview_dir):
    directory.mkdir(parents=True, exist_ok=True)

  grouped: dict[str, list[dict[str, str]]] = {}
  for row in rows:
    grouped.setdefault(row["filename"], []).append(row)

  crop_preview_paths: list[Path] = []
  for filename in sorted(grouped):
    source_path = image_paths[filename]
    shutil.copy2(source_path, image_dir / filename)
    image_rows = grouped[filename]
    with Image.open(source_path) as source:
      source_rgb = source.convert("RGB")
      rendered = draw_annotations(source_rgb, image_rows)
      full_preview = fit_within(rendered, 1600, 1600)
      full_preview.save(
        full_preview_dir / f"{Path(filename).stem}.jpg",
        "JPEG",
        quality=90,
      )
      crop = crop_around_annotations(rendered, image_rows)
      rotation = int(image_rows[0]["direction_rotation"])
      if rotation:
        crop = orient_review_crop(crop, rotation)
      crop = fit_within(crop, 1400, 700)
      crop_path = crop_preview_dir / f"{Path(filename).stem}.jpg"
      crop.save(crop_path, "JPEG", quality=92)
      crop_preview_paths.append(crop_path)

    label_lines = []
    for row in sorted(image_rows, key=lambda item: int(item["position"])):
      label_lines.append(
        f"{int(row['class_id'])} "
        f"{float(row['x_center']):.6f} "
        f"{float(row['y_center']):.6f} "
        f"{float(row['width']):.6f} "
        f"{float(row['height']):.6f}"
      )
    (annotation_dir / f"{Path(filename).stem}.txt").write_text(
      "\n".join(label_lines) + "\n",
      encoding="utf-8",
    )

  labels_text = "\n".join(CLASS_NAMES) + "\n"
  (annotation_dir / "labels.txt").write_text(labels_text, encoding="utf-8")
  (review_dir / "classes.txt").write_text(labels_text, encoding="utf-8")
  build_contact_sheet(crop_preview_paths, review_dir / "digit-box-contact-sheet.jpg")

  zip_path = review_dir / "makesense-yolo-labels.zip"
  with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.writestr("labels.txt", labels_text)
    for path in sorted(annotation_dir.glob("*.txt")):
      if path.name == "labels.txt":
        continue
      archive.write(path, arcname=path.name)

  instructions = """# Make Sense review package

1. Open https://www.makesense.ai and upload every image under `images/`.
2. Choose Object Detection.
3. At label setup, use "Load labels from file" with `labels.txt`, or create the
   labels `0` through `9` manually in exactly that order.
4. Start the project, then choose Actions > Import Annotations > YOLO.
5. Unzip `makesense-yolo-labels.zip` and select all `meter_*.txt` files in the
   unzipped folder. Do not include `labels.txt` in this annotation import.
6. Review all four boxes in every image. Each box should cover one complete
   digit-wheel aperture, not only the black ink.
7. Export a YOLO ZIP after editing. Keep the original filenames unchanged.

The previews are QA aids only. The YOLO coordinates are normalized, so they
remain valid for the original full-resolution images.

To import the reviewed ZIP, run these commands from `backend/`:

    python import_full_image_digit_annotations.py /path/to/makesense-export.zip
    python build_full_image_digit_dataset.py

The importer verifies all four classes against the trusted reading before it
updates the canonical annotation manifest.
"""
  (review_dir / "README.md").write_text(instructions, encoding="utf-8")


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent
  readings_path = resolve(base_dir, args.readings)
  roi_dataset = resolve(base_dir, args.roi_dataset)
  canonical_manifest_path = resolve(base_dir, args.canonical_manifest)
  out_dir = resolve(base_dir, args.out_dir)
  review_dir = resolve(base_dir, args.review_dir)
  source_exclusions_path = (
    resolve(base_dir, args.source_exclusions)
    if args.source_exclusions
    else out_dir / "manifests" / "source_exclusions.csv"
  )

  readings = read_readings(readings_path)
  canonical_rows = read_canonical_rows(canonical_manifest_path)
  source_exclusions = read_source_exclusions(source_exclusions_path)
  excluded_filenames = set(source_exclusions)
  unknown_exclusions = sorted(excluded_filenames - set(readings))
  if unknown_exclusions:
    raise ValueError(
      "Source exclusions are missing from the trusted readings: "
      + ", ".join(unknown_exclusions)
    )
  bootstrap_rows: list[dict[str, str]] = []
  image_paths: dict[str, Path] = {}

  for filename in sorted(readings):
    canonical = canonical_rows.get(filename)
    if canonical is None:
      raise ValueError(f"Missing canonical orientation metadata for {filename}")
    split = (canonical.get("split") or "").strip()
    if split not in VALID_SPLITS:
      raise ValueError(f"Invalid split for {filename}: {split!r}")
    image_path = roi_dataset / "images" / split / filename
    label_path = roi_dataset / "labels" / split / f"{Path(filename).stem}.txt"
    if not image_path.exists():
      raise FileNotFoundError(f"Missing source image: {image_path}")
    if not label_path.exists():
      raise FileNotFoundError(f"Missing ROI label: {label_path}")
    image_paths[filename] = image_path

    roi = load_roi_label(label_path)
    rotation = direction_rotation(canonical, filename)
    boxes = build_digit_boxes(
      readings[filename],
      roi,
      rotation,
      args.major_inset_ratio,
      args.minor_inset_ratio,
    )
    with Image.open(image_path) as image:
      image_width, image_height = image.size

    for box in boxes:
      x_center = float(box["x_center"])
      y_center = float(box["y_center"])
      width = float(box["width"])
      height = float(box["height"])
      x0 = int(round((x_center - width * 0.5) * image_width))
      y0 = int(round((y_center - height * 0.5) * image_height))
      x1 = int(round((x_center + width * 0.5) * image_width))
      y1 = int(round((y_center + height * 0.5) * image_height))
      bootstrap_rows.append({
        "split": split,
        "filename": filename,
        "reading": readings[filename],
        "position": str(box["position"]),
        "digit": str(box["digit"]),
        "class_id": str(box["class_id"]),
        "x_center": f"{x_center:.8f}",
        "y_center": f"{y_center:.8f}",
        "width": f"{width:.8f}",
        "height": f"{height:.8f}",
        "x0_px": str(x0),
        "y0_px": str(y0),
        "x1_px": str(x1),
        "y1_px": str(y1),
        "image_width": str(image_width),
        "image_height": str(image_height),
        "direction_rotation": str(rotation),
        "annotation_source": "bootstrap-roi-split",
        "review_status": "pending",
        "transition_state": "unknown",
        "notes": "",
        "roi_x_center": f"{roi[0]:.8f}",
        "roi_y_center": f"{roi[1]:.8f}",
        "roi_width": f"{roi[2]:.8f}",
        "roi_height": f"{roi[3]:.8f}",
        "major_inset_ratio": f"{args.major_inset_ratio:.4f}",
        "minor_inset_ratio": f"{args.minor_inset_ratio:.4f}",
      })

  manifest_dir = out_dir / "manifests"
  write_csv(manifest_dir / "bootstrap_boxes.csv", bootstrap_rows, BOOTSTRAP_HEADERS)
  annotation_rows, seeded_count = seed_or_preserve_annotations(
    manifest_dir / "annotations.csv",
    bootstrap_rows,
  )
  retained_validation = validate_annotations(annotation_rows, set(readings))
  active_rows = filter_excluded_annotations(
    annotation_rows,
    excluded_filenames,
  )
  active_filenames = set(readings) - excluded_filenames
  validation = validate_annotations(active_rows, active_filenames)
  cv_fold_path = manifest_dir / "cv_folds.csv"
  if args.rebuild_cv_folds:
    cv_fold_path.unlink(missing_ok=True)
  cv_fold_rows, seeded_fold_count = seed_or_preserve_cv_folds(
    cv_fold_path,
    active_rows,
    excluded_filenames=excluded_filenames,
  )
  write_yolo_labels(out_dir / "labels", active_rows)
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

  review_package: str | None = None
  if not args.skip_review_package:
    try:
      review_package = review_dir.relative_to(base_dir.parent).as_posix()
    except ValueError:
      review_package = str(review_dir)
  try:
    source_exclusions_manifest = source_exclusions_path.relative_to(out_dir).as_posix()
  except ValueError:
    source_exclusions_manifest = str(source_exclusions_path)
  summary = {
    "version": 1,
    **validation,
    "annotation_policy": "one complete digit-wheel aperture per reading position",
    "bootstrap_source": "existing register ROI + verified reading + canonical orientation",
    "major_inset_ratio": args.major_inset_ratio,
    "minor_inset_ratio": args.minor_inset_ratio,
    "new_annotations_seeded": seeded_count,
    "new_cv_assignments_seeded": seeded_fold_count,
    "cross_validation": summarize_cv_folds(cv_fold_rows),
    "source_exclusions": {
      "manifest": source_exclusions_manifest,
      "count": len(source_exclusions),
      "entries": [
        source_exclusions[filename]
        for filename in sorted(source_exclusions)
      ],
      "retained_annotation_images": retained_validation["images"],
      "active_images": validation["images"],
    },
    "review_package": review_package,
  }
  (manifest_dir / "summary.json").write_text(
    json.dumps(summary, indent=2) + "\n",
    encoding="utf-8",
  )

  if not args.skip_review_package:
    build_review_package(
      review_dir,
      {
        filename: path
        for filename, path in image_paths.items()
        if filename in active_filenames
      },
      active_rows,
    )

  print(f"Full-image digit dataset: {out_dir}")
  print(f"Images: {validation['images']}")
  print(f"Annotations: {validation['annotations']}")
  print(f"Review states: {validation['review_counts']}")
  if not args.skip_review_package:
    print(f"Make Sense review package: {review_dir}")


if __name__ == "__main__":
  main()
