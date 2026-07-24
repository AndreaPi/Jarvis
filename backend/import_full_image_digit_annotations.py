from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

try:
  from backend.build_full_image_digit_dataset import (
    ANNOTATION_HEADERS,
    VALID_SPLITS,
    annotation_key,
    read_source_exclusions,
    validate_annotations,
    write_csv,
  )
except ModuleNotFoundError:
  from build_full_image_digit_dataset import (
    ANNOTATION_HEADERS,
    VALID_SPLITS,
    annotation_key,
    read_source_exclusions,
    validate_annotations,
    write_csv,
  )


IGNORED_LABEL_FILES = {"classes.txt", "labels.txt"}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Import reviewed Make Sense YOLO boxes into the canonical full-image "
      "digit annotation manifest."
    )
  )
  parser.add_argument(
    "export",
    help="Make Sense YOLO ZIP or directory containing one TXT file per image.",
  )
  parser.add_argument(
    "--annotations",
    default="data/full_image_digit_dataset/manifests/annotations.csv",
    help="Canonical annotation manifest to update.",
  )
  parser.add_argument(
    "--source-exclusions",
    default="data/full_image_digit_dataset/manifests/source_exclusions.csv",
    help="Legacy-stress sources omitted from the active review package.",
  )
  parser.add_argument(
    "--allow-partial",
    action="store_true",
    help="Allow an export containing only a subset of dataset images.",
  )
  return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
  with path.open("r", encoding="utf-8") as handle:
    rows = [
      {key: (value or "") for key, value in row.items()}
      for row in csv.DictReader(handle)
    ]
  if not rows:
    raise ValueError(f"Annotation manifest is empty: {path}")
  return rows


def read_export_files(path: Path) -> dict[str, str]:
  files: dict[str, str] = {}
  if path.is_dir():
    candidates = sorted(path.rglob("*.txt"))
    for candidate in candidates:
      if candidate.name.lower() in IGNORED_LABEL_FILES:
        continue
      if candidate.stem in files:
        raise ValueError(f"Duplicate label file stem in export: {candidate.stem}")
      files[candidate.stem] = candidate.read_text(encoding="utf-8")
    return files

  if not zipfile.is_zipfile(path):
    raise ValueError(f"Expected a Make Sense YOLO ZIP or directory: {path}")
  with zipfile.ZipFile(path) as archive:
    for entry in archive.infolist():
      entry_path = Path(entry.filename)
      if entry.is_dir() or entry_path.suffix.lower() != ".txt":
        continue
      if entry_path.name.lower() in IGNORED_LABEL_FILES:
        continue
      if entry_path.stem in files:
        raise ValueError(f"Duplicate label file stem in export: {entry_path.stem}")
      with archive.open(entry) as handle:
        files[entry_path.stem] = handle.read().decode("utf-8")
  return files


def parse_yolo_rows(text: str, stem: str) -> list[dict[str, float | int]]:
  boxes: list[dict[str, float | int]] = []
  for line_number, line in enumerate(text.splitlines(), start=1):
    parts = line.strip().split()
    if not parts:
      continue
    if len(parts) != 5:
      raise ValueError(
        f"{stem}.txt line {line_number}: expected 5 YOLO values, got {len(parts)}"
      )
    try:
      class_id = int(parts[0])
      x_center, y_center, width, height = (float(value) for value in parts[1:])
    except ValueError as error:
      raise ValueError(
        f"{stem}.txt line {line_number}: invalid numeric YOLO row"
      ) from error
    if class_id not in range(10):
      raise ValueError(f"{stem}.txt line {line_number}: invalid class {class_id}")
    if not 0 < width <= 1 or not 0 < height <= 1:
      raise ValueError(f"{stem}.txt line {line_number}: invalid box size")
    if x_center - width * 0.5 < 0 or x_center + width * 0.5 > 1:
      raise ValueError(f"{stem}.txt line {line_number}: horizontal bounds exceed image")
    if y_center - height * 0.5 < 0 or y_center + height * 0.5 > 1:
      raise ValueError(f"{stem}.txt line {line_number}: vertical bounds exceed image")
    boxes.append({
      "class_id": class_id,
      "x_center": x_center,
      "y_center": y_center,
      "width": width,
      "height": height,
    })
  if len(boxes) != 4:
    raise ValueError(f"{stem}.txt: expected exactly 4 digit boxes, got {len(boxes)}")
  return boxes


def sort_boxes_in_reading_order(
  boxes: list[dict[str, float | int]],
  rotation: int,
) -> list[dict[str, float | int]]:
  if rotation == 0:
    return sorted(boxes, key=lambda box: float(box["x_center"]))
  if rotation == 180:
    return sorted(boxes, key=lambda box: float(box["x_center"]), reverse=True)
  if rotation == 90:
    return sorted(boxes, key=lambda box: float(box["y_center"]), reverse=True)
  if rotation == 270:
    return sorted(boxes, key=lambda box: float(box["y_center"]))
  raise ValueError(f"Unsupported reading-direction rotation: {rotation}")


def merge_reviewed_export(
  annotation_rows: list[dict[str, str]],
  export_files: dict[str, str],
  allow_partial: bool = False,
  excluded_filenames: set[str] | None = None,
) -> list[dict[str, str]]:
  grouped: dict[str, list[dict[str, str]]] = {}
  for row in annotation_rows:
    if row["split"] not in VALID_SPLITS:
      raise ValueError(f"Invalid split for {row['filename']}: {row['split']}")
    grouped.setdefault(row["filename"], []).append(row)

  excluded_filenames = excluded_filenames or set()
  unknown_exclusions = sorted(excluded_filenames - set(grouped))
  if unknown_exclusions:
    raise ValueError(
      "Source exclusions have no canonical annotations: "
      + ", ".join(unknown_exclusions)
    )
  active_grouped = {
    filename: rows
    for filename, rows in grouped.items()
    if filename not in excluded_filenames
  }

  stems: dict[str, str] = {}
  for filename in active_grouped:
    stem = Path(filename).stem
    if stem in stems:
      raise ValueError(f"Dataset filenames have duplicate stems: {stem}")
    stems[stem] = filename

  unknown_stems = sorted(set(export_files) - set(stems))
  if unknown_stems:
    raise ValueError("Export contains unknown images: " + ", ".join(unknown_stems))
  missing_stems = sorted(set(stems) - set(export_files))
  if missing_stems and not allow_partial:
    raise ValueError(
      "Export is missing dataset images: "
      + ", ".join(missing_stems)
      + ". Use --allow-partial only for an intentional subset review."
    )

  updated_by_key: dict[tuple[str, int], dict[str, str]] = {}
  for stem, text in export_files.items():
    filename = stems[stem]
    current_rows = sorted(grouped[filename], key=lambda row: int(row["position"]))
    if len(current_rows) != 4:
      raise ValueError(f"Expected 4 canonical annotations for {filename}")
    rotation_values = {int(row["direction_rotation"]) for row in current_rows}
    if len(rotation_values) != 1:
      raise ValueError(f"Conflicting direction rotations for {filename}")
    boxes = sort_boxes_in_reading_order(
      parse_yolo_rows(text, stem),
      next(iter(rotation_values)),
    )
    expected_reading = current_rows[0]["reading"]
    imported_reading = "".join(str(box["class_id"]) for box in boxes)
    if imported_reading != expected_reading:
      raise ValueError(
        f"{filename}: imported classes read {imported_reading}, "
        f"but the verified reading is {expected_reading}"
      )

    for current, box in zip(current_rows, boxes, strict=True):
      updated = dict(current)
      image_width = int(current["image_width"])
      image_height = int(current["image_height"])
      x_center = float(box["x_center"])
      y_center = float(box["y_center"])
      width = float(box["width"])
      height = float(box["height"])
      updated.update({
        "x_center": f"{x_center:.8f}",
        "y_center": f"{y_center:.8f}",
        "width": f"{width:.8f}",
        "height": f"{height:.8f}",
        "x0_px": str(int(round((x_center - width * 0.5) * image_width))),
        "y0_px": str(int(round((y_center - height * 0.5) * image_height))),
        "x1_px": str(int(round((x_center + width * 0.5) * image_width))),
        "y1_px": str(int(round((y_center + height * 0.5) * image_height))),
        "annotation_source": "human-makesense",
        "review_status": "reviewed",
      })
      updated_by_key[annotation_key(updated)] = updated

  merged = [
    updated_by_key.get(annotation_key(row), dict(row))
    for row in annotation_rows
  ]
  validate_annotations(merged, set(grouped))
  return merged


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent
  export_path = Path(args.export).expanduser().resolve()
  annotations_path = Path(args.annotations)
  if not annotations_path.is_absolute():
    annotations_path = (base_dir / annotations_path).resolve()
  source_exclusions_path = Path(args.source_exclusions)
  if not source_exclusions_path.is_absolute():
    source_exclusions_path = (base_dir / source_exclusions_path).resolve()

  rows = read_manifest(annotations_path)
  source_exclusions = read_source_exclusions(source_exclusions_path)
  export_files = read_export_files(export_path)
  if not export_files:
    raise ValueError(f"No per-image YOLO label files found in {export_path}")
  merged = merge_reviewed_export(
    rows,
    export_files,
    allow_partial=args.allow_partial,
    excluded_filenames=set(source_exclusions),
  )
  write_csv(annotations_path, merged, ANNOTATION_HEADERS)

  reviewed_images = len(export_files)
  print(f"Imported reviewed boxes for {reviewed_images} image(s).")
  print(f"Updated canonical manifest: {annotations_path}")
  print("Regenerate derived labels and QA previews with:")
  print("  python build_full_image_digit_dataset.py")


if __name__ == "__main__":
  main()
