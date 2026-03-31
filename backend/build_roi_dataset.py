from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

VALID_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Build a YOLO dataset for digit_window ROI from assets + ROI manifest."
  )
  parser.add_argument(
    "--csv",
    default="../assets/meter_readings.csv",
    help="Path to CSV with filename,value rows (relative to backend/ by default)."
  )
  parser.add_argument(
    "--roi-json",
    required=True,
    help="Path to ROI JSON list (from browser extraction)."
  )
  parser.add_argument(
    "--assets-dir",
    default="../assets",
    help="Directory containing source images (relative to backend/ by default)."
  )
  parser.add_argument(
    "--out-dir",
    default="data/roi_dataset",
    help="Output dataset root directory (relative to backend/ by default)."
  )
  parser.add_argument(
    "--preview-dir",
    default="data/roi_dataset/previews",
    help="Directory for visual QA previews (relative to backend/ by default)."
  )
  parser.add_argument(
    "--splits-json",
    default="data/roi_dataset/splits.json",
    help="Path to persistent split assignment manifest (relative to backend/ by default)."
  )
  parser.add_argument(
    "--new-split",
    choices=VALID_SPLITS,
    default="train",
    help="Split assigned to newly added rows after the split manifest has been bootstrapped."
  )
  return parser.parse_args()


def resolve(base_dir: Path, value: str) -> Path:
  path = Path(value)
  if path.is_absolute():
    return path
  return (base_dir / path).resolve()


def read_rows(csv_path: Path) -> list[dict[str, str]]:
  rows: list[dict[str, str]] = []
  with csv_path.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      filename = (row.get("filename") or "").strip()
      value = (row.get("value") or "").strip()
      if filename and value:
        rows.append({"filename": filename, "value": value})
  return rows


def read_roi_map(roi_json_path: Path) -> dict[str, dict]:
  payload = json.loads(roi_json_path.read_text(encoding="utf-8"))
  roi_map: dict[str, dict] = {}
  for item in payload:
    filename = item.get("filename")
    rect_norm = item.get("rectNorm")
    if not filename or not isinstance(rect_norm, dict):
      continue
    roi_map[str(filename)] = rect_norm
  return roi_map


def clamp01(value: float) -> float:
  return max(0.0, min(1.0, value))


def normalize_yolo(rect_norm: dict) -> tuple[float, float, float, float]:
  x = float(rect_norm.get("x", 0))
  y = float(rect_norm.get("y", 0))
  width = float(rect_norm.get("width", 0))
  height = float(rect_norm.get("height", 0))

  x = clamp01(x)
  y = clamp01(y)
  width = clamp01(width)
  height = clamp01(height)

  if width <= 0 or height <= 0:
    raise ValueError("Invalid ROI width/height in manifest.")

  if x + width > 1:
    width = max(0.001, 1 - x)
  if y + height > 1:
    height = max(0.001, 1 - y)

  xc = clamp01(x + width * 0.5)
  yc = clamp01(y + height * 0.5)
  return xc, yc, width, height


def split_for_index(index: int, total: int) -> str:
  test_count = 1 if total >= 3 else 0
  val_count = 1 if total >= 4 else 0
  train_count = max(1, total - val_count - test_count)

  if index < train_count:
    return "train"
  if index < train_count + val_count:
    return "val"
  return "test"


def read_split_map(split_json_path: Path) -> dict[str, str]:
  payload = json.loads(split_json_path.read_text(encoding="utf-8"))

  if isinstance(payload, dict) and isinstance(payload.get("assignments"), dict):
    raw_map = payload["assignments"]
  elif isinstance(payload, dict):
    raw_map = payload
  else:
    raise ValueError(f"Invalid split manifest format: {split_json_path}")

  split_map: dict[str, str] = {}
  for filename, split in raw_map.items():
    if not isinstance(filename, str) or not isinstance(split, str):
      continue
    normalized = split.strip().lower()
    if normalized not in VALID_SPLITS:
      raise ValueError(f"Invalid split '{split}' for {filename} in {split_json_path}")
    split_map[filename] = normalized
  return split_map


def write_split_map(split_json_path: Path, split_map: dict[str, str]) -> None:
  split_json_path.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    "version": 1,
    "assignments": {filename: split_map[filename] for filename in sorted(split_map)},
  }
  split_json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def bootstrap_split_map(rows: list[dict[str, str]], out_dir: Path) -> dict[str, str]:
  filename_by_stem = {Path(row["filename"]).stem: row["filename"] for row in rows}
  split_map: dict[str, str] = {}

  for split in VALID_SPLITS:
    image_dir = out_dir / "images" / split
    if image_dir.is_dir():
      for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
        filename = image_path.name
        previous = split_map.get(filename)
        if previous and previous != split:
          raise ValueError(f"Conflicting split assignments for {filename}: {previous}, {split}")
        split_map[filename] = split

    label_dir = out_dir / "labels" / split
    if label_dir.is_dir():
      for label_path in sorted(label_dir.glob("*.txt")):
        filename = filename_by_stem.get(label_path.stem)
        if not filename:
          continue
        previous = split_map.get(filename)
        if previous and previous != split:
          raise ValueError(f"Conflicting split assignments for {filename}: {previous}, {split}")
        split_map[filename] = split

  return split_map


def resolve_split_map(
  rows: list[dict[str, str]],
  out_dir: Path,
  split_json_path: Path,
  new_split: str,
) -> tuple[dict[str, str], bool]:
  row_filenames = [row["filename"] for row in rows]
  known_filenames = set(row_filenames)
  bootstrapped = False

  if split_json_path.exists():
    split_map = read_split_map(split_json_path)
  else:
    split_map = bootstrap_split_map(rows, out_dir)
    if not split_map:
      split_map = {
        row["filename"]: split_for_index(index, len(rows))
        for index, row in enumerate(rows)
      }
    bootstrapped = True

  resolved = {
    filename: split
    for filename, split in split_map.items()
    if filename in known_filenames
  }

  for filename in row_filenames:
    if filename not in resolved:
      resolved[filename] = new_split

  return resolved, bootstrapped


def ensure_dataset_dirs(root: Path) -> None:
  for split in VALID_SPLITS:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def prune_stale_outputs(
  out_dir: Path,
  preview_dir: Path,
  desired_split_map: dict[str, str],
) -> None:
  desired_by_split = {
    split: {filename for filename, assigned in desired_split_map.items() if assigned == split}
    for split in VALID_SPLITS
  }
  desired_stems_by_split = {
    split: {Path(filename).stem for filename in desired_by_split[split]}
    for split in VALID_SPLITS
  }
  desired_preview_names = {f"{Path(filename).stem}_bbox.jpg" for filename in desired_split_map}

  for split in VALID_SPLITS:
    image_dir = out_dir / "images" / split
    if image_dir.is_dir():
      for image_path in image_dir.iterdir():
        if image_path.is_file() and image_path.name not in desired_by_split[split]:
          image_path.unlink()

    label_dir = out_dir / "labels" / split
    if label_dir.is_dir():
      for label_path in label_dir.glob("*.txt"):
        if label_path.stem not in desired_stems_by_split[split]:
          label_path.unlink()

  if preview_dir.is_dir():
    for preview_path in preview_dir.glob("*_bbox.jpg"):
      if preview_path.name not in desired_preview_names:
        preview_path.unlink()


def write_preview(preview_path: Path, image_path: Path, rect_norm: dict) -> None:
  try:
    from PIL import Image, ImageDraw
  except ImportError:
    return

  with Image.open(image_path) as image:
    width, height = image.size
    x = clamp01(float(rect_norm.get("x", 0))) * width
    y = clamp01(float(rect_norm.get("y", 0))) * height
    w = clamp01(float(rect_norm.get("width", 0))) * width
    h = clamp01(float(rect_norm.get("height", 0))) * height
    draw = ImageDraw.Draw(image)
    draw.rectangle((x, y, x + w, y + h), outline=(0, 255, 255), width=max(2, width // 350))
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(preview_path)


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent

  csv_path = resolve(base_dir, args.csv)
  roi_json_path = resolve(base_dir, args.roi_json)
  assets_dir = resolve(base_dir, args.assets_dir)
  out_dir = resolve(base_dir, args.out_dir)
  preview_dir = resolve(base_dir, args.preview_dir)
  split_json_path = resolve(base_dir, args.splits_json)

  if not csv_path.exists():
    raise FileNotFoundError(f"CSV not found: {csv_path}")
  if not roi_json_path.exists():
    raise FileNotFoundError(f"ROI manifest not found: {roi_json_path}")
  if not assets_dir.exists():
    raise FileNotFoundError(f"Assets dir not found: {assets_dir}")

  rows = read_rows(csv_path)
  if not rows:
    raise RuntimeError(f"No dataset rows found in CSV: {csv_path}")

  roi_map = read_roi_map(roi_json_path)
  split_map, bootstrapped = resolve_split_map(rows, out_dir, split_json_path, args.new_split)

  planned = []
  for row in rows:
    filename = row["filename"]
    source_image = assets_dir / filename
    if not source_image.exists():
      raise FileNotFoundError(f"Missing source image: {source_image}")
    if filename not in roi_map:
      raise KeyError(f"Missing ROI entry for {filename} in {roi_json_path}")

    split = split_map[filename]
    xc, yc, width, height = normalize_yolo(roi_map[filename])
    planned.append((filename, split, source_image, xc, yc, width, height, roi_map[filename]))

  ensure_dataset_dirs(out_dir)
  preview_dir.mkdir(parents=True, exist_ok=True)
  prune_stale_outputs(out_dir, preview_dir, split_map)

  created = []
  for filename, split, source_image, xc, yc, width, height, rect_norm in planned:
    target_image = out_dir / "images" / split / filename
    target_label = out_dir / "labels" / split / f"{Path(filename).stem}.txt"

    shutil.copy2(source_image, target_image)
    target_label.write_text(f"0 {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}\n", encoding="utf-8")

    preview_path = preview_dir / f"{Path(filename).stem}_bbox.jpg"
    write_preview(preview_path, source_image, rect_norm)
    created.append((filename, split, target_image, target_label, preview_path))

  manifest_target = out_dir / "roi_boxes.json"
  shutil.copy2(roi_json_path, manifest_target)
  write_split_map(split_json_path, split_map)

  split_counts = {"train": 0, "val": 0, "test": 0}
  for _, split, *_ in created:
    split_counts[split] += 1

  print(f"Built ROI dataset at: {out_dir}")
  print(f"Rows: {len(created)} (train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']})")
  print(f"ROI manifest copied to: {manifest_target}")
  print(f"Split manifest: {split_json_path}{' (bootstrapped)' if bootstrapped else ''}")
  print(f"Preview images: {preview_dir}")


if __name__ == "__main__":
  main()
