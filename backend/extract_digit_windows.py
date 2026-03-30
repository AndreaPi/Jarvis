from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Extract ROI digit windows as standalone images from the ROI dataset."
  )
  parser.add_argument(
    "--csv",
    default="../assets/meter_readings.csv",
    help="CSV with filename,value rows (default resolves from backend/)."
  )
  parser.add_argument(
    "--roi-dataset-dir",
    default="data/roi_dataset",
    help="ROI dataset root containing images/<split> and labels/<split>."
  )
  parser.add_argument(
    "--out-dir",
    default="data/digit_dataset",
    help="Output dataset root for extracted windows."
  )
  parser.add_argument(
    "--expand-x",
    type=float,
    default=0.26,
    help="Expand ROI horizontally by this ratio of ROI width on each side."
  )
  parser.add_argument(
    "--expand-y",
    type=float,
    default=0.16,
    help="Expand ROI vertically by this ratio of ROI height on each side."
  )
  parser.add_argument(
    "--clean",
    action="store_true",
    help="Delete output directory before writing."
  )
  return parser.parse_args()


def resolve(base_dir: Path, value: str) -> Path:
  path = Path(value)
  if path.is_absolute():
    return path
  return (base_dir / path).resolve()


def clamp(value: float, low: float, high: float) -> float:
  return max(low, min(high, value))


def read_value_map(csv_path: Path) -> dict[str, str]:
  mapping: dict[str, str] = {}
  with csv_path.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      filename = (row.get("filename") or "").strip()
      value = (row.get("value") or "").strip()
      if filename and value:
        mapping[filename] = value
  return mapping


def load_roi_label(label_path: Path) -> tuple[float, float, float, float]:
  raw = label_path.read_text(encoding="utf-8").strip().splitlines()
  for line in raw:
    parts = line.strip().split()
    if len(parts) != 5:
      continue
    _, x_center, y_center, width, height = parts
    return float(x_center), float(y_center), float(width), float(height)
  raise ValueError(f"No valid YOLO row in label file: {label_path}")


def yolo_to_pixel_rect(
  image_width: int,
  image_height: int,
  x_center: float,
  y_center: float,
  width: float,
  height: float,
  expand_x: float,
  expand_y: float
) -> tuple[int, int, int, int]:
  x = (x_center - width * 0.5) * image_width
  y = (y_center - height * 0.5) * image_height
  w = width * image_width
  h = height * image_height

  x -= w * expand_x
  y -= h * expand_y
  w *= (1 + 2 * expand_x)
  h *= (1 + 2 * expand_y)

  x0 = int(round(clamp(x, 0, image_width - 1)))
  y0 = int(round(clamp(y, 0, image_height - 1)))
  x1 = int(round(clamp(x + w, x0 + 1, image_width)))
  y1 = int(round(clamp(y + h, y0 + 1, image_height)))
  return x0, y0, x1 - x0, y1 - y0


def write_csv(path: Path, rows: list[dict[str, str]], headers: list[str]) -> None:
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)


def ensure_dirs(out_dir: Path) -> None:
  (out_dir / "windows").mkdir(parents=True, exist_ok=True)
  (out_dir / "manifests").mkdir(parents=True, exist_ok=True)
  for split in ("train", "val", "test"):
    (out_dir / "windows" / split).mkdir(parents=True, exist_ok=True)


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent

  csv_path = resolve(base_dir, args.csv)
  roi_dataset_dir = resolve(base_dir, args.roi_dataset_dir)
  out_dir = resolve(base_dir, args.out_dir)

  if not csv_path.exists():
    raise FileNotFoundError(f"CSV not found: {csv_path}")
  if not roi_dataset_dir.exists():
    raise FileNotFoundError(f"ROI dataset dir not found: {roi_dataset_dir}")

  value_map = read_value_map(csv_path)

  if args.clean and out_dir.exists():
    shutil.rmtree(out_dir)
  ensure_dirs(out_dir)

  rows: list[dict[str, str]] = []
  skipped: list[dict[str, str]] = []
  split_counts = {"train": 0, "val": 0, "test": 0}
  missing_reading_count = 0

  for split in ("train", "val", "test"):
    image_dir = roi_dataset_dir / "images" / split
    label_dir = roi_dataset_dir / "labels" / split
    if not image_dir.exists():
      continue

    for image_path in sorted(image_dir.iterdir()):
      if not image_path.is_file():
        continue
      if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

      label_path = label_dir / f"{image_path.stem}.txt"
      if not label_path.exists():
        skipped.append({"split": split, "filename": image_path.name, "reason": "missing-roi-label"})
        continue

      try:
        x_center, y_center, width_norm, height_norm = load_roi_label(label_path)
      except ValueError:
        skipped.append({"split": split, "filename": image_path.name, "reason": "invalid-roi-label"})
        continue

      with Image.open(image_path) as source:
        source_rgb = source.convert("RGB")
        roi_x, roi_y, roi_w, roi_h = yolo_to_pixel_rect(
          image_width=source_rgb.width,
          image_height=source_rgb.height,
          x_center=x_center,
          y_center=y_center,
          width=width_norm,
          height=height_norm,
          expand_x=args.expand_x,
          expand_y=args.expand_y
        )

        window = source_rgb.crop((roi_x, roi_y, roi_x + roi_w, roi_y + roi_h))
        window_path = out_dir / "windows" / split / f"{image_path.stem}.png"
        window.save(window_path)

      reading = (value_map.get(image_path.name) or "").strip()
      if not reading:
        missing_reading_count += 1

      rows.append({
        "split": split,
        "filename": image_path.name,
        "window_path": str(window_path.relative_to(out_dir)),
        "reading": reading,
        "roi_x": str(roi_x),
        "roi_y": str(roi_y),
        "roi_w": str(roi_w),
        "roi_h": str(roi_h)
      })
      split_counts[split] += 1

  write_csv(
    out_dir / "manifests" / "windows.csv",
    rows,
    ["split", "filename", "window_path", "reading", "roi_x", "roi_y", "roi_w", "roi_h"]
  )
  write_csv(
    out_dir / "manifests" / "skipped.csv",
    skipped,
    ["split", "filename", "reason"]
  )

  summary = {
    "rows_exported": len(rows),
    "rows_skipped": len(skipped),
    "splits": split_counts,
    "missing_reading_count": missing_reading_count,
    "expand_x": args.expand_x,
    "expand_y": args.expand_y,
    "csv": str(csv_path),
    "roi_dataset_dir": str(roi_dataset_dir),
    "out_dir": str(out_dir)
  }
  (out_dir / "manifests" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

  print(f"Digit windows exported: {out_dir}")
  print(f"Rows: {summary['rows_exported']} (skipped={summary['rows_skipped']})")
  print(
    "Split rows: "
    f"train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}"
  )
  print(f"Missing readings: {missing_reading_count}")
  print(f"Manifests: {out_dir / 'manifests'}")


if __name__ == "__main__":
  main()
