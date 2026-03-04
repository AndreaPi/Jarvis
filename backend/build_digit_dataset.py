from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


@dataclass
class ExportRow:
  split: str
  filename: str
  strip_path: Path
  label_path: Path
  label: str
  roi_x: int
  roi_y: int
  roi_w: int
  roi_h: int


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Build digit OCR datasets (strip-level + per-cell) from ROI labels and readings CSV."
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
    help="Output dataset root."
  )
  parser.add_argument(
    "--cell-count",
    type=int,
    default=4,
    help="Number of cells (default: 4)."
  )
  parser.add_argument(
    "--cell-overlap",
    type=float,
    default=0.03,
    help="Horizontal overlap ratio per cell (default: 0.03)."
  )
  parser.add_argument(
    "--expand-x",
    type=float,
    default=0.0,
    help="Expand ROI horizontally by this ratio of ROI width on each side."
  )
  parser.add_argument(
    "--expand-y",
    type=float,
    default=0.0,
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


def split_cells(strip: Image.Image, count: int, overlap_ratio: float) -> list[tuple[int, int, int, int]]:
  width, height = strip.size
  cell_width = width / count
  overlap = cell_width * overlap_ratio
  boxes: list[tuple[int, int, int, int]] = []
  for index in range(count):
    x0 = int(round(clamp(cell_width * index - overlap, 0, width - 1)))
    x1 = int(round(clamp(cell_width * (index + 1) + overlap, x0 + 1, width)))
    boxes.append((x0, 0, x1, height))
  return boxes


def ensure_dirs(root: Path) -> None:
  (root / "strips").mkdir(parents=True, exist_ok=True)
  (root / "strip_labels").mkdir(parents=True, exist_ok=True)
  (root / "cells").mkdir(parents=True, exist_ok=True)
  (root / "qa_previews").mkdir(parents=True, exist_ok=True)
  (root / "manifests").mkdir(parents=True, exist_ok=True)
  for split in ("train", "val", "test"):
    (root / "strips" / split).mkdir(parents=True, exist_ok=True)
    (root / "strip_labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "qa_previews" / split).mkdir(parents=True, exist_ok=True)
    for digit in range(10):
      (root / "cells" / split / str(digit)).mkdir(parents=True, exist_ok=True)


def write_qa_preview(
  source: Image.Image,
  out_path: Path,
  roi_rect: tuple[int, int, int, int],
  cell_boxes: list[tuple[int, int, int, int]]
) -> None:
  preview = source.convert("RGB").copy()
  draw = ImageDraw.Draw(preview)
  x, y, w, h = roi_rect
  line_width = max(2, source.width // 350)
  draw.rectangle((x, y, x + w, y + h), outline=(33, 188, 255), width=line_width)
  for box in cell_boxes:
    cx0, cy0, cx1, cy1 = box
    draw.line((x + cx0, y + cy0, x + cx0, y + cy1), fill=(255, 204, 0), width=max(1, line_width - 1))
    draw.line((x + cx1, y + cy0, x + cx1, y + cy1), fill=(255, 204, 0), width=max(1, line_width - 1))
  out_path.parent.mkdir(parents=True, exist_ok=True)
  preview.save(out_path, quality=92)


def write_csv(path: Path, rows: list[dict[str, str]], headers: list[str]) -> None:
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)


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
  if args.cell_count <= 0:
    raise ValueError("--cell-count must be positive.")
  if args.cell_overlap < 0:
    raise ValueError("--cell-overlap must be >= 0.")

  value_map = read_value_map(csv_path)
  if not value_map:
    raise RuntimeError(f"No readings found in CSV: {csv_path}")

  if args.clean and out_dir.exists():
    shutil.rmtree(out_dir)
  ensure_dirs(out_dir)

  strip_manifest_rows: list[dict[str, str]] = []
  cell_manifest_rows: list[dict[str, str]] = []
  exported_rows: list[ExportRow] = []

  split_counts = {"train": 0, "val": 0, "test": 0}
  cell_counts_by_split = {"train": 0, "val": 0, "test": 0}
  cell_counts_by_digit = {str(digit): 0 for digit in range(10)}
  skipped: list[dict[str, str]] = []

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

      filename = image_path.name
      reading = value_map.get(filename)
      if not reading:
        skipped.append({"filename": filename, "reason": "missing-reading"})
        continue
      if len(reading) != args.cell_count or not reading.isdigit():
        skipped.append({"filename": filename, "reason": f"unsupported-reading:{reading}"})
        continue

      label_path = label_dir / f"{image_path.stem}.txt"
      if not label_path.exists():
        skipped.append({"filename": filename, "reason": "missing-roi-label"})
        continue

      x_center, y_center, width_norm, height_norm = load_roi_label(label_path)

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
        strip = source_rgb.crop((roi_x, roi_y, roi_x + roi_w, roi_y + roi_h))

        strip_path = out_dir / "strips" / split / f"{image_path.stem}.png"
        label_target = out_dir / "strip_labels" / split / f"{image_path.stem}.txt"
        strip.save(strip_path)
        label_target.write_text(f"{reading}\n", encoding="utf-8")

        strip_manifest_rows.append({
          "split": split,
          "filename": filename,
          "strip_path": str(strip_path.relative_to(out_dir)),
          "label_path": str(label_target.relative_to(out_dir)),
          "label": reading,
          "roi_x": str(roi_x),
          "roi_y": str(roi_y),
          "roi_w": str(roi_w),
          "roi_h": str(roi_h)
        })

        cell_boxes = split_cells(strip, args.cell_count, args.cell_overlap)
        for cell_index, cell_box in enumerate(cell_boxes):
          digit_label = reading[cell_index]
          cell = strip.crop(cell_box)
          cell_name = f"{image_path.stem}_c{cell_index}_{digit_label}.png"
          cell_path = out_dir / "cells" / split / digit_label / cell_name
          cell.save(cell_path)
          cell_manifest_rows.append({
            "split": split,
            "filename": filename,
            "cell_index": str(cell_index),
            "digit": digit_label,
            "cell_path": str(cell_path.relative_to(out_dir))
          })
          cell_counts_by_split[split] += 1
          cell_counts_by_digit[digit_label] += 1

        qa_path = out_dir / "qa_previews" / split / f"{image_path.stem}_qa.jpg"
        write_qa_preview(source_rgb, qa_path, (roi_x, roi_y, roi_w, roi_h), cell_boxes)

      exported_rows.append(
        ExportRow(
          split=split,
          filename=filename,
          strip_path=strip_path,
          label_path=label_target,
          label=reading,
          roi_x=roi_x,
          roi_y=roi_y,
          roi_w=roi_w,
          roi_h=roi_h
        )
      )
      split_counts[split] += 1

  write_csv(
    out_dir / "manifests" / "strips.csv",
    strip_manifest_rows,
    ["split", "filename", "strip_path", "label_path", "label", "roi_x", "roi_y", "roi_w", "roi_h"]
  )
  write_csv(
    out_dir / "manifests" / "cells.csv",
    cell_manifest_rows,
    ["split", "filename", "cell_index", "digit", "cell_path"]
  )
  write_csv(
    out_dir / "manifests" / "skipped.csv",
    skipped,
    ["filename", "reason"]
  )

  summary = {
    "rows_exported": len(exported_rows),
    "rows_skipped": len(skipped),
    "splits": split_counts,
    "cells_per_split": cell_counts_by_split,
    "cells_per_digit": cell_counts_by_digit,
    "cell_count": args.cell_count,
    "cell_overlap": args.cell_overlap,
    "expand_x": args.expand_x,
    "expand_y": args.expand_y,
    "csv": str(csv_path),
    "roi_dataset_dir": str(roi_dataset_dir),
    "out_dir": str(out_dir)
  }
  (out_dir / "manifests" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

  print(f"Digit dataset exported: {out_dir}")
  print(f"Rows: {summary['rows_exported']} (skipped={summary['rows_skipped']})")
  print(
    "Split rows: "
    f"train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}"
  )
  print(
    "Cells: "
    f"train={cell_counts_by_split['train']} "
    f"val={cell_counts_by_split['val']} "
    f"test={cell_counts_by_split['test']}"
  )
  print(f"Manifests: {out_dir / 'manifests'}")


if __name__ == "__main__":
  main()
