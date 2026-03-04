from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Split digit-window crops into equal sections along each window's major dimension."
  )
  parser.add_argument(
    "--dataset-dir",
    default="data/digit_dataset",
    help="Dataset root containing manifests/windows.csv and windows/<split>."
  )
  parser.add_argument(
    "--windows-manifest",
    default="manifests/windows.csv",
    help="Path to windows manifest, relative to --dataset-dir unless absolute."
  )
  parser.add_argument(
    "--section-count",
    type=int,
    default=4,
    help="Number of equal sections per window."
  )
  parser.add_argument(
    "--direction-overrides",
    default="manifests/direction_overrides.csv",
    help="Optional CSV with filename,flip180 (0/1) to override reading-direction flips."
  )
  parser.add_argument(
    "--clean",
    action="store_true",
    help="Delete existing sections and canonical windows output before writing."
  )
  return parser.parse_args()


def resolve(base: Path, value: str) -> Path:
  path = Path(value)
  if path.is_absolute():
    return path
  return (base / path).resolve()


def ensure_dirs(dataset_dir: Path) -> None:
  (dataset_dir / "windows_canonical").mkdir(parents=True, exist_ok=True)
  (dataset_dir / "sections").mkdir(parents=True, exist_ok=True)
  (dataset_dir / "manifests").mkdir(parents=True, exist_ok=True)
  for split in ("train", "val", "test"):
    (dataset_dir / "windows_canonical" / split).mkdir(parents=True, exist_ok=True)
    (dataset_dir / "sections" / split).mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, str]], headers: list[str]) -> None:
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)


def section_bounds(length: int, count: int, index: int) -> tuple[int, int]:
  start = int(round(index * length / count))
  end = int(round((index + 1) * length / count))
  if end <= start:
    end = min(length, start + 1)
  return start, end


def major_axis(width: int, height: int) -> str:
  return "x" if width >= height else "y"


def rotate_right_angle(image: Image.Image, angle: int) -> Image.Image:
  normalized = angle % 360
  if normalized == 0:
    return image.copy()
  return image.rotate(normalized, expand=True)


def ink_lowerness_score(image: Image.Image) -> float:
  gray = image.convert("L")
  width, height = gray.size
  pixels = gray.load()
  denom_y = max(1, height - 1)
  total = 0.0
  weighted = 0.0
  for y in range(height):
    y_weight = y / denom_y
    for x in range(width):
      lum = pixels[x, y]
      ink = max(0, 220 - int(lum))
      if ink <= 0:
        continue
      total += ink
      weighted += ink * y_weight
  if total <= 0:
    return 0.5
  return weighted / total


def parse_bool_token(raw: str) -> bool:
  value = (raw or "").strip().lower()
  return value in {"1", "true", "yes", "y", "flip"}


def load_direction_overrides(path: Path) -> dict[str, bool]:
  if not path.exists():
    return {}
  mapping: dict[str, bool] = {}
  with path.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      filename = (row.get("filename") or "").strip()
      if not filename:
        continue
      flip_raw = row.get("flip180")
      if flip_raw is None:
        flip_raw = row.get("flip")
      mapping[filename] = parse_bool_token(flip_raw or "")
  return mapping


def canonicalize_window_orientation(
  source: Image.Image,
  filename: str,
  direction_overrides: dict[str, bool]
) -> tuple[Image.Image, dict[str, str]]:
  source_major_axis = major_axis(source.width, source.height)
  # For vertical windows, rotate clockwise to horizontal first.
  axis_rotation = 0 if source_major_axis == "x" else 270

  primary = rotate_right_angle(source, axis_rotation)
  flipped = rotate_right_angle(primary, 180)

  primary_score = ink_lowerness_score(primary)
  flipped_score = ink_lowerness_score(flipped)

  direction_flip = bool(direction_overrides.get(filename, False))
  direction_source = "override" if filename in direction_overrides else "default"
  if direction_flip:
    canonical = flipped
    applied_rotation = (axis_rotation + 180) % 360
  else:
    canonical = primary
    applied_rotation = axis_rotation

  metadata = {
    "source_major_axis": source_major_axis,
    "canonical_major_axis": major_axis(canonical.width, canonical.height),
    "axis_rotation": str(axis_rotation),
    "direction_flip": "1" if direction_flip else "0",
    "direction_source": direction_source,
    "applied_rotation": str(applied_rotation),
    "primary_lowerness": f"{primary_score:.6f}",
    "flipped_lowerness": f"{flipped_score:.6f}"
  }
  return canonical, metadata


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent

  dataset_dir = resolve(base_dir, args.dataset_dir)
  manifest_path = resolve(dataset_dir, args.windows_manifest)
  direction_overrides_path = resolve(dataset_dir, args.direction_overrides)

  if args.section_count <= 0:
    raise ValueError("--section-count must be positive.")
  if not dataset_dir.exists():
    raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
  if not manifest_path.exists():
    raise FileNotFoundError(f"Windows manifest not found: {manifest_path}")

  sections_dir = dataset_dir / "sections"
  canonical_dir = dataset_dir / "windows_canonical"
  if args.clean and sections_dir.exists():
    shutil.rmtree(sections_dir)
  if args.clean and canonical_dir.exists():
    shutil.rmtree(canonical_dir)
  ensure_dirs(dataset_dir)
  direction_overrides = load_direction_overrides(direction_overrides_path)

  canonical_rows: list[dict[str, str]] = []
  rows: list[dict[str, str]] = []
  skipped: list[dict[str, str]] = []
  split_counts = {"train": 0, "val": 0, "test": 0}
  source_axis_counts = {"x": 0, "y": 0}
  canonical_axis_counts = {"x": 0, "y": 0}
  rotation_counts = {"0": 0, "90": 0, "180": 0, "270": 0}

  with manifest_path.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      split = (row.get("split") or "").strip()
      filename = (row.get("filename") or "").strip()
      reading = (row.get("reading") or "").strip()
      window_rel = (row.get("window_path") or "").strip()
      if split not in {"train", "val", "test"}:
        skipped.append({"split": split, "filename": filename, "reason": "invalid-split"})
        continue
      if not window_rel:
        skipped.append({"split": split, "filename": filename, "reason": "missing-window-path"})
        continue

      window_path = dataset_dir / window_rel
      if not window_path.exists():
        skipped.append({"split": split, "filename": filename, "reason": "missing-window-file"})
        continue

      with Image.open(window_path) as image:
        source = image.convert("RGB")
        canonical, orientation_meta = canonicalize_window_orientation(
          source,
          filename,
          direction_overrides
        )
        width, height = canonical.size
        source_major_axis = orientation_meta["source_major_axis"]
        canonical_major_axis = orientation_meta["canonical_major_axis"]
        source_axis_counts[source_major_axis] += 1
        canonical_axis_counts[canonical_major_axis] += 1
        applied_rotation = orientation_meta["applied_rotation"]
        if applied_rotation in rotation_counts:
          rotation_counts[applied_rotation] += 1

        canonical_path = dataset_dir / "windows_canonical" / split / f"{Path(window_path).stem}.png"
        canonical.save(canonical_path)
        canonical_rel = str(canonical_path.relative_to(dataset_dir))

        canonical_rows.append({
          "split": split,
          "filename": filename,
          "reading": reading,
          "source_window_path": window_rel,
          "canonical_window_path": canonical_rel,
          "source_width": str(source.width),
          "source_height": str(source.height),
          "canonical_width": str(width),
          "canonical_height": str(height),
          "source_major_axis": source_major_axis,
          "canonical_major_axis": canonical_major_axis,
          "axis_rotation": orientation_meta["axis_rotation"],
          "direction_flip": orientation_meta["direction_flip"],
          "direction_source": orientation_meta["direction_source"],
          "applied_rotation": applied_rotation,
          "primary_lowerness": orientation_meta["primary_lowerness"],
          "flipped_lowerness": orientation_meta["flipped_lowerness"]
        })

        for section_index in range(args.section_count):
          x0, x1 = section_bounds(width, args.section_count, section_index)
          y0, y1 = 0, height

          section = canonical.crop((x0, y0, x1, y1))
          section_name = f"{Path(window_path).stem}_s{section_index}.png"
          section_path = dataset_dir / "sections" / split / section_name
          section.save(section_path)

          rows.append({
            "split": split,
            "filename": filename,
            "reading": reading,
            "source_window_path": window_rel,
            "canonical_window_path": canonical_rel,
            "section_index": str(section_index),
            "major_axis": canonical_major_axis,
            "applied_rotation": applied_rotation,
            "x0": str(x0),
            "y0": str(y0),
            "x1": str(x1),
            "y1": str(y1),
            "section_path": str(section_path.relative_to(dataset_dir))
          })

      split_counts[split] += 1

  write_csv(
    dataset_dir / "manifests" / "canonical_windows.csv",
    canonical_rows,
    [
      "split",
      "filename",
      "reading",
      "source_window_path",
      "canonical_window_path",
      "source_width",
      "source_height",
      "canonical_width",
      "canonical_height",
      "source_major_axis",
      "canonical_major_axis",
      "axis_rotation",
      "direction_flip",
      "direction_source",
      "applied_rotation",
      "primary_lowerness",
      "flipped_lowerness"
    ]
  )
  write_csv(
    dataset_dir / "manifests" / "sections.csv",
    rows,
    [
      "split",
      "filename",
      "reading",
      "source_window_path",
      "canonical_window_path",
      "section_index",
      "major_axis",
      "applied_rotation",
      "x0",
      "y0",
      "x1",
      "y1",
      "section_path"
    ]
  )
  write_csv(
    dataset_dir / "manifests" / "sections_skipped.csv",
    skipped,
    ["split", "filename", "reason"]
  )

  summary = {
    "windows_processed": sum(split_counts.values()),
    "sections_exported": len(rows),
    "windows_skipped": len(skipped),
    "section_count": args.section_count,
    "splits": split_counts,
    "source_axis_counts": source_axis_counts,
    "canonical_axis_counts": canonical_axis_counts,
    "applied_rotation_counts": rotation_counts,
    "dataset_dir": str(dataset_dir),
    "windows_manifest": str(manifest_path),
    "direction_overrides": str(direction_overrides_path),
    "direction_override_count": len(direction_overrides)
  }
  (dataset_dir / "manifests" / "sections_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
  )

  print(f"Sections exported under: {dataset_dir / 'sections'}")
  print(f"Windows processed: {summary['windows_processed']} (skipped={summary['windows_skipped']})")
  print(f"Sections exported: {summary['sections_exported']}")
  print(
    "Split windows: "
    f"train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}"
  )
  print(f"Source major axis counts: x={source_axis_counts['x']} y={source_axis_counts['y']}")
  print(f"Canonical major axis counts: x={canonical_axis_counts['x']} y={canonical_axis_counts['y']}")
  print(
    "Applied rotations: "
    f"0={rotation_counts['0']} 90={rotation_counts['90']} "
    f"180={rotation_counts['180']} 270={rotation_counts['270']}"
  )
  print(f"Manifests: {dataset_dir / 'manifests'}")


if __name__ == "__main__":
  main()
