from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Assign digit labels to canonical window sections from reading strings."
  )
  parser.add_argument(
    "--dataset-dir",
    default="data/digit_dataset",
    help="Dataset root containing manifests/sections.csv and sections/<split>."
  )
  parser.add_argument(
    "--sections-manifest",
    default="manifests/sections.csv",
    help="Path to sections manifest, relative to --dataset-dir unless absolute."
  )
  parser.add_argument(
    "--section-count",
    type=int,
    default=4,
    help="Expected section count/read length."
  )
  parser.add_argument(
    "--clean",
    action="store_true",
    help="Delete existing labeled section output before writing."
  )
  return parser.parse_args()


def resolve(base: Path, value: str) -> Path:
  path = Path(value)
  if path.is_absolute():
    return path
  return (base / path).resolve()


def ensure_dirs(dataset_dir: Path) -> None:
  (dataset_dir / "sections_labeled").mkdir(parents=True, exist_ok=True)
  (dataset_dir / "manifests").mkdir(parents=True, exist_ok=True)
  for split in ("train", "val", "test"):
    for digit in range(10):
      (dataset_dir / "sections_labeled" / split / str(digit)).mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, str]], headers: list[str]) -> None:
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)


def parse_int(raw: str) -> int | None:
  try:
    return int(raw)
  except (TypeError, ValueError):
    return None


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent

  dataset_dir = resolve(base_dir, args.dataset_dir)
  sections_manifest_path = resolve(dataset_dir, args.sections_manifest)

  if args.section_count <= 0:
    raise ValueError("--section-count must be positive.")
  if not dataset_dir.exists():
    raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
  if not sections_manifest_path.exists():
    raise FileNotFoundError(f"Sections manifest not found: {sections_manifest_path}")

  labeled_root = dataset_dir / "sections_labeled"
  if args.clean and labeled_root.exists():
    shutil.rmtree(labeled_root)
  ensure_dirs(dataset_dir)

  labels_rows: list[dict[str, str]] = []
  skipped_rows: list[dict[str, str]] = []
  split_counts = {"train": 0, "val": 0, "test": 0}
  digit_counts = {str(d): 0 for d in range(10)}

  with sections_manifest_path.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      split = (row.get("split") or "").strip()
      filename = (row.get("filename") or "").strip()
      reading = (row.get("reading") or "").strip()
      section_index_raw = (row.get("section_index") or "").strip()
      section_rel = (row.get("section_path") or "").strip()
      canonical_window_path = (row.get("canonical_window_path") or "").strip()
      source_window_path = (row.get("source_window_path") or "").strip()

      if split not in {"train", "val", "test"}:
        skipped_rows.append({
          "split": split,
          "filename": filename,
          "section_index": section_index_raw,
          "section_path": section_rel,
          "reason": "invalid-split"
        })
        continue

      section_index = parse_int(section_index_raw)
      if section_index is None:
        skipped_rows.append({
          "split": split,
          "filename": filename,
          "section_index": section_index_raw,
          "section_path": section_rel,
          "reason": "invalid-section-index"
        })
        continue

      if section_index < 0 or section_index >= args.section_count:
        skipped_rows.append({
          "split": split,
          "filename": filename,
          "section_index": section_index_raw,
          "section_path": section_rel,
          "reason": "section-index-out-of-range"
        })
        continue

      if len(reading) != args.section_count or not reading.isdigit():
        skipped_rows.append({
          "split": split,
          "filename": filename,
          "section_index": section_index_raw,
          "section_path": section_rel,
          "reason": "invalid-reading"
        })
        continue

      section_path = dataset_dir / section_rel
      if not section_path.exists():
        skipped_rows.append({
          "split": split,
          "filename": filename,
          "section_index": section_index_raw,
          "section_path": section_rel,
          "reason": "missing-section-file"
        })
        continue

      digit = reading[section_index]
      labeled_name = f"{section_path.stem}_d{digit}{section_path.suffix.lower() or '.png'}"
      labeled_path = labeled_root / split / digit / labeled_name
      shutil.copy2(section_path, labeled_path)

      labels_rows.append({
        "split": split,
        "filename": filename,
        "reading": reading,
        "section_index": str(section_index),
        "digit": digit,
        "section_path": section_rel,
        "labeled_path": str(labeled_path.relative_to(dataset_dir)),
        "source_window_path": source_window_path,
        "canonical_window_path": canonical_window_path
      })
      split_counts[split] += 1
      digit_counts[digit] += 1

  write_csv(
    dataset_dir / "manifests" / "section_labels.csv",
    labels_rows,
    [
      "split",
      "filename",
      "reading",
      "section_index",
      "digit",
      "section_path",
      "labeled_path",
      "source_window_path",
      "canonical_window_path"
    ]
  )
  write_csv(
    dataset_dir / "manifests" / "section_labels_skipped.csv",
    skipped_rows,
    ["split", "filename", "section_index", "section_path", "reason"]
  )

  summary = {
    "labels_exported": len(labels_rows),
    "labels_skipped": len(skipped_rows),
    "section_count": args.section_count,
    "split_counts": split_counts,
    "digit_counts": digit_counts,
    "dataset_dir": str(dataset_dir),
    "sections_manifest": str(sections_manifest_path)
  }
  (dataset_dir / "manifests" / "section_labels_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
  )

  print(f"Labeled sections exported under: {labeled_root}")
  print(f"Labels exported: {summary['labels_exported']} (skipped={summary['labels_skipped']})")
  print(
    "Split labels: "
    f"train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}"
  )
  print(
    "Digit counts: "
    + ", ".join([f"{digit}={digit_counts[digit]}" for digit in sorted(digit_counts.keys())])
  )
  print(f"Manifests: {dataset_dir / 'manifests'}")


if __name__ == "__main__":
  main()
