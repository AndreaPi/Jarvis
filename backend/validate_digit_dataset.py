from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Validate digit dataset manifests, files, and QA preview coverage."
  )
  parser.add_argument(
    "--dataset-root",
    default="data/digit_dataset",
    help="Digit dataset root."
  )
  parser.add_argument(
    "--cell-count",
    type=int,
    default=4,
    help="Expected number of cell crops per strip row."
  )
  return parser.parse_args()


def resolve(base_dir: Path, value: str) -> Path:
  path = Path(value)
  if path.is_absolute():
    return path
  return (base_dir / path).resolve()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
  rows: list[dict[str, str]] = []
  with path.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      rows.append({key: (value or "").strip() for key, value in row.items()})
  return rows


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent
  dataset_root = resolve(base_dir, args.dataset_root)
  manifests_dir = dataset_root / "manifests"
  strips_csv = manifests_dir / "strips.csv"
  cells_csv = manifests_dir / "cells.csv"

  if args.cell_count <= 0:
    raise ValueError("--cell-count must be positive.")
  if not dataset_root.exists():
    raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
  if not strips_csv.exists():
    raise FileNotFoundError(f"Strips manifest not found: {strips_csv}")
  if not cells_csv.exists():
    raise FileNotFoundError(f"Cells manifest not found: {cells_csv}")

  strip_rows = load_csv_rows(strips_csv)
  cell_rows = load_csv_rows(cells_csv)
  if not strip_rows:
    raise RuntimeError(f"No rows found in strips manifest: {strips_csv}")
  if not cell_rows:
    raise RuntimeError(f"No rows found in cells manifest: {cells_csv}")

  errors: list[str] = []
  warnings: list[str] = []
  cells_by_strip: dict[tuple[str, str], list[dict[str, str]]] = {}
  split_counts = {"train": 0, "val": 0, "test": 0}
  digit_counts = {str(digit): 0 for digit in range(10)}

  for row in cell_rows:
    split = row.get("split", "")
    filename = row.get("filename", "")
    digit = row.get("digit", "")
    cell_index_text = row.get("cell_index", "")
    cell_path_rel = row.get("cell_path", "")

    if split not in split_counts:
      errors.append(f"Unexpected split in cells.csv: {split}")
      continue
    if digit not in digit_counts:
      errors.append(f"Unexpected digit in cells.csv: {digit}")
      continue
    split_counts[split] += 1
    digit_counts[digit] += 1

    try:
      cell_index = int(cell_index_text)
    except ValueError:
      errors.append(f"Invalid cell_index for {filename}: {cell_index_text}")
      continue
    if cell_index < 0 or cell_index >= args.cell_count:
      errors.append(f"Out-of-range cell_index for {filename}: {cell_index}")

    key = (split, filename)
    cells_by_strip.setdefault(key, []).append(row)

    cell_path = dataset_root / cell_path_rel
    if not cell_path.exists():
      errors.append(f"Missing cell file: {cell_path}")
    else:
      # Enforce file routing by split/digit to avoid silent mislabeled moves.
      expected_parent = dataset_root / "cells" / split / digit
      if cell_path.parent != expected_parent:
        errors.append(
          f"Cell path not in expected split/digit folder: {cell_path} "
          f"(expected parent {expected_parent})"
        )

  for row in strip_rows:
    split = row.get("split", "")
    filename = row.get("filename", "")
    strip_path_rel = row.get("strip_path", "")
    label_path_rel = row.get("label_path", "")

    if split not in {"train", "val", "test"}:
      errors.append(f"Unexpected split in strips.csv: {split}")
      continue
    stem = Path(filename).stem
    key = (split, filename)
    matching_cells = cells_by_strip.get(key, [])
    if len(matching_cells) != args.cell_count:
      errors.append(
        f"Expected {args.cell_count} cells for {split}/{filename}, found {len(matching_cells)}."
      )
    else:
      index_values = sorted(
        int(cell_row["cell_index"])
        for cell_row in matching_cells
        if cell_row.get("cell_index", "").isdigit()
      )
      if index_values != list(range(args.cell_count)):
        errors.append(
          f"Unexpected cell_index set for {split}/{filename}: {index_values}"
        )

    strip_path = dataset_root / strip_path_rel
    label_path = dataset_root / label_path_rel
    qa_preview_path = dataset_root / "qa_previews" / split / f"{stem}_qa.jpg"
    if not strip_path.exists():
      errors.append(f"Missing strip file: {strip_path}")
    if not label_path.exists():
      errors.append(f"Missing strip label file: {label_path}")
    if not qa_preview_path.exists():
      errors.append(f"Missing QA preview image: {qa_preview_path}")

  strip_keys = {(row.get("split", ""), row.get("filename", "")) for row in strip_rows}
  for key in cells_by_strip.keys():
    if key not in strip_keys:
      warnings.append(f"Cell rows exist without strip row: split={key[0]} filename={key[1]}")

  print(f"Dataset root: {dataset_root}")
  print(f"Strip rows: {len(strip_rows)}")
  print(f"Cell rows: {len(cell_rows)}")
  print(
    "Cell rows by split: "
    f"train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}"
  )
  print(
    "Cell rows by digit: "
    + ", ".join([f"{digit}:{digit_counts[digit]}" for digit in map(str, range(10))])
  )

  if warnings:
    print("Warnings:")
    for message in warnings:
      print(f"  - {message}")

  if errors:
    print("Errors:")
    for message in errors:
      print(f"  - {message}")
    raise SystemExit(1)

  print("Validation passed.")


if __name__ == "__main__":
  main()
