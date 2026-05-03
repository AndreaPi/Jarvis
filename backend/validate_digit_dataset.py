from __future__ import annotations

import argparse
import csv
from pathlib import Path

VALID_SPLITS = {"train", "val", "test"}
DIGITS = {str(digit) for digit in range(10)}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Validate current digit dataset manifests, files, and split consistency."
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
    help="Expected number of sections/cells per digit window."
  )
  parser.add_argument(
    "--skip-synthetic",
    action="store_true",
    help="Skip optional validation of sections_synthetic/manifests/synthetic_cells.csv."
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


def require_columns(path: Path, rows: list[dict[str, str]], columns: set[str]) -> None:
  if not rows:
    raise RuntimeError(f"No rows found in manifest: {path}")
  missing = columns.difference(rows[0].keys())
  if missing:
    raise RuntimeError(f"Missing columns in {path}: {', '.join(sorted(missing))}")


def parse_int(value: str, context: str, errors: list[str]) -> int | None:
  try:
    return int(value)
  except ValueError:
    errors.append(f"Invalid integer for {context}: {value}")
    return None


def validate_reading(reading: str, filename: str, errors: list[str], cell_count: int) -> None:
  if len(reading) != cell_count or not reading.isdigit():
    errors.append(
      f"Expected {cell_count}-digit numeric reading for {filename}, found {reading!r}"
    )


def add_split_count(split_counts: dict[str, int], split: str, context: str, errors: list[str]) -> bool:
  if split not in split_counts:
    errors.append(f"Unexpected split in {context}: {split}")
    return False
  split_counts[split] += 1
  return True


def ensure_manifest_file(dataset_root: Path, rel_path: str, context: str, errors: list[str]) -> Path:
  path = dataset_root / rel_path
  if not rel_path:
    errors.append(f"Missing path value for {context}")
  elif not path.exists():
    errors.append(f"Missing file for {context}: {path}")
  return path


def window_key(row: dict[str, str]) -> tuple[str, str]:
  return (row.get("split", ""), row.get("filename", ""))


def section_key(row: dict[str, str]) -> tuple[str, str, str]:
  return (row.get("split", ""), row.get("filename", ""), row.get("section_index", ""))


def validate_current_dataset(dataset_root: Path, cell_count: int, skip_synthetic: bool) -> None:
  manifests_dir = dataset_root / "manifests"
  windows_csv = manifests_dir / "windows.csv"
  canonical_csv = manifests_dir / "canonical_windows.csv"
  sections_csv = manifests_dir / "sections.csv"
  labels_csv = manifests_dir / "section_labels.csv"

  for path in [windows_csv, canonical_csv, sections_csv, labels_csv]:
    if not path.exists():
      raise FileNotFoundError(f"Current digit manifest not found: {path}")

  window_rows = load_csv_rows(windows_csv)
  canonical_rows = load_csv_rows(canonical_csv)
  section_rows = load_csv_rows(sections_csv)
  label_rows = load_csv_rows(labels_csv)

  require_columns(
    windows_csv,
    window_rows,
    {"split", "filename", "window_path", "reading", "roi_x", "roi_y", "roi_w", "roi_h"},
  )
  require_columns(
    canonical_csv,
    canonical_rows,
    {
      "split",
      "filename",
      "reading",
      "source_window_path",
      "canonical_window_path",
      "source_width",
      "source_height",
      "canonical_width",
      "canonical_height",
    },
  )
  require_columns(
    sections_csv,
    section_rows,
    {
      "split",
      "filename",
      "reading",
      "source_window_path",
      "canonical_window_path",
      "section_index",
      "x0",
      "y0",
      "x1",
      "y1",
      "section_path",
    },
  )
  require_columns(
    labels_csv,
    label_rows,
    {
      "split",
      "filename",
      "reading",
      "section_index",
      "digit",
      "section_path",
      "labeled_path",
      "source_window_path",
      "canonical_window_path",
    },
  )

  errors: list[str] = []
  warnings: list[str] = []
  window_split_counts = {"train": 0, "val": 0, "test": 0}
  section_split_counts = {"train": 0, "val": 0, "test": 0}
  digit_counts = {str(digit): 0 for digit in range(10)}

  windows_by_key: dict[tuple[str, str], dict[str, str]] = {}
  for row in window_rows:
    split = row.get("split", "")
    filename = row.get("filename", "")
    reading = row.get("reading", "")
    if add_split_count(window_split_counts, split, "windows.csv", errors):
      validate_reading(reading, filename, errors, cell_count)
    key = window_key(row)
    if key in windows_by_key:
      errors.append(f"Duplicate window row for {split}/{filename}")
    windows_by_key[key] = row
    ensure_manifest_file(dataset_root, row.get("window_path", ""), f"window {split}/{filename}", errors)
    for coord in ["roi_x", "roi_y", "roi_w", "roi_h"]:
      value = parse_int(row.get(coord, ""), f"{coord} in windows.csv for {filename}", errors)
      if value is not None and coord in {"roi_w", "roi_h"} and value <= 0:
        errors.append(f"Expected positive {coord} in windows.csv for {filename}, found {value}")

  canonical_by_key: dict[tuple[str, str], dict[str, str]] = {}
  for row in canonical_rows:
    split = row.get("split", "")
    filename = row.get("filename", "")
    reading = row.get("reading", "")
    key = window_key(row)
    source_row = windows_by_key.get(key)
    if not source_row:
      errors.append(f"Canonical row has no matching window row: {split}/{filename}")
    elif source_row.get("reading", "") != reading:
      errors.append(
        f"Reading mismatch between windows.csv and canonical_windows.csv for {split}/{filename}: "
        f"{source_row.get('reading', '')} vs {reading}"
      )
    if split not in VALID_SPLITS:
      errors.append(f"Unexpected split in canonical_windows.csv: {split}")
    validate_reading(reading, filename, errors, cell_count)
    if key in canonical_by_key:
      errors.append(f"Duplicate canonical row for {split}/{filename}")
    canonical_by_key[key] = row
    ensure_manifest_file(
      dataset_root,
      row.get("source_window_path", ""),
      f"canonical source window {split}/{filename}",
      errors,
    )
    ensure_manifest_file(
      dataset_root,
      row.get("canonical_window_path", ""),
      f"canonical window {split}/{filename}",
      errors,
    )
    for dimension in ["source_width", "source_height", "canonical_width", "canonical_height"]:
      value = parse_int(
        row.get(dimension, ""),
        f"{dimension} in canonical_windows.csv for {filename}",
        errors,
      )
      if value is not None and value <= 0:
        errors.append(f"Expected positive {dimension} for {split}/{filename}, found {value}")

  for key in windows_by_key.keys():
    if key not in canonical_by_key:
      errors.append(f"Window row has no matching canonical row: {key[0]}/{key[1]}")

  sections_by_window: dict[tuple[str, str], list[dict[str, str]]] = {}
  sections_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
  for row in section_rows:
    split = row.get("split", "")
    filename = row.get("filename", "")
    reading = row.get("reading", "")
    index_text = row.get("section_index", "")
    key = window_key(row)
    if add_split_count(section_split_counts, split, "sections.csv", errors):
      validate_reading(reading, filename, errors, cell_count)
    if key not in canonical_by_key:
      errors.append(f"Section row has no matching canonical row: {split}/{filename}")
    elif canonical_by_key[key].get("reading", "") != reading:
      errors.append(f"Reading mismatch between canonical and sections manifests for {split}/{filename}")
    section_index = parse_int(index_text, f"section_index in sections.csv for {filename}", errors)
    if section_index is not None and not 0 <= section_index < cell_count:
      errors.append(f"Out-of-range section_index for {split}/{filename}: {section_index}")
    for coord in ["x0", "y0", "x1", "y1"]:
      parse_int(row.get(coord, ""), f"{coord} in sections.csv for {filename}", errors)
    ensure_manifest_file(dataset_root, row.get("section_path", ""), f"section {split}/{filename}/{index_text}", errors)
    sections_by_window.setdefault(key, []).append(row)
    skey = section_key(row)
    if skey in sections_by_key:
      errors.append(f"Duplicate section row for {split}/{filename}/s{index_text}")
    sections_by_key[skey] = row

  for key in canonical_by_key.keys():
    rows = sections_by_window.get(key, [])
    if len(rows) != cell_count:
      errors.append(f"Expected {cell_count} section rows for {key[0]}/{key[1]}, found {len(rows)}")
      continue
    indices = sorted(row.get("section_index", "") for row in rows)
    expected = [str(index) for index in range(cell_count)]
    if indices != expected:
      errors.append(f"Unexpected section_index set for {key[0]}/{key[1]}: {indices}")

  labels_by_window: dict[tuple[str, str], list[dict[str, str]]] = {}
  for row in label_rows:
    split = row.get("split", "")
    filename = row.get("filename", "")
    reading = row.get("reading", "")
    index_text = row.get("section_index", "")
    digit = row.get("digit", "")
    key = window_key(row)
    skey = section_key(row)
    if split not in VALID_SPLITS:
      errors.append(f"Unexpected split in section_labels.csv: {split}")
    validate_reading(reading, filename, errors, cell_count)
    section_index = parse_int(index_text, f"section_index in section_labels.csv for {filename}", errors)
    if section_index is not None:
      if not 0 <= section_index < cell_count:
        errors.append(f"Out-of-range section label index for {split}/{filename}: {section_index}")
      elif reading.isdigit() and len(reading) == cell_count and digit != reading[section_index]:
        errors.append(
          f"Digit label mismatch for {split}/{filename}/s{section_index}: "
          f"expected {reading[section_index]}, found {digit}"
        )
    if digit not in DIGITS:
      errors.append(f"Unexpected digit in section_labels.csv for {split}/{filename}/s{index_text}: {digit}")
    else:
      digit_counts[digit] += 1
    section_row = sections_by_key.get(skey)
    if not section_row:
      errors.append(f"Label row has no matching section row: {split}/{filename}/s{index_text}")
    elif section_row.get("section_path", "") != row.get("section_path", ""):
      errors.append(f"Section path mismatch for label row: {split}/{filename}/s{index_text}")
    ensure_manifest_file(
      dataset_root,
      row.get("section_path", ""),
      f"labeled source section {split}/{filename}/s{index_text}",
      errors,
    )
    labeled_path = ensure_manifest_file(
      dataset_root,
      row.get("labeled_path", ""),
      f"labeled section {split}/{filename}/s{index_text}",
      errors,
    )
    expected_parent = dataset_root / "sections_labeled" / split / digit
    if digit in DIGITS and labeled_path.exists() and labeled_path.parent != expected_parent:
      errors.append(
        f"Labeled path not in expected split/digit folder: {labeled_path} "
        f"(expected parent {expected_parent})"
      )
    labels_by_window.setdefault(key, []).append(row)

  for key in canonical_by_key.keys():
    rows = labels_by_window.get(key, [])
    if len(rows) != cell_count:
      errors.append(f"Expected {cell_count} label rows for {key[0]}/{key[1]}, found {len(rows)}")
      continue
    indices = sorted(row.get("section_index", "") for row in rows)
    expected = [str(index) for index in range(cell_count)]
    if indices != expected:
      errors.append(f"Unexpected label section_index set for {key[0]}/{key[1]}: {indices}")

  synthetic_count = 0
  if not skip_synthetic:
    synthetic_count, synthetic_warnings, synthetic_errors = validate_synthetic_dataset(dataset_root)
    warnings.extend(synthetic_warnings)
    errors.extend(synthetic_errors)

  print(f"Dataset root: {dataset_root}")
  print("Workflow: current windows/canonical/sections")
  print(f"Window rows: {len(window_rows)}")
  print(f"Canonical rows: {len(canonical_rows)}")
  print(f"Section rows: {len(section_rows)}")
  print(f"Label rows: {len(label_rows)}")
  if not skip_synthetic:
    print(f"Synthetic rows: {synthetic_count}")
  print(
    "Window rows by split: "
    f"train={window_split_counts['train']} val={window_split_counts['val']} test={window_split_counts['test']}"
  )
  print(
    "Section rows by split: "
    f"train={section_split_counts['train']} val={section_split_counts['val']} test={section_split_counts['test']}"
  )
  print(
    "Label rows by digit: "
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


def validate_synthetic_dataset(dataset_root: Path) -> tuple[int, list[str], list[str]]:
  synthetic_root = dataset_root / "sections_synthetic"
  manifest_path = synthetic_root / "manifests" / "synthetic_cells.csv"
  warnings: list[str] = []
  errors: list[str] = []
  if not manifest_path.exists():
    warnings.append(f"Synthetic manifest not found; skipping synthetic validation: {manifest_path}")
    return 0, warnings, errors

  rows = load_csv_rows(manifest_path)
  require_columns(
    manifest_path,
    rows,
    {"kind", "digit", "output_path", "source_path", "window_id", "cell_index", "sequence"},
  )

  sections_labeled_root = dataset_root / "sections_labeled"
  for row in rows:
    kind = row.get("kind", "")
    digit = row.get("digit", "")
    output_path = synthetic_root / row.get("output_path", "")
    if kind not in {"direct", "composed"}:
      errors.append(f"Unexpected synthetic kind: {kind}")
    if digit not in DIGITS:
      errors.append(f"Unexpected synthetic digit: {digit}")
    if not output_path.exists():
      errors.append(f"Missing synthetic output file: {output_path}")
    elif digit in DIGITS and output_path.parent != synthetic_root / "train" / digit:
      errors.append(
        f"Synthetic output path not in expected digit folder: {output_path} "
        f"(expected parent {synthetic_root / 'train' / digit})"
      )

    for source_rel in row.get("source_path", "").split("|"):
      if not source_rel:
        errors.append(f"Missing synthetic source path for output: {output_path}")
        continue
      source_path = sections_labeled_root / source_rel
      if not source_path.exists():
        errors.append(f"Missing synthetic source file: {source_path}")

  return len(rows), warnings, errors


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent
  dataset_root = resolve(base_dir, args.dataset_root)

  if args.cell_count <= 0:
    raise ValueError("--cell-count must be positive.")
  if not dataset_root.exists():
    raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

  validate_current_dataset(dataset_root, args.cell_count, args.skip_synthetic)


if __name__ == "__main__":
  main()
