from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DIGIT_COUNT = 10
DEFAULT_READING_WIDTH = 4


@dataclass
class DigitCounts:
  train: int = 0
  val: int = 0
  test: int = 0

  @property
  def total(self) -> int:
    return self.train + self.val + self.test


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Plan targeted dataset expansion for underrepresented digit classes."
  )
  parser.add_argument(
    "--cells-manifest",
    default="data/digit_dataset/manifests/cells.csv",
    help="Path to exported cells manifest."
  )
  parser.add_argument(
    "--readings-csv",
    default="../assets/meter_readings.csv",
    help="Path to readings CSV used for seed value suggestions."
  )
  parser.add_argument(
    "--target-train-per-digit",
    type=int,
    default=12,
    help="Desired minimum number of train samples per digit."
  )
  parser.add_argument(
    "--priority-digits",
    default="4,5,6,9",
    help="Comma-separated digit classes to prioritize."
  )
  parser.add_argument(
    "--max-suggestions-per-digit",
    type=int,
    default=10,
    help="Maximum suggested reading labels to emit per priority digit."
  )
  parser.add_argument(
    "--out-json",
    default="data/digit_dataset/manifests/capture_plan.json",
    help="Output JSON plan path."
  )
  parser.add_argument(
    "--out-md",
    default="data/digit_dataset/manifests/capture_plan.md",
    help="Output markdown checklist path."
  )
  return parser.parse_args()


def resolve(base_dir: Path, value: str) -> Path:
  path = Path(value)
  if path.is_absolute():
    return path
  return (base_dir / path).resolve()


def parse_priority_digits(raw: str) -> list[str]:
  result: list[str] = []
  seen = set()
  for token in (raw or "").split(","):
    value = token.strip()
    if not value:
      continue
    if len(value) != 1 or value < "0" or value > "9":
      raise ValueError(f"Invalid priority digit: {value}")
    if value in seen:
      continue
    seen.add(value)
    result.append(value)
  if not result:
    raise ValueError("At least one priority digit is required.")
  return result


def load_digit_counts(cells_manifest_path: Path) -> dict[str, DigitCounts]:
  counts = {str(digit): DigitCounts() for digit in range(DEFAULT_DIGIT_COUNT)}
  with cells_manifest_path.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      digit = (row.get("digit") or "").strip()
      split = (row.get("split") or "").strip().lower()
      if digit not in counts:
        continue
      if split == "train":
        counts[digit].train += 1
      elif split == "val":
        counts[digit].val += 1
      elif split == "test":
        counts[digit].test += 1
  return counts


def load_seed_reading(readings_csv_path: Path) -> str:
  best_value = None
  with readings_csv_path.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      value = (row.get("value") or "").strip()
      if len(value) != DEFAULT_READING_WIDTH or not value.isdigit():
        continue
      parsed = int(value)
      if best_value is None or parsed > best_value:
        best_value = parsed
  if best_value is None:
    return "2300"
  return f"{best_value:0{DEFAULT_READING_WIDTH}d}"


def generate_label_suggestions(seed_label: str, digit: str, limit: int) -> list[str]:
  if len(seed_label) != DEFAULT_READING_WIDTH or not seed_label.isdigit():
    seed_label = "2300"
  candidates: list[str] = []

  # Single-position replacements.
  for position in range(DEFAULT_READING_WIDTH):
    chars = list(seed_label)
    chars[position] = digit
    candidates.append("".join(chars))

  # Two-position replacements.
  for left in range(DEFAULT_READING_WIDTH):
    for right in range(left + 1, DEFAULT_READING_WIDTH):
      chars = list(seed_label)
      chars[left] = digit
      chars[right] = digit
      candidates.append("".join(chars))

  candidates.append(digit * DEFAULT_READING_WIDTH)
  deduped = []
  seen = set()
  for value in candidates:
    if value in seen:
      continue
    seen.add(value)
    deduped.append(value)
    if len(deduped) >= limit:
      break
  return deduped


def build_plan_payload(
  counts: dict[str, DigitCounts],
  target_train_per_digit: int,
  priority_digits: list[str],
  seed_label: str,
  max_suggestions_per_digit: int,
  cells_manifest_path: Path,
  readings_csv_path: Path
) -> dict:
  digits_summary = []
  for digit in map(str, range(DEFAULT_DIGIT_COUNT)):
    item = counts[digit]
    train_deficit = max(0, target_train_per_digit - item.train)
    digits_summary.append({
      "digit": digit,
      "train": item.train,
      "val": item.val,
      "test": item.test,
      "total": item.total,
      "train_deficit": train_deficit
    })

  priority_actions = []
  for rank, digit in enumerate(priority_digits, start=1):
    item = counts[digit]
    deficit = max(0, target_train_per_digit - item.train)
    priority_actions.append({
      "priority_rank": rank,
      "digit": digit,
      "current_train": item.train,
      "target_train": target_train_per_digit,
      "train_deficit": deficit,
      "suggested_labels": generate_label_suggestions(seed_label, digit, max_suggestions_per_digit)
    })

  return {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "inputs": {
      "cells_manifest": str(cells_manifest_path),
      "readings_csv": str(readings_csv_path),
      "target_train_per_digit": target_train_per_digit,
      "priority_digits": priority_digits,
      "seed_label": seed_label
    },
    "digits": digits_summary,
    "priority_actions": priority_actions
  }


def render_markdown(plan: dict) -> str:
  lines = []
  lines.append("# Digit Capture Plan")
  lines.append("")
  lines.append(f"- Generated: `{plan['generated_at']}`")
  lines.append(f"- Target train samples per digit: `{plan['inputs']['target_train_per_digit']}`")
  lines.append(f"- Priority digits: `{','.join(plan['inputs']['priority_digits'])}`")
  lines.append(f"- Seed label for examples: `{plan['inputs']['seed_label']}`")
  lines.append("")
  lines.append("## Coverage Snapshot")
  lines.append("")
  lines.append("| Digit | Train | Val | Test | Total | Train Deficit |")
  lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
  for item in plan["digits"]:
    lines.append(
      f"| {item['digit']} | {item['train']} | {item['val']} | {item['test']} | "
      f"{item['total']} | {item['train_deficit']} |"
    )
  lines.append("")
  lines.append("## Priority Checklist")
  lines.append("")
  active_items = [item for item in plan["priority_actions"] if item["train_deficit"] > 0]
  if not active_items:
    lines.append("- No deficits for priority digits.")
    lines.append("")
    return "\n".join(lines)

  for item in active_items:
    lines.append(
      f"- [ ] Digit `{item['digit']}`: collect at least `{item['train_deficit']}` additional "
      "train occurrences."
    )
    lines.append(
      f"  Current train count: `{item['current_train']}`; "
      f"target: `{item['target_train']}`."
    )
    suggestions = ", ".join([f"`{value}`" for value in item["suggested_labels"]])
    lines.append(f"  Suggested reading labels to target: {suggestions}")
    lines.append("")

  lines.append("## QA Loop")
  lines.append("")
  lines.append(
    "- After adding labels, rebuild dataset and run `python validate_digit_dataset.py` "
    "before training."
  )
  return "\n".join(lines)


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent
  cells_manifest_path = resolve(base_dir, args.cells_manifest)
  readings_csv_path = resolve(base_dir, args.readings_csv)
  out_json_path = resolve(base_dir, args.out_json)
  out_md_path = resolve(base_dir, args.out_md)

  if not cells_manifest_path.exists():
    raise FileNotFoundError(f"Cells manifest not found: {cells_manifest_path}")
  if not readings_csv_path.exists():
    raise FileNotFoundError(f"Readings CSV not found: {readings_csv_path}")
  if args.target_train_per_digit <= 0:
    raise ValueError("--target-train-per-digit must be positive.")
  if args.max_suggestions_per_digit <= 0:
    raise ValueError("--max-suggestions-per-digit must be positive.")

  priority_digits = parse_priority_digits(args.priority_digits)
  counts = load_digit_counts(cells_manifest_path)
  seed_label = load_seed_reading(readings_csv_path)
  plan = build_plan_payload(
    counts=counts,
    target_train_per_digit=args.target_train_per_digit,
    priority_digits=priority_digits,
    seed_label=seed_label,
    max_suggestions_per_digit=args.max_suggestions_per_digit,
    cells_manifest_path=cells_manifest_path,
    readings_csv_path=readings_csv_path
  )

  out_json_path.parent.mkdir(parents=True, exist_ok=True)
  out_md_path.parent.mkdir(parents=True, exist_ok=True)
  out_json_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
  out_md_path.write_text(render_markdown(plan), encoding="utf-8")

  print(f"Wrote JSON plan: {out_json_path}")
  print(f"Wrote markdown checklist: {out_md_path}")
  print("Priority deficits:")
  for item in plan["priority_actions"]:
    print(
      f"  digit {item['digit']}: current_train={item['current_train']} "
      f"target={item['target_train']} deficit={item['train_deficit']}"
    )


if __name__ == "__main__":
  main()
