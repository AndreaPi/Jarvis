from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
  from .digit_model import DEFAULT_NUM_CLASSES, DEFAULT_IMAGE_SIZE, build_digit_cnn
  from .train_digit_classifier import (
    DigitCellDataset,
    Sample,
    collect_split_samples,
    count_labels,
    make_class_weights,
    run_epoch,
    select_synthetic_samples,
    set_seed,
    state_dict_to_cpu
  )
except ImportError:
  from digit_model import DEFAULT_NUM_CLASSES, DEFAULT_IMAGE_SIZE, build_digit_cnn
  from train_digit_classifier import (
    DigitCellDataset,
    Sample,
    collect_split_samples,
    count_labels,
    make_class_weights,
    run_epoch,
    select_synthetic_samples,
    set_seed,
    state_dict_to_cpu
  )


SUPPORTED_SPLITS = {"train", "val", "test"}


@dataclass
class SourceSample:
  path: Path
  label: int
  filename: str
  source: str


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Grouped cross-validation for digit classifier fine-tuning recipes."
  )
  parser.add_argument("--dataset-root", default="data/digit_dataset")
  parser.add_argument("--runtime-root", default="data/runtime_failure_dataset")
  parser.add_argument("--synthetic-root", default="data/digit_dataset/sections_synthetic")
  parser.add_argument("--init-checkpoint", default="models/digit_classifier.pt")
  parser.add_argument("--output-dir", default="runs/digit-classifier-finetune-cv")
  parser.add_argument("--folds", type=int, default=5)
  parser.add_argument("--epochs", type=int, default=60)
  parser.add_argument("--patience", type=int, default=12)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=4.0e-4)
  parser.add_argument("--weight-decay", type=float, default=1.0e-4)
  parser.add_argument("--label-smoothing", type=float, default=0.03)
  parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
  parser.add_argument("--synthetic-target-ratio", type=float, default=2.0)
  parser.add_argument(
    "--synthetic-selection-strategy",
    choices=("balanced", "proportional"),
    default="balanced"
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cpu")
  parser.add_argument(
    "--clean-splits",
    default="train",
    help=(
      "Comma-separated clean manifest splits to include in grouped CV. "
      "Default is train only so the fixed test holdout remains untouched."
    )
  )
  return parser.parse_args()


def resolve_path(base_dir: Path, value: str) -> Path:
  path = Path(value)
  if path.is_absolute():
    return path
  return (base_dir / path).resolve()


def resolve_device(raw: str) -> torch.device:
  normalized = (raw or "cpu").strip().lower()
  if normalized == "auto":
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  if normalized.isdigit():
    return torch.device(f"cuda:{normalized}")
  return torch.device(normalized)


def parse_clean_splits(raw: str) -> set[str]:
  splits = {value.strip().lower() for value in raw.split(",") if value.strip()}
  unknown = splits - SUPPORTED_SPLITS
  if unknown:
    raise ValueError(f"Unsupported split(s): {sorted(unknown)}")
  return splits or {"train"}


def load_clean_samples(dataset_root: Path, splits: set[str]) -> list[SourceSample]:
  manifest_path = dataset_root / "manifests" / "section_labels.csv"
  if not manifest_path.exists():
    raise FileNotFoundError(f"Section labels manifest not found: {manifest_path}")
  samples: list[SourceSample] = []
  with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      split = (row.get("split") or "").strip().lower()
      if split not in splits:
        continue
      filename = (row.get("filename") or "").strip()
      digit = int(row["digit"])
      labeled_path = dataset_root / (row.get("labeled_path") or "")
      if not filename or not labeled_path.exists():
        continue
      samples.append(SourceSample(
        path=labeled_path,
        label=digit,
        filename=filename,
        source="clean"
      ))
  return samples


def load_runtime_samples(runtime_root: Path, stem_to_filename: dict[str, str]) -> list[SourceSample]:
  samples: list[SourceSample] = []
  root = runtime_root / "sections_labeled" / "train"
  if not root.exists():
    return samples
  for digit in range(DEFAULT_NUM_CLASSES):
    digit_dir = root / str(digit)
    if not digit_dir.exists():
      continue
    for path in sorted(digit_dir.glob("*.png")):
      source_stem = path.name.split("__", 1)[0]
      filename = stem_to_filename.get(source_stem)
      if not filename:
        continue
      samples.append(SourceSample(
        path=path,
        label=digit,
        filename=filename,
        source="runtime-failure"
      ))
  return samples


def make_folds(filenames: list[str], fold_count: int, seed: int) -> list[list[str]]:
  if fold_count < 2:
    raise ValueError("--folds must be at least 2")
  if fold_count > len(filenames):
    fold_count = len(filenames)
  rng = random.Random(seed)
  shuffled = list(filenames)
  rng.shuffle(shuffled)
  folds = [[] for _ in range(fold_count)]
  for index, filename in enumerate(shuffled):
    folds[index % fold_count].append(filename)
  return [sorted(fold) for fold in folds]


def as_train_samples(samples: list[SourceSample] | list[Sample]) -> list[Sample]:
  return [Sample(path=sample.path, label=sample.label) for sample in samples]


def evaluate_predictions(
  model: nn.Module,
  samples: list[SourceSample],
  image_size: int,
  batch_size: int,
  device: torch.device
) -> dict[str, object]:
  if not samples:
    return {
      "samples": 0,
      "accuracy": 0.0,
      "confusion": [[0 for _ in range(DEFAULT_NUM_CLASSES)] for _ in range(DEFAULT_NUM_CLASSES)]
    }
  dataset = DigitCellDataset(as_train_samples(samples), image_size=image_size, augment=False)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
  confusion = [[0 for _ in range(DEFAULT_NUM_CLASSES)] for _ in range(DEFAULT_NUM_CLASSES)]
  correct = 0
  total = 0
  model.eval()
  with torch.no_grad():
    for inputs, targets in loader:
      inputs = inputs.to(device)
      targets = targets.to(device)
      logits = model(inputs)
      predictions = torch.argmax(logits, dim=1)
      for target, prediction in zip(targets.cpu().tolist(), predictions.cpu().tolist()):
        confusion[target][prediction] += 1
        total += 1
        if target == prediction:
          correct += 1
  return {
    "samples": total,
    "accuracy": correct / total if total else 0.0,
    "confusion": confusion
  }


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
  payload = torch.load(str(checkpoint_path), map_location="cpu")
  if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
    raise RuntimeError(f"Checkpoint missing state_dict: {checkpoint_path}")
  model = build_digit_cnn(DEFAULT_NUM_CLASSES)
  model.load_state_dict(payload["state_dict"], strict=True)
  model.to(device)
  return model


def train_from_checkpoint(
  checkpoint_path: Path,
  train_samples: list[SourceSample],
  val_samples: list[SourceSample],
  args: argparse.Namespace,
  device: torch.device
) -> tuple[nn.Module, dict[str, object]]:
  model = load_model(checkpoint_path, device)
  train_dataset = DigitCellDataset(
    as_train_samples(train_samples),
    image_size=args.image_size,
    augment=True,
    split_jitter_x=0.08,
    split_jitter_y=0.08,
    split_jitter_prob=0.85
  )
  val_dataset = DigitCellDataset(as_train_samples(val_samples), image_size=args.image_size, augment=False)
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
  val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
  class_weights = make_class_weights(as_train_samples(train_samples), device)
  criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay
  )
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
    threshold=1.0e-4,
    min_lr=1.0e-5
  )

  best_state = state_dict_to_cpu(model)
  best_epoch = 0
  best_loss = float("inf")
  epochs_without_improvement = 0
  history: list[dict[str, float | int]] = []
  start = time.perf_counter()
  for epoch in range(1, args.epochs + 1):
    train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
    val_metrics = run_epoch(model, val_loader, criterion, None, device)
    scheduler.step(val_metrics["loss"])
    history.append({
      "epoch": epoch,
      "train_loss": train_metrics["loss"],
      "train_accuracy": train_metrics["accuracy"],
      "val_loss": val_metrics["loss"],
      "val_accuracy": val_metrics["accuracy"],
      "learning_rate": optimizer.param_groups[0]["lr"]
    })
    if val_metrics["loss"] < best_loss - 1.0e-4:
      best_loss = val_metrics["loss"]
      best_epoch = epoch
      best_state = state_dict_to_cpu(model)
      epochs_without_improvement = 0
    else:
      epochs_without_improvement += 1
    if epochs_without_improvement >= args.patience:
      break

  model.load_state_dict(best_state)
  model.to(device)
  return model, {
    "best_epoch": best_epoch,
    "training_seconds": time.perf_counter() - start,
    "history": history
  }


def summarize_counts(samples: list[SourceSample] | list[Sample]) -> dict[str, int]:
  return count_labels(as_train_samples(samples) if samples and isinstance(samples[0], SourceSample) else samples)


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent
  dataset_root = resolve_path(base_dir, args.dataset_root)
  runtime_root = resolve_path(base_dir, args.runtime_root)
  synthetic_root = resolve_path(base_dir, args.synthetic_root)
  init_checkpoint = resolve_path(base_dir, args.init_checkpoint)
  output_dir = resolve_path(base_dir, args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  if not init_checkpoint.exists():
    raise FileNotFoundError(f"Init checkpoint not found: {init_checkpoint}")
  splits = parse_clean_splits(args.clean_splits)
  clean_samples = load_clean_samples(dataset_root, splits)
  if not clean_samples:
    raise RuntimeError("No clean samples found for CV.")
  stem_to_filename = {
    Path(sample.filename).stem: sample.filename
    for sample in clean_samples
  }
  runtime_samples = load_runtime_samples(runtime_root, stem_to_filename)
  synthetic_samples = collect_split_samples(synthetic_root, "train") if synthetic_root.exists() else []
  filenames = sorted({sample.filename for sample in clean_samples})
  folds = make_folds(filenames, args.folds, args.seed)

  device = resolve_device(args.device)
  if device.type.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA device requested but CUDA is not available.")
  set_seed(args.seed)

  configs = [
    {
      "name": "restored-baseline",
      "train": False,
      "use_runtime": False,
      "use_synthetic": False
    },
    {
      "name": "finetune-clean",
      "train": True,
      "use_runtime": False,
      "use_synthetic": False
    },
    {
      "name": "finetune-clean-runtime",
      "train": True,
      "use_runtime": True,
      "use_synthetic": True
    }
  ]
  summary: dict[str, object] = {
    "init_checkpoint": str(init_checkpoint),
    "dataset_root": str(dataset_root),
    "runtime_root": str(runtime_root),
    "synthetic_root": str(synthetic_root) if synthetic_root.exists() else None,
    "clean_splits": sorted(splits),
    "source_filenames": filenames,
    "folds": [],
    "configs": {},
    "notes": [
      "Folds are grouped by source filename to avoid sibling-cell leakage.",
      "By default, grouped CV uses the train pool only; keep fixed holdout splits for final diagnostics.",
      "All trainable configs fine-tune from the restored promoted checkpoint; this is an incremental recipe check, not a from-scratch independence test.",
      "UI benchmark metrics are not run inside CV; run the UI test set separately before promotion."
    ]
  }

  for config in configs:
    summary["configs"][config["name"]] = {
      "folds": []
    }

  for fold_index, val_filenames in enumerate(folds, start=1):
    val_set = set(val_filenames)
    train_clean = [sample for sample in clean_samples if sample.filename not in val_set]
    val_clean = [sample for sample in clean_samples if sample.filename in val_set]
    train_runtime = [sample for sample in runtime_samples if sample.filename not in val_set]
    fold_payload = {
      "fold": fold_index,
      "val_filenames": val_filenames,
      "train_clean_samples": len(train_clean),
      "val_clean_samples": len(val_clean),
      "train_runtime_samples_available": len(train_runtime)
    }
    summary["folds"].append(fold_payload)

    for config in configs:
      set_seed(args.seed + fold_index)
      train_samples = list(train_clean)
      runtime_count = 0
      synthetic_count = 0
      if config["use_runtime"]:
        train_samples.extend(train_runtime)
        runtime_count = len(train_runtime)
      selected_synthetic = []
      if config["use_synthetic"] and synthetic_samples:
        selected_synthetic = select_synthetic_samples(
          real_samples=as_train_samples(train_samples),
          synthetic_samples=synthetic_samples,
          target_ratio=args.synthetic_target_ratio,
          seed=args.seed + fold_index,
          strategy=args.synthetic_selection_strategy
        )
        train_samples.extend(
          SourceSample(
            path=sample.path,
            label=sample.label,
            filename="__synthetic__",
            source="synthetic"
          )
          for sample in selected_synthetic
        )
        synthetic_count = len(selected_synthetic)

      if config["train"]:
        model, training_summary = train_from_checkpoint(
          checkpoint_path=init_checkpoint,
          train_samples=train_samples,
          val_samples=val_clean,
          args=args,
          device=device
        )
      else:
        model = load_model(init_checkpoint, device)
        training_summary = {
          "best_epoch": 0,
          "training_seconds": 0.0,
          "history": []
        }
      evaluation = evaluate_predictions(
        model=model,
        samples=val_clean,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=device
      )
      summary["configs"][config["name"]]["folds"].append({
        "fold": fold_index,
        "val_filenames": val_filenames,
        "train_samples": len(train_samples),
        "train_clean_samples": len(train_clean),
        "train_runtime_samples": runtime_count,
        "train_synthetic_samples": synthetic_count,
        "train_counts": summarize_counts(train_samples),
        "val_counts": summarize_counts(val_clean),
        "evaluation": evaluation,
        "training": training_summary
      })
      print(
        f"fold {fold_index}/{len(folds)} {config['name']}: "
        f"val_acc={evaluation['accuracy'] * 100:.1f}% "
        f"train={len(train_samples)} runtime={runtime_count} synthetic={synthetic_count}"
      )

  for config in configs:
    folds_payload = summary["configs"][config["name"]]["folds"]
    total_samples = sum(row["evaluation"]["samples"] for row in folds_payload)
    weighted_accuracy = (
      sum(row["evaluation"]["accuracy"] * row["evaluation"]["samples"] for row in folds_payload)
      / max(1, total_samples)
    )
    summary["configs"][config["name"]]["aggregate"] = {
      "samples": total_samples,
      "weighted_accuracy": weighted_accuracy
    }

  summary_path = output_dir / "digit_classifier_finetune_cv_summary.json"
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps({
    "summary": str(summary_path),
    "configs": {
      name: payload["aggregate"]
      for name, payload in summary["configs"].items()
    }
  }, indent=2))


if __name__ == "__main__":
  main()
