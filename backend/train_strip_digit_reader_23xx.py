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
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
  from .strip_digit_model import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_NUM_CLASSES,
    DEFAULT_STRIP_HEIGHT,
    DEFAULT_STRIP_WIDTH,
    build_strip_digit_reader_23xx,
    prepare_strip_tensor
  )
except ImportError:
  from strip_digit_model import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_NUM_CLASSES,
    DEFAULT_STRIP_HEIGHT,
    DEFAULT_STRIP_WIDTH,
    build_strip_digit_reader_23xx,
    prepare_strip_tensor
  )


FIXED_PREFIX = "23"


@dataclass
class Strip23xxSample:
  path: Path
  reading: str
  filename: str
  split: str

  @property
  def guard_label(self) -> int:
    return 1 if len(self.reading) >= 2 and self.reading[1] == "3" else 0

  @property
  def suffix(self) -> str:
    return self.reading[2:4]


class Strip23xxDataset(Dataset):
  def __init__(
    self,
    samples: list[Strip23xxSample],
    image_width: int,
    image_height: int,
    augment: bool = False
  ) -> None:
    self.samples = samples
    self.image_width = image_width
    self.image_height = image_height
    self.augment = augment

  def __len__(self) -> int:
    return len(self.samples)

  def _augment_image(self, image: Image.Image) -> Image.Image:
    if not self.augment:
      return image
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    if random.random() < 0.8:
      scale = random.uniform(0.94, 1.05)
      scaled = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        resample
      )
      canvas = Image.new("L", image.size, color=255)
      max_dx = max(1, int(round(image.width * 0.07)))
      max_dy = max(1, int(round(image.height * 0.09)))
      left = (image.width - scaled.width) // 2 + random.randint(-max_dx, max_dx)
      top = (image.height - scaled.height) // 2 + random.randint(-max_dy, max_dy)
      canvas.paste(scaled, (left, top))
      image = canvas
    if random.random() < 0.85:
      image = image.rotate(random.uniform(-3.0, 3.0), resample=resample, fillcolor=255)
    if random.random() < 0.45:
      shear = random.uniform(-0.04, 0.04)
      dx = random.uniform(-image.width * 0.025, image.width * 0.025)
      dy = random.uniform(-image.height * 0.03, image.height * 0.03)
      image = image.transform(
        image.size,
        Image.Transform.AFFINE if hasattr(Image, "Transform") else Image.AFFINE,
        (1, shear, -dx, 0, 1, -dy),
        resample=resample,
        fillcolor=255
      )
    if random.random() < 0.65:
      image = ImageEnhance.Contrast(image).enhance(random.uniform(0.72, 1.35))
    if random.random() < 0.65:
      image = ImageEnhance.Brightness(image).enhance(random.uniform(0.78, 1.22))
    if random.random() < 0.35:
      image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.65)))
    return image

  def __getitem__(self, index: int):
    sample = self.samples[index]
    with Image.open(sample.path) as source:
      image = source.convert("L")
      image = self._augment_image(image)
      tensor = prepare_strip_tensor(image, width=self.image_width, height=self.image_height)
    guard = torch.tensor(sample.guard_label, dtype=torch.long)
    suffix = torch.tensor([int(digit) for digit in sample.suffix], dtype=torch.long)
    return tensor, guard, suffix


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Train a guarded 23xx whole-strip reader for Jarvis OCR shadow evaluation."
  )
  parser.add_argument("--dataset-root", default="data/digit_dataset")
  parser.add_argument("--manifest", default="manifests/canonical_windows.csv")
  parser.add_argument("--epochs", type=int, default=120)
  parser.add_argument("--cv-epochs", type=int, default=50)
  parser.add_argument("--cv-folds", type=int, default=5)
  parser.add_argument("--batch-size", type=int, default=8)
  parser.add_argument("--learning-rate", type=float, default=8.0e-4)
  parser.add_argument("--weight-decay", type=float, default=1.0e-4)
  parser.add_argument("--patience", type=int, default=28)
  parser.add_argument("--image-width", type=int, default=DEFAULT_STRIP_WIDTH)
  parser.add_argument("--image-height", type=int, default=DEFAULT_STRIP_HEIGHT)
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument("--label-smoothing", type=float, default=0.03)
  parser.add_argument("--guard-threshold", type=float, default=0.98)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="auto")
  parser.add_argument("--project", default="runs")
  parser.add_argument("--name", default="strip-digit-reader-23xx")
  parser.add_argument("--copy-to", default="models/digit_strip_reader_23xx.pt")
  parser.add_argument("--skip-cv", action="store_true")
  return parser.parse_args()


def resolve_path(base_dir: Path, value: str) -> Path:
  path = Path(value)
  if path.is_absolute():
    return path
  return (base_dir / path).resolve()


def resolve_device(raw: str) -> torch.device:
  normalized = (raw or "").strip().lower()
  if not normalized or normalized == "auto":
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  if normalized in {"cpu", "mps"}:
    return torch.device(normalized)
  if normalized.isdigit():
    return torch.device(f"cuda:{normalized}")
  return torch.device(normalized)


def set_seed(seed: int) -> None:
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def collect_samples(dataset_root: Path, manifest_path: Path) -> list[Strip23xxSample]:
  samples: list[Strip23xxSample] = []
  with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      split = (row.get("split") or "").strip()
      reading = (row.get("reading") or "").strip()
      relative_path = (row.get("canonical_window_path") or "").strip()
      filename = (row.get("filename") or "").strip()
      if split not in {"train", "val", "test"}:
        continue
      if len(reading) != 4 or not reading.isdigit():
        continue
      path = dataset_root / relative_path
      if not relative_path or not path.exists():
        continue
      samples.append(Strip23xxSample(path=path, reading=reading, filename=filename, split=split))
  return samples


def state_dict_to_cpu(model: nn.Module) -> dict:
  return {
    key: value.detach().cpu().clone()
    for key, value in model.state_dict().items()
  }


def make_guard_weights(samples: list[Strip23xxSample], device: torch.device) -> torch.Tensor:
  positives = sum(sample.guard_label for sample in samples)
  negatives = len(samples) - positives
  if positives <= 0 or negatives <= 0:
    return torch.ones(2, dtype=torch.float32, device=device)
  mean_count = (positives + negatives) / 2.0
  return torch.tensor([mean_count / negatives, mean_count / positives], dtype=torch.float32, device=device)


def make_suffix_weights(samples: list[Strip23xxSample], device: torch.device) -> torch.Tensor:
  counts = {str(index): 0 for index in range(DEFAULT_NUM_CLASSES)}
  for sample in samples:
    for digit in sample.suffix:
      counts[digit] += 1
  non_zero = [count for count in counts.values() if count > 0]
  if not non_zero:
    return torch.ones(DEFAULT_NUM_CLASSES, dtype=torch.float32, device=device)
  mean_count = sum(non_zero) / len(non_zero)
  weights = torch.zeros(DEFAULT_NUM_CLASSES, dtype=torch.float32, device=device)
  for class_index in range(DEFAULT_NUM_CLASSES):
    count = counts[str(class_index)]
    weights[class_index] = 0.0 if count <= 0 else mean_count / count
  return weights


def run_epoch(
  model: nn.Module,
  loader: DataLoader,
  guard_criterion: nn.Module,
  suffix_criterion: nn.Module,
  optimizer: torch.optim.Optimizer | None,
  device: torch.device,
  guard_threshold: float
) -> dict[str, object]:
  training = optimizer is not None
  model.train() if training else model.eval()
  total_loss = 0.0
  total_count = 0
  guard_correct = 0
  guard_fp = 0
  guard_fn = 0
  guard_positive_count = 0
  suffix_exact_positive = 0
  full_exact_when_accepted = 0
  accepted_count = 0
  suffix_position_correct = np.zeros(2, dtype=np.int64)

  for inputs, guard_targets, suffix_targets in loader:
    inputs = inputs.to(device, non_blocking=True)
    guard_targets = guard_targets.to(device, non_blocking=True)
    suffix_targets = suffix_targets.to(device, non_blocking=True)
    if training:
      optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
      outputs = model(inputs)
      guard_logits = outputs["guard_logits"]
      suffix_logits = outputs["suffix_logits"]
      guard_loss = guard_criterion(guard_logits, guard_targets)
      suffix_loss = suffix_criterion(
        suffix_logits.reshape(-1, DEFAULT_NUM_CLASSES),
        suffix_targets.reshape(-1)
      )
      loss = guard_loss + suffix_loss
      if training:
        loss.backward()
        optimizer.step()

    batch_size = int(guard_targets.shape[0])
    total_loss += float(loss.item()) * batch_size
    total_count += batch_size
    guard_prob = torch.softmax(guard_logits, dim=1)[:, 1]
    guard_pred = guard_prob >= guard_threshold
    guard_truth = guard_targets == 1
    guard_correct += int((guard_pred == guard_truth).sum().item())
    guard_fp += int((guard_pred & ~guard_truth).sum().item())
    guard_fn += int((~guard_pred & guard_truth).sum().item())
    guard_positive_count += int(guard_truth.sum().item())

    suffix_pred = torch.argmax(suffix_logits, dim=2)
    suffix_matches = suffix_pred == suffix_targets
    suffix_position_correct += suffix_matches.sum(dim=0).detach().cpu().numpy().astype(np.int64)
    suffix_exact = suffix_matches.all(dim=1)
    suffix_exact_positive += int((suffix_exact & guard_truth).sum().item())
    accepted_count += int(guard_pred.sum().item())
    full_exact_when_accepted += int((suffix_exact & guard_pred & guard_truth).sum().item())

  if total_count == 0:
    return {
      "loss": 0.0,
      "guard_accuracy": 0.0,
      "guard_false_positive_count": 0,
      "guard_false_positive_rate": 0.0,
      "guard_false_negative_count": 0,
      "guard_false_negative_rate": 0.0,
      "guard_accepted_count": 0,
      "suffix_exact_accuracy_when_guard_target_true": 0.0,
      "full_23xx_exact_accuracy_when_guard_accepts": 0.0,
      "suffix_position_accuracy": 0.0,
      "per_suffix_position_accuracy": [0.0, 0.0]
    }

  negatives = total_count - guard_positive_count
  suffix_total_positions = total_count * 2
  return {
    "loss": total_loss / total_count,
    "guard_accuracy": guard_correct / total_count,
    "guard_false_positive_count": guard_fp,
    "guard_false_positive_rate": guard_fp / max(1, negatives),
    "guard_false_negative_count": guard_fn,
    "guard_false_negative_rate": guard_fn / max(1, guard_positive_count),
    "guard_accepted_count": accepted_count,
    "suffix_exact_accuracy_when_guard_target_true": suffix_exact_positive / max(1, guard_positive_count),
    "full_23xx_exact_accuracy_when_guard_accepts": full_exact_when_accepted / max(1, accepted_count),
    "suffix_position_accuracy": float(suffix_position_correct.sum() / max(1, suffix_total_positions)),
    "per_suffix_position_accuracy": [
      float(suffix_position_correct[position] / total_count)
      for position in range(2)
    ]
  }


def fit_model(
  train_samples: list[Strip23xxSample],
  eval_samples: list[Strip23xxSample],
  args: argparse.Namespace,
  device: torch.device,
  epochs: int
) -> tuple[nn.Module, dict[str, object], list[dict[str, object]], int]:
  train_dataset = Strip23xxDataset(train_samples, args.image_width, args.image_height, augment=True)
  train_eval_dataset = Strip23xxDataset(train_samples, args.image_width, args.image_height, augment=False)
  eval_dataset = Strip23xxDataset(eval_samples, args.image_width, args.image_height, augment=False)
  train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=device.type == "cuda"
  )
  train_eval_loader = DataLoader(
    train_eval_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=device.type == "cuda"
  )
  eval_loader = DataLoader(
    eval_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=device.type == "cuda"
  )
  model = build_strip_digit_reader_23xx().to(device)
  guard_criterion = nn.CrossEntropyLoss(
    weight=make_guard_weights(train_samples, device),
    label_smoothing=args.label_smoothing
  )
  suffix_criterion = nn.CrossEntropyLoss(
    weight=make_suffix_weights(train_samples, device),
    label_smoothing=args.label_smoothing
  )
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
  best_state = state_dict_to_cpu(model)
  best_epoch = 0
  best_key: tuple[float, float, float] | None = None
  history: list[dict[str, object]] = []
  stale_epochs = 0

  for epoch in range(1, epochs + 1):
    train_metrics = run_epoch(
      model, train_loader, guard_criterion, suffix_criterion, optimizer, device, args.guard_threshold
    )
    train_eval_metrics = run_epoch(
      model, train_eval_loader, guard_criterion, suffix_criterion, None, device, args.guard_threshold
    )
    key = (
      -float(train_eval_metrics["guard_false_positive_count"]),
      float(train_eval_metrics["full_23xx_exact_accuracy_when_guard_accepts"]),
      -float(train_eval_metrics["loss"])
    )
    if best_key is None or key > best_key:
      best_key = key
      best_state = state_dict_to_cpu(model)
      best_epoch = epoch
      stale_epochs = 0
    else:
      stale_epochs += 1
    history.append({
      "epoch": epoch,
      "train": train_metrics,
      "train_eval": train_eval_metrics
    })
    if stale_epochs >= args.patience:
      break

  model.load_state_dict(best_state)
  model.to(device)
  eval_metrics = run_epoch(
    model, eval_loader, guard_criterion, suffix_criterion, None, device, args.guard_threshold
  )
  return model, eval_metrics, history, best_epoch


def make_folds(samples: list[Strip23xxSample], fold_count: int, seed: int) -> list[list[Strip23xxSample]]:
  shuffled = list(samples)
  random.Random(seed).shuffle(shuffled)
  bounded_count = max(1, min(len(shuffled), fold_count))
  return [shuffled[index::bounded_count] for index in range(bounded_count)]


def aggregate_cv(fold_metrics: list[dict[str, object]]) -> dict[str, object]:
  if not fold_metrics:
    return {}
  totals = {
    "sample_count": 0,
    "guard_false_positive_count": 0,
    "guard_false_negative_count": 0,
    "guard_accepted_count": 0
  }
  rates = [
    "guard_accuracy",
    "guard_false_positive_rate",
    "guard_false_negative_rate",
    "suffix_exact_accuracy_when_guard_target_true",
    "full_23xx_exact_accuracy_when_guard_accepts",
    "suffix_position_accuracy"
  ]
  rate_sums = {name: 0.0 for name in rates}
  for row in fold_metrics:
    sample_count = int(row.get("sample_count") or 0)
    totals["sample_count"] += sample_count
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    for name in totals:
      if name == "sample_count":
        continue
      totals[name] += int(metrics.get(name) or 0)
    for name in rates:
      rate_sums[name] += float(metrics.get(name) or 0.0)
  return {
    **totals,
    **{name: rate_sums[name] / len(fold_metrics) for name in rates}
  }


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent
  dataset_root = resolve_path(base_dir, args.dataset_root)
  manifest_path = resolve_path(dataset_root, args.manifest)
  project_root = resolve_path(base_dir, args.project)
  output_path = resolve_path(base_dir, args.copy_to)
  run_dir = project_root / args.name
  run_dir.mkdir(parents=True, exist_ok=True)

  if args.guard_threshold <= 0 or args.guard_threshold >= 1:
    raise ValueError("--guard-threshold must be in (0, 1).")
  if args.epochs <= 0 or args.cv_epochs <= 0:
    raise ValueError("--epochs and --cv-epochs must be positive.")

  set_seed(args.seed)
  device = resolve_device(args.device)
  samples = collect_samples(dataset_root, manifest_path)
  if not samples:
    raise RuntimeError(f"No canonical strip samples found in {manifest_path}")
  positives = sum(sample.guard_label for sample in samples)
  negatives = len(samples) - positives
  guard_data_limited = negatives < 3

  print(f"Samples: {len(samples)}")
  print(f"Guard positives: {positives}")
  print(f"Guard negatives: {negatives}")
  print(f"Guard data limited: {guard_data_limited}")

  cv_rows: list[dict[str, object]] = []
  if not args.skip_cv:
    folds = make_folds(samples, args.cv_folds, args.seed)
    for index, holdout in enumerate(folds, start=1):
      train_samples = [sample for sample in samples if sample not in holdout]
      print(f"CV fold {index}/{len(folds)}: train={len(train_samples)} holdout={len(holdout)}")
      _, metrics, _, best_epoch = fit_model(train_samples, holdout, args, device, args.cv_epochs)
      cv_rows.append({
        "fold": index,
        "sample_count": len(holdout),
        "filenames": [sample.filename for sample in holdout],
        "best_epoch": best_epoch,
        "metrics": metrics
      })

  start_time = time.perf_counter()
  final_model, final_metrics, history, best_epoch = fit_model(samples, samples, args, device, args.epochs)
  elapsed_s = time.perf_counter() - start_time
  final_state = state_dict_to_cpu(final_model)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  checkpoint = {
    "state_dict": final_state,
    "class_names": DEFAULT_CLASS_NAMES,
    "fixed_prefix": FIXED_PREFIX,
    "guard_threshold": args.guard_threshold,
    "image_width": args.image_width,
    "image_height": args.image_height,
    "best_epoch": best_epoch,
    "device": str(device),
    "guard_positive_count": positives,
    "guard_negative_count": negatives,
    "guard_data_limited": guard_data_limited,
    "cv_metrics": aggregate_cv(cv_rows),
    "cv_folds": cv_rows,
    "final_metrics": final_metrics,
    "history": history,
    "training_seconds": elapsed_s,
    "args": vars(args)
  }
  torch.save(checkpoint, output_path)

  summary_path = run_dir / "strip_digit_reader_23xx_summary.json"
  summary_payload = {
    "output_path": str(output_path),
    "best_epoch": best_epoch,
    "training_seconds": elapsed_s,
    "device": str(device),
    "fixed_prefix": FIXED_PREFIX,
    "guard_threshold": args.guard_threshold,
    "guard_positive_count": positives,
    "guard_negative_count": negatives,
    "guard_data_limited": guard_data_limited,
    "cv_metrics": aggregate_cv(cv_rows),
    "cv_folds": cv_rows,
    "final_metrics": final_metrics
  }
  summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
  print(f"Saved 23xx strip reader to: {output_path}")
  print(f"Saved summary to: {summary_path}")
  print(json.dumps(summary_payload["cv_metrics"], indent=2))


if __name__ == "__main__":
  main()
