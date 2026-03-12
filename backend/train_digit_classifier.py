from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
  from .digit_model import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_NUM_CLASSES,
    DEFAULT_IMAGE_SIZE,
    build_digit_cnn,
    prepare_digit_tensor
  )
except ImportError:
  from digit_model import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_NUM_CLASSES,
    DEFAULT_IMAGE_SIZE,
    build_digit_cnn,
    prepare_digit_tensor
  )

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
  path: Path
  label: int


class DigitCellDataset(Dataset):
  def __init__(
    self,
    samples: list[Sample],
    image_size: int,
    augment: bool = False,
    split_jitter_x: float = 0.0,
    split_jitter_y: float = 0.0,
    split_jitter_prob: float = 0.0
  ):
    self.samples = samples
    self.image_size = image_size
    self.augment = augment
    self.split_jitter_x = max(0.0, float(split_jitter_x))
    self.split_jitter_y = max(0.0, float(split_jitter_y))
    self.split_jitter_prob = min(1.0, max(0.0, float(split_jitter_prob)))

  def __len__(self) -> int:
    return len(self.samples)

  def _augment_image(self, image: Image.Image) -> Image.Image:
    if not self.augment:
      return image
    if self.split_jitter_prob > 0 and random.random() < self.split_jitter_prob:
      max_dx = int(round(image.width * self.split_jitter_x))
      max_dy = int(round(image.height * self.split_jitter_y))
      if max_dx > 0 or max_dy > 0:
        dx = random.randint(-max_dx, max_dx) if max_dx > 0 else 0
        dy = random.randint(-max_dy, max_dy) if max_dy > 0 else 0
        shifted = Image.new("L", image.size, color=255)
        src_x0 = max(0, -dx)
        src_y0 = max(0, -dy)
        src_x1 = min(image.width, image.width - dx) if dx >= 0 else image.width
        src_y1 = min(image.height, image.height - dy) if dy >= 0 else image.height
        if src_x1 > src_x0 and src_y1 > src_y0:
          patch = image.crop((src_x0, src_y0, src_x1, src_y1))
          dst_x = max(0, dx)
          dst_y = max(0, dy)
          shifted.paste(patch, (dst_x, dst_y))
          image = shifted
    if random.random() < 0.9:
      if hasattr(Image, "Resampling"):
        image = image.rotate(
          random.uniform(-6.5, 6.5),
          resample=Image.Resampling.BILINEAR,
          fillcolor=255
        )
      else:
        image = image.rotate(
          random.uniform(-6.5, 6.5),
          resample=Image.BILINEAR,
          fillcolor=255
        )
    if random.random() < 0.6:
      image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.3))
    if random.random() < 0.5:
      image = ImageEnhance.Brightness(image).enhance(random.uniform(0.88, 1.15))
    return image

  def __getitem__(self, index: int):
    sample = self.samples[index]
    with Image.open(sample.path) as source:
      image = source.convert("L")
      image = self._augment_image(image)
      tensor = prepare_digit_tensor(image, self.image_size)
    return tensor, sample.label


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Train a per-cell digit classifier for the Jarvis OCR pipeline."
  )
  parser.add_argument(
    "--dataset-root",
    default="data/digit_dataset/sections_labeled",
    help="Dataset root with split folders (train/val/test)."
  )
  parser.add_argument("--epochs", type=int, default=180)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=1.2e-3)
  parser.add_argument("--weight-decay", type=float, default=1.0e-4)
  parser.add_argument("--patience", type=int, default=40)
  parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
  parser.add_argument("--epoch-sample-multiplier", type=int, default=6)
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument(
    "--split-jitter-x",
    type=float,
    default=0.08,
    help="Max horizontal translation as a fraction of sample width for split-jitter augmentation."
  )
  parser.add_argument(
    "--split-jitter-y",
    type=float,
    default=0.08,
    help="Max vertical translation as a fraction of sample height for split-jitter augmentation."
  )
  parser.add_argument(
    "--split-jitter-prob",
    type=float,
    default=0.85,
    help="Probability of applying split-jitter augmentation on each train sample."
  )
  parser.add_argument(
    "--synthetic-root",
    default="",
    help=(
      "Optional synthetic dataset root with train/<digit> folders. "
      "When provided, synthetic samples are mixed into train only."
    )
  )
  parser.add_argument(
    "--synthetic-target-ratio",
    type=float,
    default=0.0,
    help=(
      "Target synthetic-to-real ratio for effective training set (train split only). "
      "Example: 2.0 means up to 2 synthetic samples per real sample."
    )
  )
  parser.add_argument(
    "--synthetic-seed",
    type=int,
    default=42,
    help="Sampling seed used when selecting synthetic samples for the requested ratio."
  )
  parser.add_argument(
    "--synthetic-selection-strategy",
    choices=("balanced", "proportional"),
    default="balanced",
    help=(
      "How to select synthetic train samples when a target ratio is requested. "
      "'balanced' prioritizes underrepresented digits; 'proportional' mirrors the real-label distribution."
    )
  )
  parser.add_argument(
    "--label-smoothing",
    type=float,
    default=0.05,
    help="Cross-entropy label smoothing for classifier training."
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
    "--device",
    default="auto",
    help="Training device: auto, cpu, cuda, cuda:0, or GPU index."
  )
  parser.add_argument("--project", default="runs")
  parser.add_argument("--name", default="digit-classifier")
  parser.add_argument("--copy-to", default="models/digit_classifier.pt")
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


def collect_split_samples(dataset_root: Path, split: str) -> list[Sample]:
  split_dir = dataset_root / split
  if not split_dir.exists():
    return []

  samples: list[Sample] = []
  for digit in range(DEFAULT_NUM_CLASSES):
    digit_dir = split_dir / str(digit)
    if not digit_dir.exists():
      continue
    for image_path in sorted(digit_dir.iterdir()):
      if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        continue
      samples.append(Sample(path=image_path, label=digit))
  return samples


def count_labels(samples: list[Sample]) -> dict[str, int]:
  counts = {str(index): 0 for index in range(DEFAULT_NUM_CLASSES)}
  for sample in samples:
    counts[str(sample.label)] += 1
  return counts


def format_counts(counts: dict[str, int]) -> str:
  return ", ".join([f"{digit}:{counts[str(digit)]}" for digit in range(DEFAULT_NUM_CLASSES)])


def select_synthetic_samples(
  real_samples: list[Sample],
  synthetic_samples: list[Sample],
  target_ratio: float,
  seed: int,
  strategy: str = "balanced"
) -> list[Sample]:
  if not real_samples or not synthetic_samples:
    return []
  if target_ratio <= 0:
    return []

  desired_total = int(round(len(real_samples) * target_ratio))
  if desired_total <= 0:
    return []
  desired_total = min(desired_total, len(synthetic_samples))

  rng = random.Random(seed)
  real_counts = count_labels(real_samples)
  synthetic_by_label: dict[int, list[Sample]] = {index: [] for index in range(DEFAULT_NUM_CLASSES)}
  for sample in synthetic_samples:
    synthetic_by_label[sample.label].append(sample)

  for bucket in synthetic_by_label.values():
    rng.shuffle(bucket)

  if strategy == "balanced":
    current_counts = {
      label: real_counts[str(label)]
      for label in range(DEFAULT_NUM_CLASSES)
    }
    selected_counts = {label: 0 for label in range(DEFAULT_NUM_CLASSES)}
    selected: list[Sample] = []
    target_per_class = int(math.ceil((len(real_samples) + desired_total) / DEFAULT_NUM_CLASSES))

    # First, bring weak classes toward a roughly uniform effective count.
    for label in sorted(range(DEFAULT_NUM_CLASSES), key=lambda value: (current_counts[value], value)):
      pool = synthetic_by_label[label]
      deficit = max(0, target_per_class - current_counts[label])
      take = min(deficit, len(pool), desired_total - len(selected))
      if take <= 0:
        continue
      selected.extend(pool[:take])
      synthetic_by_label[label] = pool[take:]
      selected_counts[label] += take
      current_counts[label] += take
      if len(selected) >= desired_total:
        return selected

    # If synthetic budget remains, keep distributing one sample at a time to the smallest classes.
    while len(selected) < desired_total:
      candidates = [
        label
        for label in range(DEFAULT_NUM_CLASSES)
        if synthetic_by_label[label]
      ]
      if not candidates:
        break
      candidates.sort(key=lambda value: (current_counts[value], selected_counts[value], value))
      chosen = candidates[0]
      selected.append(synthetic_by_label[chosen].pop())
      selected_counts[chosen] += 1
      current_counts[chosen] += 1
    return selected

  total_real = max(1, len(real_samples))
  selected: list[Sample] = []
  leftovers: list[Sample] = []
  for label in range(DEFAULT_NUM_CLASSES):
    pool = synthetic_by_label[label]
    if not pool:
      continue
    quota = int(round(desired_total * (real_counts[str(label)] / total_real)))
    quota = min(quota, len(pool))
    if quota > 0:
      selected.extend(pool[:quota])
      leftovers.extend(pool[quota:])
    else:
      leftovers.extend(pool)

  if len(selected) > desired_total:
    rng.shuffle(selected)
    selected = selected[:desired_total]
    return selected

  if len(selected) < desired_total and leftovers:
    rng.shuffle(leftovers)
    needed = desired_total - len(selected)
    selected.extend(leftovers[:needed])

  return selected


def make_class_weights(train_samples: list[Sample], device: torch.device) -> torch.Tensor:
  label_counts = count_labels(train_samples)
  class_weights = torch.zeros(DEFAULT_NUM_CLASSES, dtype=torch.float32, device=device)
  non_zero_counts = [count for count in label_counts.values() if count > 0]
  if not non_zero_counts:
    return class_weights + 1.0

  mean_count = float(sum(non_zero_counts)) / len(non_zero_counts)
  for class_index in range(DEFAULT_NUM_CLASSES):
    count = label_counts[str(class_index)]
    if count <= 0:
      class_weights[class_index] = 0.0
      continue
    class_weights[class_index] = mean_count / float(count)

  return class_weights


def make_train_sampler(train_samples: list[Sample], multiplier: int) -> WeightedRandomSampler | None:
  if multiplier <= 1:
    return None

  label_counts = count_labels(train_samples)
  sample_weights: list[float] = []
  for sample in train_samples:
    count = label_counts[str(sample.label)]
    sample_weights.append(1.0 / float(max(count, 1)))

  base_count = len(train_samples)
  sampled_count = max(base_count * multiplier, base_count)
  return WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.double),
    num_samples=sampled_count,
    replacement=True
  )


def run_epoch(
  model: nn.Module,
  loader: DataLoader,
  criterion: nn.Module,
  optimizer: torch.optim.Optimizer | None,
  device: torch.device
) -> dict[str, float]:
  training = optimizer is not None
  if training:
    model.train()
  else:
    model.eval()

  total_loss = 0.0
  total_count = 0
  correct = 0

  for inputs, targets in loader:
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    if training:
      optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
      logits = model(inputs)
      loss = criterion(logits, targets)
      if training:
        loss.backward()
        optimizer.step()

    batch_size = targets.shape[0]
    total_loss += float(loss.item()) * batch_size
    total_count += batch_size
    predictions = torch.argmax(logits, dim=1)
    correct += int((predictions == targets).sum().item())

  if total_count == 0:
    return {"loss": 0.0, "accuracy": 0.0}

  return {
    "loss": total_loss / total_count,
    "accuracy": correct / total_count
  }


def state_dict_to_cpu(model: nn.Module) -> dict:
  return {
    key: value.detach().cpu().clone()
    for key, value in model.state_dict().items()
  }


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent
  dataset_root = resolve_path(base_dir, args.dataset_root)
  synthetic_root = resolve_path(base_dir, args.synthetic_root) if args.synthetic_root else None
  project_root = resolve_path(base_dir, args.project)
  output_path = resolve_path(base_dir, args.copy_to)
  run_dir = project_root / args.name
  run_dir.mkdir(parents=True, exist_ok=True)

  if not dataset_root.exists():
    raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
  if synthetic_root and not synthetic_root.exists():
    raise FileNotFoundError(f"Synthetic root not found: {synthetic_root}")
  if args.image_size <= 0:
    raise ValueError("--image-size must be positive.")
  if args.batch_size <= 0:
    raise ValueError("--batch-size must be positive.")
  if args.epochs <= 0:
    raise ValueError("--epochs must be positive.")
  if args.split_jitter_x < 0:
    raise ValueError("--split-jitter-x must be >= 0.")
  if args.split_jitter_y < 0:
    raise ValueError("--split-jitter-y must be >= 0.")
  if args.split_jitter_prob < 0 or args.split_jitter_prob > 1:
    raise ValueError("--split-jitter-prob must be in [0, 1].")
  if args.synthetic_target_ratio < 0:
    raise ValueError("--synthetic-target-ratio must be >= 0.")
  if args.label_smoothing < 0 or args.label_smoothing >= 1:
    raise ValueError("--label-smoothing must be in [0, 1).")

  set_seed(args.seed)
  device = resolve_device(args.device)
  if device.type.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA device requested but CUDA is not available.")

  real_train_samples = collect_split_samples(dataset_root, "train")
  val_samples = collect_split_samples(dataset_root, "val")
  test_samples = collect_split_samples(dataset_root, "test")
  if not real_train_samples:
    raise RuntimeError(f"No train samples found under {dataset_root / 'train'}")

  synthetic_train_all = (
    collect_split_samples(synthetic_root, "train")
    if synthetic_root is not None
    else []
  )
  synthetic_train_selected = select_synthetic_samples(
    real_samples=real_train_samples,
    synthetic_samples=synthetic_train_all,
    target_ratio=args.synthetic_target_ratio,
    seed=args.synthetic_seed,
    strategy=args.synthetic_selection_strategy
  )
  train_samples = [*real_train_samples, *synthetic_train_selected]

  train_counts = count_labels(train_samples)
  real_train_counts = count_labels(real_train_samples)
  synthetic_train_counts = count_labels(synthetic_train_selected)
  val_counts = count_labels(val_samples)
  test_counts = count_labels(test_samples)
  print(f"Dataset: {dataset_root}")
  if synthetic_root is not None:
    print(f"Synthetic root: {synthetic_root}")
    print(
      f"Train real samples: {len(real_train_samples)} "
      f"({format_counts(real_train_counts)})"
    )
    print(
      f"Train synthetic selected: {len(synthetic_train_selected)} / {len(synthetic_train_all)} "
      f"(target_ratio={args.synthetic_target_ratio:.2f}, seed={args.synthetic_seed}, "
      f"strategy={args.synthetic_selection_strategy}) "
      f"({format_counts(synthetic_train_counts)})"
    )
  print(f"Train effective samples: {len(train_samples)} ({format_counts(train_counts)})")
  print(f"Val samples: {len(val_samples)} ({format_counts(val_counts)})")
  print(f"Test samples: {len(test_samples)} ({format_counts(test_counts)})")

  train_dataset = DigitCellDataset(
    train_samples,
    image_size=args.image_size,
    augment=True,
    split_jitter_x=args.split_jitter_x,
    split_jitter_y=args.split_jitter_y,
    split_jitter_prob=args.split_jitter_prob
  )
  val_dataset = DigitCellDataset(val_samples, image_size=args.image_size, augment=False)
  test_dataset = DigitCellDataset(test_samples, image_size=args.image_size, augment=False)

  train_sampler = make_train_sampler(train_samples, args.epoch_sample_multiplier)
  train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=train_sampler,
    shuffle=train_sampler is None,
    num_workers=args.num_workers,
    pin_memory=device.type == "cuda"
  )
  val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=device.type == "cuda"
  ) if len(val_dataset) else None
  test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=device.type == "cuda"
  ) if len(test_dataset) else None

  model = build_digit_cnn(DEFAULT_NUM_CLASSES).to(device)
  class_weights = make_class_weights(train_samples, device)
  criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=args.label_smoothing
  )
  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    weight_decay=args.weight_decay
  )
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=8,
    threshold=1.0e-4,
    min_lr=1.0e-5
  )

  best_epoch = 0
  best_metric = float("inf")
  best_state = state_dict_to_cpu(model)
  epochs_without_improvement = 0
  history: list[dict[str, float | int]] = []
  start_time = time.perf_counter()

  for epoch in range(1, args.epochs + 1):
    train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
    val_metrics = run_epoch(model, val_loader, criterion, None, device) if val_loader else None
    monitored_loss = val_metrics["loss"] if val_metrics else train_metrics["loss"]
    scheduler.step(monitored_loss)

    improved = monitored_loss < (best_metric - 1.0e-4)
    if improved:
      best_metric = monitored_loss
      best_epoch = epoch
      best_state = state_dict_to_cpu(model)
      epochs_without_improvement = 0
    else:
      epochs_without_improvement += 1

    current_lr = optimizer.param_groups[0]["lr"]
    epoch_row = {
      "epoch": epoch,
      "learning_rate": current_lr,
      "train_loss": train_metrics["loss"],
      "train_accuracy": train_metrics["accuracy"],
      "val_loss": val_metrics["loss"] if val_metrics else None,
      "val_accuracy": val_metrics["accuracy"] if val_metrics else None
    }
    history.append(epoch_row)

    train_acc = train_metrics["accuracy"] * 100
    train_loss = train_metrics["loss"]
    if val_metrics:
      val_acc = val_metrics["accuracy"] * 100
      val_loss = val_metrics["loss"]
      print(
        f"[{epoch:03d}] lr={current_lr:.6f} "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.1f}% "
        f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}%"
      )
    else:
      print(
        f"[{epoch:03d}] lr={current_lr:.6f} "
        f"train_loss={train_loss:.4f} train_acc={train_acc:.1f}%"
      )

    if val_loader and epochs_without_improvement >= args.patience:
      print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
      break

  model.load_state_dict(best_state)
  model.to(device)
  final_train = run_epoch(model, train_loader, criterion, None, device)
  final_val = run_epoch(model, val_loader, criterion, None, device) if val_loader else None
  final_test = run_epoch(model, test_loader, criterion, None, device) if test_loader else None
  elapsed_s = time.perf_counter() - start_time

  output_path.parent.mkdir(parents=True, exist_ok=True)
  checkpoint = {
    "state_dict": best_state,
    "class_names": DEFAULT_CLASS_NAMES,
    "image_size": args.image_size,
    "best_epoch": best_epoch,
    "device": str(device),
    "train_real_counts": real_train_counts,
    "train_synthetic_counts": synthetic_train_counts,
    "train_counts": train_counts,
    "val_counts": val_counts,
    "test_counts": test_counts,
    "final_metrics": {
      "train": final_train,
      "val": final_val,
      "test": final_test
    },
    "history": history,
    "training_seconds": elapsed_s,
    "args": vars(args)
  }
  torch.save(checkpoint, output_path)

  summary_path = run_dir / "digit_classifier_summary.json"
  summary_payload = {
    "output_path": str(output_path),
    "best_epoch": best_epoch,
    "training_seconds": elapsed_s,
    "device": str(device),
    "class_names": DEFAULT_CLASS_NAMES,
    "synthetic_root": str(synthetic_root) if synthetic_root is not None else None,
    "synthetic_target_ratio": args.synthetic_target_ratio,
    "synthetic_seed": args.synthetic_seed,
    "train_real_counts": real_train_counts,
    "train_synthetic_counts": synthetic_train_counts,
    "train_counts": train_counts,
    "val_counts": val_counts,
    "test_counts": test_counts,
    "final_metrics": {
      "train": final_train,
      "val": final_val,
      "test": final_test
    },
    "history": history
  }
  summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

  print(f"Best epoch: {best_epoch}")
  print(f"Saved classifier to: {output_path}")
  print(f"Saved summary to: {summary_path}")
  if final_val:
    print(
      "Validation: "
      f"loss={final_val['loss']:.4f} "
      f"acc={final_val['accuracy'] * 100:.1f}%"
    )
  if final_test:
    print(
      "Test: "
      f"loss={final_test['loss']:.4f} "
      f"acc={final_test['accuracy'] * 100:.1f}%"
    )


if __name__ == "__main__":
  main()
