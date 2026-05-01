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
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_STRIP_HEIGHT,
    DEFAULT_STRIP_WIDTH,
    build_strip_digit_reader,
    prepare_strip_tensor
  )
except ImportError:
  from strip_digit_model import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_NUM_CLASSES,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_STRIP_HEIGHT,
    DEFAULT_STRIP_WIDTH,
    build_strip_digit_reader,
    prepare_strip_tensor
  )


@dataclass
class StripSample:
  path: Path
  reading: str
  filename: str
  split: str


class StripDigitDataset(Dataset):
  def __init__(
    self,
    samples: list[StripSample],
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
        (
          max(1, int(round(image.width * scale))),
          max(1, int(round(image.height * scale)))
        ),
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
      image = image.rotate(
        random.uniform(-3.0, 3.0),
        resample=resample,
        fillcolor=255
      )

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
      tensor = prepare_strip_tensor(
        image,
        width=self.image_width,
        height=self.image_height
      )
    target = torch.tensor([int(digit) for digit in sample.reading], dtype=torch.long)
    return tensor, target


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Train a fixed four-digit whole-strip reader for Jarvis OCR."
  )
  parser.add_argument("--dataset-root", default="data/digit_dataset")
  parser.add_argument("--manifest", default="manifests/canonical_windows.csv")
  parser.add_argument("--epochs", type=int, default=220)
  parser.add_argument("--batch-size", type=int, default=8)
  parser.add_argument("--learning-rate", type=float, default=8.0e-4)
  parser.add_argument("--weight-decay", type=float, default=1.0e-4)
  parser.add_argument("--patience", type=int, default=55)
  parser.add_argument("--image-width", type=int, default=DEFAULT_STRIP_WIDTH)
  parser.add_argument("--image-height", type=int, default=DEFAULT_STRIP_HEIGHT)
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument("--label-smoothing", type=float, default=0.04)
  parser.add_argument("--selection-split", choices=["auto", "train", "val"], default="auto")
  parser.add_argument("--min-val-samples-for-selection", type=int, default=4)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="auto")
  parser.add_argument("--project", default="runs")
  parser.add_argument("--name", default="strip-digit-reader")
  parser.add_argument("--copy-to", default="models/digit_strip_reader.pt")
  parser.add_argument("--init-checkpoint", default="")
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


def collect_samples(dataset_root: Path, manifest_path: Path) -> dict[str, list[StripSample]]:
  samples = {"train": [], "val": [], "test": []}
  with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      split = (row.get("split") or "").strip()
      reading = (row.get("reading") or "").strip()
      relative_path = (row.get("canonical_window_path") or "").strip()
      filename = (row.get("filename") or "").strip()
      if split not in samples:
        continue
      if len(reading) != DEFAULT_SEQUENCE_LENGTH or not reading.isdigit():
        continue
      if not relative_path:
        continue
      path = dataset_root / relative_path
      if not path.exists():
        continue
      samples[split].append(StripSample(path=path, reading=reading, filename=filename, split=split))
  return samples


def count_position_labels(samples: list[StripSample]) -> list[dict[str, int]]:
  counts = [{str(index): 0 for index in range(DEFAULT_NUM_CLASSES)} for _ in range(DEFAULT_SEQUENCE_LENGTH)]
  for sample in samples:
    for position, digit in enumerate(sample.reading):
      counts[position][digit] += 1
  return counts


def make_class_weights(samples: list[StripSample], device: torch.device) -> torch.Tensor:
  total_counts = {str(index): 0 for index in range(DEFAULT_NUM_CLASSES)}
  for sample in samples:
    for digit in sample.reading:
      total_counts[digit] += 1
  non_zero_counts = [count for count in total_counts.values() if count > 0]
  if not non_zero_counts:
    return torch.ones(DEFAULT_NUM_CLASSES, dtype=torch.float32, device=device)
  mean_count = float(sum(non_zero_counts)) / len(non_zero_counts)
  weights = torch.zeros(DEFAULT_NUM_CLASSES, dtype=torch.float32, device=device)
  for class_index in range(DEFAULT_NUM_CLASSES):
    count = total_counts[str(class_index)]
    weights[class_index] = 0.0 if count <= 0 else mean_count / float(count)
  return weights


def serialize_confusion(confusion: np.ndarray) -> list[dict[str, object]]:
  rows = []
  for position in range(confusion.shape[0]):
    matrix = confusion[position]
    rows.append({
      "position": position,
      "matrix": matrix.astype(int).tolist()
    })
  return rows


def run_epoch(
  model: nn.Module,
  loader: DataLoader,
  criterion: nn.Module,
  optimizer: torch.optim.Optimizer | None,
  device: torch.device,
  include_confusion: bool = False
) -> dict[str, object]:
  training = optimizer is not None
  model.train() if training else model.eval()

  total_loss = 0.0
  total_count = 0
  exact_correct = 0
  position_correct = np.zeros(DEFAULT_SEQUENCE_LENGTH, dtype=np.int64)
  confusion = np.zeros(
    (DEFAULT_SEQUENCE_LENGTH, DEFAULT_NUM_CLASSES, DEFAULT_NUM_CLASSES),
    dtype=np.int64
  )

  for inputs, targets in loader:
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    if training:
      optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
      logits = model(inputs)
      loss = criterion(
        logits.reshape(-1, DEFAULT_NUM_CLASSES),
        targets.reshape(-1)
      )
      if training:
        loss.backward()
        optimizer.step()

    batch_size = targets.shape[0]
    total_loss += float(loss.item()) * batch_size
    total_count += batch_size
    predictions = torch.argmax(logits, dim=2)
    matches = predictions == targets
    exact_correct += int(matches.all(dim=1).sum().item())
    position_correct += matches.sum(dim=0).detach().cpu().numpy().astype(np.int64)

    if include_confusion:
      pred_cpu = predictions.detach().cpu().numpy()
      target_cpu = targets.detach().cpu().numpy()
      for row_index in range(batch_size):
        for position in range(DEFAULT_SEQUENCE_LENGTH):
          confusion[position, target_cpu[row_index, position], pred_cpu[row_index, position]] += 1

  if total_count == 0:
    payload: dict[str, object] = {
      "loss": 0.0,
      "exact_match_accuracy": 0.0,
      "position_accuracy": 0.0,
      "per_position_accuracy": [0.0 for _ in range(DEFAULT_SEQUENCE_LENGTH)]
    }
  else:
    per_position = [
      float(position_correct[position] / total_count)
      for position in range(DEFAULT_SEQUENCE_LENGTH)
    ]
    payload = {
      "loss": total_loss / total_count,
      "exact_match_accuracy": exact_correct / total_count,
      "position_accuracy": float(position_correct.sum() / (total_count * DEFAULT_SEQUENCE_LENGTH)),
      "per_position_accuracy": per_position
    }
  if include_confusion:
    payload["confusion_by_position"] = serialize_confusion(confusion)
  return payload


def state_dict_to_cpu(model: nn.Module) -> dict:
  return {
    key: value.detach().cpu().clone()
    for key, value in model.state_dict().items()
  }


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent
  dataset_root = resolve_path(base_dir, args.dataset_root)
  manifest_path = resolve_path(dataset_root, args.manifest)
  project_root = resolve_path(base_dir, args.project)
  output_path = resolve_path(base_dir, args.copy_to)
  init_checkpoint = resolve_path(base_dir, args.init_checkpoint) if args.init_checkpoint else None
  run_dir = project_root / args.name
  run_dir.mkdir(parents=True, exist_ok=True)

  if not dataset_root.exists():
    raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
  if not manifest_path.exists():
    raise FileNotFoundError(f"Canonical windows manifest not found: {manifest_path}")
  if args.image_width <= 0 or args.image_height <= 0:
    raise ValueError("--image-width and --image-height must be positive.")
  if args.batch_size <= 0:
    raise ValueError("--batch-size must be positive.")
  if args.epochs <= 0:
    raise ValueError("--epochs must be positive.")
  if args.label_smoothing < 0 or args.label_smoothing >= 1:
    raise ValueError("--label-smoothing must be in [0, 1).")
  if args.min_val_samples_for_selection < 1:
    raise ValueError("--min-val-samples-for-selection must be positive.")
  if init_checkpoint and not init_checkpoint.exists():
    raise FileNotFoundError(f"Init checkpoint not found: {init_checkpoint}")

  set_seed(args.seed)
  device = resolve_device(args.device)
  if device.type.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA device requested but CUDA is not available.")

  samples = collect_samples(dataset_root, manifest_path)
  train_samples = samples["train"]
  val_samples = samples["val"]
  test_samples = samples["test"]
  if not train_samples:
    raise RuntimeError(f"No train strip samples found in {manifest_path}")

  print(f"Dataset: {dataset_root}")
  print(f"Manifest: {manifest_path}")
  print(f"Train samples: {len(train_samples)}")
  print(f"Val samples: {len(val_samples)}")
  print(f"Test samples: {len(test_samples)}")
  if args.selection_split == "auto":
    selection_split = "val" if len(val_samples) >= args.min_val_samples_for_selection else "train"
  else:
    selection_split = args.selection_split
  if selection_split == "val" and not val_samples:
    raise RuntimeError("--selection-split val requested, but no validation samples were found.")
  print(f"Checkpoint selection split: {selection_split}")

  train_dataset = StripDigitDataset(
    train_samples,
    image_width=args.image_width,
    image_height=args.image_height,
    augment=True
  )
  train_eval_dataset = StripDigitDataset(
    train_samples,
    image_width=args.image_width,
    image_height=args.image_height,
    augment=False
  )
  val_dataset = StripDigitDataset(
    val_samples,
    image_width=args.image_width,
    image_height=args.image_height,
    augment=False
  )
  test_dataset = StripDigitDataset(
    test_samples,
    image_width=args.image_width,
    image_height=args.image_height,
    augment=False
  )

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

  model = build_strip_digit_reader(
    sequence_length=DEFAULT_SEQUENCE_LENGTH,
    num_classes=DEFAULT_NUM_CLASSES
  ).to(device)
  if init_checkpoint is not None:
    payload = torch.load(str(init_checkpoint), map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
      raise RuntimeError(f"Init checkpoint missing state_dict: {init_checkpoint}")
    model.load_state_dict(payload["state_dict"], strict=True)

  criterion = nn.CrossEntropyLoss(
    weight=make_class_weights(train_samples, device),
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
    patience=10,
    threshold=1.0e-4,
    min_lr=1.0e-5
  )

  best_epoch = 0
  best_metric = float("inf")
  best_selection_key: tuple[float, float, float] | None = None
  best_state = state_dict_to_cpu(model)
  epochs_without_improvement = 0
  history: list[dict[str, object]] = []
  start_time = time.perf_counter()

  for epoch in range(1, args.epochs + 1):
    train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
    train_eval_metrics = run_epoch(model, train_eval_loader, criterion, None, device) \
      if selection_split == "train" else None
    val_metrics = run_epoch(model, val_loader, criterion, None, device) if val_loader else None
    monitored_metrics = val_metrics if selection_split == "val" and val_metrics else (train_eval_metrics or train_metrics)
    monitored_loss = float(monitored_metrics["loss"])
    scheduler.step(monitored_loss)

    selection_key = (
      float(monitored_metrics["exact_match_accuracy"]),
      float(monitored_metrics["position_accuracy"]),
      -monitored_loss
    )
    if selection_split == "train":
      improved = best_selection_key is None or selection_key > best_selection_key
    else:
      improved = monitored_loss < (best_metric - 1.0e-4)
    if improved:
      best_metric = monitored_loss
      best_selection_key = selection_key
      best_epoch = epoch
      best_state = state_dict_to_cpu(model)
      epochs_without_improvement = 0
    else:
      epochs_without_improvement += 1

    current_lr = optimizer.param_groups[0]["lr"]
    epoch_row = {
      "epoch": epoch,
      "learning_rate": current_lr,
      "selection_split": selection_split,
      "selection_loss": monitored_loss,
      "selection_exact_match_accuracy": monitored_metrics["exact_match_accuracy"],
      "selection_position_accuracy": monitored_metrics["position_accuracy"],
      "train_loss": train_metrics["loss"],
      "train_exact_match_accuracy": train_metrics["exact_match_accuracy"],
      "train_position_accuracy": train_metrics["position_accuracy"],
      "train_eval_loss": train_eval_metrics["loss"] if train_eval_metrics else None,
      "train_eval_exact_match_accuracy": train_eval_metrics["exact_match_accuracy"] if train_eval_metrics else None,
      "train_eval_position_accuracy": train_eval_metrics["position_accuracy"] if train_eval_metrics else None,
      "val_loss": val_metrics["loss"] if val_metrics else None,
      "val_exact_match_accuracy": val_metrics["exact_match_accuracy"] if val_metrics else None,
      "val_position_accuracy": val_metrics["position_accuracy"] if val_metrics else None
    }
    history.append(epoch_row)

    train_exact = float(train_metrics["exact_match_accuracy"]) * 100
    train_pos = float(train_metrics["position_accuracy"]) * 100
    train_eval_suffix = ""
    if train_eval_metrics:
      train_eval_suffix = (
        f" train_eval_exact={float(train_eval_metrics['exact_match_accuracy']) * 100:.1f}%"
        f" train_eval_pos={float(train_eval_metrics['position_accuracy']) * 100:.1f}%"
      )
    if val_metrics:
      val_exact = float(val_metrics["exact_match_accuracy"]) * 100
      val_pos = float(val_metrics["position_accuracy"]) * 100
      print(
        f"[{epoch:03d}] lr={current_lr:.6f} "
        f"train_loss={float(train_metrics['loss']):.4f} "
        f"train_exact={train_exact:.1f}% train_pos={train_pos:.1f}% "
        f"{train_eval_suffix} "
        f"val_loss={float(val_metrics['loss']):.4f} "
        f"val_exact={val_exact:.1f}% val_pos={val_pos:.1f}%"
      )
    else:
      print(
        f"[{epoch:03d}] lr={current_lr:.6f} "
        f"train_loss={float(train_metrics['loss']):.4f} "
        f"train_exact={train_exact:.1f}% train_pos={train_pos:.1f}% "
        f"{train_eval_suffix}"
      )

    if val_loader and epochs_without_improvement >= args.patience:
      print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
      break

  model.load_state_dict(best_state)
  model.to(device)
  final_train = run_epoch(model, train_loader, criterion, None, device, include_confusion=True)
  final_val = run_epoch(model, val_loader, criterion, None, device, include_confusion=True) if val_loader else None
  final_test = run_epoch(model, test_loader, criterion, None, device, include_confusion=True) if test_loader else None
  elapsed_s = time.perf_counter() - start_time

  train_counts = count_position_labels(train_samples)
  val_counts = count_position_labels(val_samples)
  test_counts = count_position_labels(test_samples)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  checkpoint = {
    "state_dict": best_state,
    "class_names": DEFAULT_CLASS_NAMES,
    "sequence_length": DEFAULT_SEQUENCE_LENGTH,
    "image_width": args.image_width,
    "image_height": args.image_height,
    "best_epoch": best_epoch,
    "device": str(device),
    "train_counts_by_position": train_counts,
    "val_counts_by_position": val_counts,
    "test_counts_by_position": test_counts,
    "final_metrics": {
      "train": final_train,
      "val": final_val,
      "test": final_test
    },
    "history": history,
    "training_seconds": elapsed_s,
    "selection_split": selection_split,
    "min_val_samples_for_selection": args.min_val_samples_for_selection,
    "args": vars(args)
  }
  torch.save(checkpoint, output_path)

  summary_path = run_dir / "strip_digit_reader_summary.json"
  summary_payload = {
    "output_path": str(output_path),
    "best_epoch": best_epoch,
    "training_seconds": elapsed_s,
    "device": str(device),
    "selection_split": selection_split,
    "min_val_samples_for_selection": args.min_val_samples_for_selection,
    "class_names": DEFAULT_CLASS_NAMES,
    "sequence_length": DEFAULT_SEQUENCE_LENGTH,
    "image_width": args.image_width,
    "image_height": args.image_height,
    "train_counts_by_position": train_counts,
    "val_counts_by_position": val_counts,
    "test_counts_by_position": test_counts,
    "final_metrics": {
      "train": final_train,
      "val": final_val,
      "test": final_test
    },
    "history": history
  }
  summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

  print(f"Best epoch: {best_epoch}")
  print(f"Saved strip reader to: {output_path}")
  print(f"Saved summary to: {summary_path}")
  for split_name, metrics in (("train", final_train), ("val", final_val), ("test", final_test)):
    if metrics:
      print(
        f"{split_name}: loss={float(metrics['loss']):.4f} "
        f"exact={float(metrics['exact_match_accuracy']) * 100:.1f}% "
        f"position={float(metrics['position_accuracy']) * 100:.1f}%"
      )


if __name__ == "__main__":
  main()
