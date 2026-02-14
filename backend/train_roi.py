from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Fine-tune a pretrained YOLO model to detect the water meter digit window."
  )
  parser.add_argument(
    "--data",
    default="data/roi_dataset.yaml",
    help="Path to YOLO dataset yaml (relative to backend/ by default)."
  )
  parser.add_argument(
    "--base-model",
    default="yolov8n.pt",
    help="Pretrained model checkpoint to fine-tune (e.g. yolov8n.pt, yolov8s.pt)."
  )
  parser.add_argument("--epochs", type=int, default=120)
  parser.add_argument("--imgsz", type=int, default=960)
  parser.add_argument("--batch", type=int, default=8)
  parser.add_argument("--patience", type=int, default=30)
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument(
    "--device",
    default="auto",
    help="Training device: cpu, auto, 0, or cuda:0 (default: auto)."
  )
  parser.add_argument("--project", default="runs")
  parser.add_argument("--name", default="roi-finetune")
  parser.add_argument("--copy-to", default="models/roi.pt", help="Where to copy best.pt after training.")
  return parser.parse_args()


def resolve_path(base_dir: Path, value: str) -> Path:
  path = Path(value)
  if path.is_absolute():
    return path
  return (base_dir / path).resolve()


def resolve_dataset_yaml(data_path: Path) -> Path:
  payload = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
  if not isinstance(payload, dict):
    return data_path

  dataset_root = payload.get("path")
  if not dataset_root or not isinstance(dataset_root, str):
    return data_path

  root_path = Path(dataset_root)
  if root_path.is_absolute():
    return data_path

  payload["path"] = str((data_path.parent / root_path).resolve())
  with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
    yaml.safe_dump(payload, handle, sort_keys=False)
    return Path(handle.name)


def resolve_device(device: str) -> str | None:
  normalized = device.strip()
  if not normalized:
    return None
  if normalized.lower() == "auto":
    return None
  return normalized


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent
  data_path = resolve_path(base_dir, args.data)
  resolved_data_path = resolve_dataset_yaml(data_path)
  project_path = resolve_path(base_dir, args.project)
  copy_to_path = resolve_path(base_dir, args.copy_to)
  device = resolve_device(args.device)

  if not data_path.exists():
    raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

  try:
    from ultralytics import YOLO
  except ImportError as error:
    raise RuntimeError("ultralytics is required. Install backend/requirements.txt first.") from error

  model = YOLO(args.base_model)
  try:
    model.train(
      data=str(resolved_data_path),
      epochs=args.epochs,
      imgsz=args.imgsz,
      batch=args.batch,
      patience=args.patience,
      workers=args.workers,
      project=str(project_path),
      name=args.name,
      device=device
    )
  finally:
    if resolved_data_path != data_path and resolved_data_path.exists():
      resolved_data_path.unlink(missing_ok=True)

  trainer = getattr(model, "trainer", None)
  if trainer is None:
    raise RuntimeError("Training completed but model.trainer is unavailable; cannot locate best checkpoint.")

  best_checkpoint = getattr(trainer, "best", None)
  if best_checkpoint:
    best_path = Path(best_checkpoint)
  else:
    save_dir = getattr(trainer, "save_dir", None)
    if not save_dir:
      raise RuntimeError("Training completed but trainer metadata does not include a checkpoint path.")
    best_path = Path(save_dir) / "weights" / "best.pt"

  if not best_path.exists():
    raise FileNotFoundError(f"Training completed but no best.pt found at {best_path}")

  copy_to_path.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(best_path, copy_to_path)
  print(f"Best model: {best_path}")
  print(f"Copied to: {copy_to_path}")


if __name__ == "__main__":
  main()
