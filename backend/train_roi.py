from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import yaml

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HEAVY_AUGMENT_KWARGS = {
  "degrees": 180.0,
  "translate": 0.2,
  "scale": 0.5,
  "shear": 15.0,
  "perspective": 0.001,
  "flipud": 0.5,
  "fliplr": 0.5,
  "mosaic": 1.0,
  "mixup": 0.2
}


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
  parser.add_argument(
    "--rotation-angles",
    default="",
    help=(
      "Comma-separated clockwise right-angle rotations to materialize in the train split "
      "(allowed: 0,90,180,270,360). Example: 90,180,270,360"
    )
  )
  parser.add_argument(
    "--heavy-augment",
    action="store_true",
    help="Enable aggressive online augmentation (rotate/translate/scale/shear/flip/mosaic/mixup)."
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


def clamp01(value: float) -> float:
  return max(0.0, min(1.0, value))


def parse_rotation_angles(raw: str) -> list[tuple[int, int]]:
  value = (raw or "").strip()
  if not value:
    return []
  allowed = {0, 90, 180, 270, 360}
  result: list[tuple[int, int]] = []
  seen: set[int] = set()
  for token in value.split(","):
    angle_text = token.strip()
    if not angle_text:
      continue
    try:
      requested = int(angle_text)
    except ValueError as error:
      raise ValueError(f"Invalid angle: {angle_text}") from error
    if requested not in allowed:
      raise ValueError(f"Unsupported angle {requested}. Allowed: 0,90,180,270,360")
    if requested in seen:
      continue
    seen.add(requested)
    result.append((requested, requested % 360))
  return result


def rotate_yolo_bbox(xc: float, yc: float, width: float, height: float, angle_norm: int) -> tuple[float, float, float, float]:
  if angle_norm == 0:
    return xc, yc, width, height
  if angle_norm == 90:
    return 1 - yc, xc, height, width
  if angle_norm == 180:
    return 1 - xc, 1 - yc, width, height
  if angle_norm == 270:
    return yc, 1 - xc, height, width
  raise ValueError(f"Unexpected normalized angle: {angle_norm}")


def write_rotated_label(source_label: Path, target_label: Path, angle_norm: int) -> None:
  output_lines: list[str] = []
  for line in source_label.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if not stripped:
      continue
    parts = stripped.split()
    if len(parts) != 5:
      raise ValueError(
        f"Only box labels with 5 columns are supported for rotation augmentation: {source_label}"
      )
    class_id = parts[0]
    xc, yc, width, height = (float(parts[i]) for i in range(1, 5))
    rot_xc, rot_yc, rot_w, rot_h = rotate_yolo_bbox(
      clamp01(xc),
      clamp01(yc),
      clamp01(width),
      clamp01(height),
      angle_norm
    )
    output_lines.append(
      f"{class_id} {clamp01(rot_xc):.6f} {clamp01(rot_yc):.6f} {clamp01(rot_w):.6f} {clamp01(rot_h):.6f}"
    )
  target_label.write_text("\n".join(output_lines) + "\n", encoding="utf-8")


def rotate_and_save_image(source_image: Path, target_image: Path, angle_norm: int) -> None:
  try:
    from PIL import Image
  except ImportError as error:
    raise RuntimeError("Pillow is required for --rotation-angles augmentation.") from error

  with Image.open(source_image) as image:
    if angle_norm == 0:
      rotated = image.copy()
    else:
      rotated = image.rotate(-angle_norm, expand=True)
    if target_image.suffix.lower() in {".jpg", ".jpeg"}:
      rotated.save(target_image, quality=95)
    else:
      rotated.save(target_image)


def build_rotated_dataset(resolved_data_path: Path, rotations: list[tuple[int, int]]) -> tuple[Path, Path, int]:
  payload = yaml.safe_load(resolved_data_path.read_text(encoding="utf-8")) or {}
  if not isinstance(payload, dict):
    raise ValueError(f"Invalid dataset yaml: {resolved_data_path}")

  dataset_root_raw = payload.get("path")
  train_images_raw = payload.get("train")
  if not isinstance(dataset_root_raw, str) or not dataset_root_raw.strip():
    raise ValueError(f"Dataset yaml is missing a valid 'path': {resolved_data_path}")
  if not isinstance(train_images_raw, str) or not train_images_raw.strip():
    raise ValueError(f"Dataset yaml is missing a valid 'train' entry: {resolved_data_path}")

  dataset_root = Path(dataset_root_raw)
  if not dataset_root.is_absolute():
    dataset_root = (resolved_data_path.parent / dataset_root).resolve()
  if not dataset_root.exists():
    raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

  train_images_rel = Path(train_images_raw)
  train_images_dir = (dataset_root / train_images_rel).resolve()
  split_name = train_images_rel.name
  train_labels_dir = (dataset_root / "labels" / split_name).resolve()
  if not train_images_dir.exists():
    raise FileNotFoundError(f"Train image directory not found: {train_images_dir}")
  if not train_labels_dir.exists():
    raise FileNotFoundError(f"Train label directory not found: {train_labels_dir}")

  temp_root = Path(tempfile.mkdtemp(prefix="roi_rot_aug_"))
  try:
    augmented_dataset_root = temp_root / dataset_root.name
    shutil.copytree(dataset_root, augmented_dataset_root)

    augmented_images_dir = (augmented_dataset_root / train_images_rel).resolve()
    augmented_labels_dir = (augmented_dataset_root / "labels" / split_name).resolve()
    base_images = sorted(
      path
      for path in augmented_images_dir.iterdir()
      if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )

    generated_count = 0
    for image_path in base_images:
      label_path = augmented_labels_dir / f"{image_path.stem}.txt"
      if not label_path.exists():
        raise FileNotFoundError(f"Missing label for train image: {label_path}")
      for requested_angle, normalized_angle in rotations:
        suffix = f"_rot{requested_angle}"
        rotated_image = image_path.with_name(f"{image_path.stem}{suffix}{image_path.suffix}")
        rotated_label = label_path.with_name(f"{label_path.stem}{suffix}.txt")
        rotate_and_save_image(image_path, rotated_image, normalized_angle)
        write_rotated_label(label_path, rotated_label, normalized_angle)
        generated_count += 1

    payload["path"] = str(augmented_dataset_root.resolve())
    augmented_yaml_path = temp_root / "dataset_rot_aug.yaml"
    augmented_yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return augmented_yaml_path, temp_root, generated_count
  except Exception:
    shutil.rmtree(temp_root, ignore_errors=True)
    raise


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent
  data_path = resolve_path(base_dir, args.data)
  project_path = resolve_path(base_dir, args.project)
  copy_to_path = resolve_path(base_dir, args.copy_to)
  device = resolve_device(args.device)
  rotations = parse_rotation_angles(args.rotation_angles)

  if not data_path.exists():
    raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

  temporary_files: list[Path] = []
  temporary_dirs: list[Path] = []
  try:
    resolved_data_path = resolve_dataset_yaml(data_path)
    if resolved_data_path != data_path:
      temporary_files.append(resolved_data_path)
    train_data_path = resolved_data_path
    if rotations:
      train_data_path, augmented_root, generated = build_rotated_dataset(resolved_data_path, rotations)
      temporary_files.append(train_data_path)
      temporary_dirs.append(augmented_root)
      print(
        f"Rotation augmentation enabled: base train images expanded with {generated} generated variants "
        f"for angles {[requested for requested, _ in rotations]}"
      )

    try:
      from ultralytics import YOLO
    except ImportError as error:
      raise RuntimeError("ultralytics is required. Install backend/requirements.txt first.") from error

    model = YOLO(args.base_model)
    train_kwargs = dict(HEAVY_AUGMENT_KWARGS) if args.heavy_augment else {}
    if args.heavy_augment:
      print(f"Heavy online augmentation enabled: {train_kwargs}")

    model.train(
      data=str(train_data_path),
      epochs=args.epochs,
      imgsz=args.imgsz,
      batch=args.batch,
      patience=args.patience,
      workers=args.workers,
      project=str(project_path),
      name=args.name,
      device=device,
      **train_kwargs
    )

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
  finally:
    for path in temporary_files:
      if path.exists():
        path.unlink(missing_ok=True)
    for directory in temporary_dirs:
      if directory.exists():
        shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
  main()
