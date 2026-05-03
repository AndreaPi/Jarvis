from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DIGITS = [str(index) for index in range(10)]


@dataclass
class SourceSample:
  path: Path
  digit: str


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      "Generate synthetic train-only digit sections from labeled train patches. "
      "Val/test remain real-only."
    )
  )
  parser.add_argument(
    "--dataset-root",
    default="data/digit_dataset/sections_labeled",
    help="Root with labeled split folders (train/val/test)."
  )
  parser.add_argument(
    "--output-root",
    default="data/digit_dataset/sections_synthetic",
    help="Output root for synthetic train sections."
  )
  parser.add_argument(
    "--direct-per-real",
    type=int,
    default=6,
    help="Number of directly-augmented synthetic cells per real train cell."
  )
  parser.add_argument(
    "--compose-window-count",
    type=int,
    default=180,
    help=(
      "Optional number of synthetic 4-digit windows to compose from train patches; "
      "each window emits 4 equispaced cells."
    )
  )
  parser.add_argument(
    "--save-composed-windows",
    action="store_true",
    help="Persist synthetic composed windows for QA/debug under output-root/windows."
  )
  parser.add_argument(
    "--clean",
    action="store_true",
    help="Delete existing output-root before generation."
  )
  parser.add_argument("--seed", type=int, default=42)
  return parser.parse_args()


def resolve_path(base_dir: Path, raw: str) -> Path:
  path = Path(raw)
  if path.is_absolute():
    return path
  return (base_dir / path).resolve()


def collect_train_samples(dataset_root: Path) -> list[SourceSample]:
  split_dir = dataset_root / "train"
  samples: list[SourceSample] = []
  for digit in DEFAULT_DIGITS:
    digit_dir = split_dir / digit
    if not digit_dir.exists():
      continue
    for image_path in sorted(digit_dir.iterdir()):
      if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        continue
      samples.append(SourceSample(path=image_path, digit=digit))
  return samples


def ensure_output_structure(output_root: Path) -> None:
  for digit in DEFAULT_DIGITS:
    (output_root / "train" / digit).mkdir(parents=True, exist_ok=True)
  (output_root / "manifests").mkdir(parents=True, exist_ok=True)


def apply_brightness_contrast(image: Image.Image, rng: random.Random) -> Image.Image:
  image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.28))
  image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.86, 1.18))
  return image


def apply_noise(image: Image.Image, rng: random.Random) -> Image.Image:
  sigma = rng.uniform(2.0, 13.0)
  pixels = np.asarray(image, dtype=np.float32)
  noise = rng.gauss(0.0, sigma)
  if sigma > 0:
    noise = np.random.default_rng(rng.randint(0, 1_000_000)).normal(
      loc=0.0,
      scale=sigma,
      size=pixels.shape
    )
  noisy = np.clip(pixels + noise, 0, 255).astype(np.uint8)
  return Image.fromarray(noisy, mode="L")


def apply_perspective_quad(image: Image.Image, rng: random.Random, jitter_ratio: float) -> Image.Image:
  width, height = image.size
  max_dx = max(1.0, width * jitter_ratio)
  max_dy = max(1.0, height * jitter_ratio)
  quad = (
    rng.uniform(-max_dx, max_dx), rng.uniform(-max_dy, max_dy),
    width + rng.uniform(-max_dx, max_dx), rng.uniform(-max_dy, max_dy),
    width + rng.uniform(-max_dx, max_dx), height + rng.uniform(-max_dy, max_dy),
    rng.uniform(-max_dx, max_dx), height + rng.uniform(-max_dy, max_dy)
  )
  if hasattr(Image, "Transform"):
    return image.transform(
      size=(width, height),
      method=Image.Transform.QUAD,
      data=quad,
      resample=Image.Resampling.BILINEAR,
      fillcolor=255
    )
  return image.transform(
    size=(width, height),
    method=Image.QUAD,
    data=quad,
    resample=Image.BILINEAR,
    fillcolor=255
  )


def apply_edge_clip(image: Image.Image, rng: random.Random) -> Image.Image:
  width, height = image.size
  clipped = image.copy()
  draw = ImageDraw.Draw(clipped)
  side = rng.choice(["left", "right", "top", "bottom"])
  if side in {"left", "right"}:
    clip = max(1, int(round(width * rng.uniform(0.04, 0.16))))
    if side == "left":
      draw.rectangle((0, 0, clip, height), fill=255)
    else:
      draw.rectangle((width - clip, 0, width, height), fill=255)
  else:
    clip = max(1, int(round(height * rng.uniform(0.04, 0.16))))
    if side == "top":
      draw.rectangle((0, 0, width, clip), fill=255)
    else:
      draw.rectangle((0, height - clip, width, height), fill=255)
  return clipped


def apply_partial_occlusion(image: Image.Image, rng: random.Random) -> Image.Image:
  width, height = image.size
  occ_w = max(1, int(round(width * rng.uniform(0.1, 0.34))))
  occ_h = max(1, int(round(height * rng.uniform(0.08, 0.3))))
  x0 = rng.randint(0, max(0, width - occ_w))
  y0 = rng.randint(0, max(0, height - occ_h))
  occluded = image.copy()
  draw = ImageDraw.Draw(occluded)
  draw.rectangle((x0, y0, x0 + occ_w, y0 + occ_h), fill=255)
  return occluded


def fit_patch_into_cell(
  patch: Image.Image,
  cell_width: int,
  cell_height: int,
  rng: random.Random
) -> Image.Image:
  source = patch.convert("L")
  target = Image.new("L", (cell_width, cell_height), color=255)
  scale_x = rng.uniform(0.68, 0.95)
  scale_y = rng.uniform(0.68, 0.95)
  max_w = max(8, int(round(cell_width * scale_x)))
  max_h = max(8, int(round(cell_height * scale_y)))
  ratio = min(max_w / max(1, source.width), max_h / max(1, source.height))
  resized_w = max(6, int(round(source.width * ratio)))
  resized_h = max(6, int(round(source.height * ratio)))
  resized = source.resize((resized_w, resized_h), Image.Resampling.BILINEAR)

  offset_x = (cell_width - resized_w) // 2 + rng.randint(-max(1, cell_width // 12), max(1, cell_width // 12))
  offset_y = (cell_height - resized_h) // 2 + rng.randint(-max(1, cell_height // 12), max(1, cell_height // 12))
  target.paste(resized, (offset_x, offset_y))
  return target


def augment_digit_cell(image: Image.Image, rng: random.Random, strong: bool = True) -> Image.Image:
  source = image.convert("L")
  width, height = source.size

  scale = rng.uniform(0.88, 1.12 if strong else 1.06)
  scaled_w = max(8, int(round(width * scale)))
  scaled_h = max(8, int(round(height * scale)))
  scaled = source.resize((scaled_w, scaled_h), Image.Resampling.BILINEAR)
  canvas = Image.new("L", (width, height), color=255)
  jitter_x = rng.randint(-max(1, width // 11), max(1, width // 11))
  jitter_y = rng.randint(-max(1, height // 11), max(1, height // 11))
  paste_x = (width - scaled_w) // 2 + jitter_x
  paste_y = (height - scaled_h) // 2 + jitter_y
  canvas.paste(scaled, (paste_x, paste_y))

  angle_limit = 8.5 if strong else 4.5
  rotated = canvas.rotate(
    rng.uniform(-angle_limit, angle_limit),
    resample=Image.Resampling.BILINEAR,
    fillcolor=255
  )

  warped = apply_perspective_quad(rotated, rng, jitter_ratio=0.075 if strong else 0.04)
  adjusted = apply_brightness_contrast(warped, rng)
  if rng.random() < (0.75 if strong else 0.45):
    adjusted = apply_noise(adjusted, rng)
  if rng.random() < (0.68 if strong else 0.35):
    adjusted = adjusted.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 1.1 if strong else 0.6)))
  if rng.random() < (0.45 if strong else 0.2):
    adjusted = apply_edge_clip(adjusted, rng)
  if rng.random() < (0.3 if strong else 0.1):
    adjusted = apply_partial_occlusion(adjusted, rng)
  return adjusted


def split_equispaced_major(image: Image.Image, count: int) -> list[Image.Image]:
  width, height = image.size
  sections: list[Image.Image] = []
  if width >= height:
    for index in range(count):
      x0 = int(round((index * width) / count))
      x1 = int(round(((index + 1) * width) / count))
      x1 = max(x0 + 1, x1)
      sections.append(image.crop((x0, 0, x1, height)))
  else:
    for index in range(count):
      y0 = int(round((index * height) / count))
      y1 = int(round(((index + 1) * height) / count))
      y1 = max(y0 + 1, y1)
      sections.append(image.crop((0, y0, width, y1)))
  return sections


def compose_synthetic_window(
  digit_pools: dict[str, list[SourceSample]],
  rng: random.Random
) -> tuple[Image.Image, list[str], list[str]]:
  available_digits = [digit for digit in DEFAULT_DIGITS if digit_pools.get(digit)]
  if not available_digits:
    raise RuntimeError("No source digits available for synthetic composition.")

  choices = rng.choices(available_digits, k=4)
  source_paths: list[str] = []
  patches: list[Image.Image] = []
  for digit in choices:
    sample = rng.choice(digit_pools[digit])
    source_paths.append(str(sample.path))
    with Image.open(sample.path) as image_file:
      patch = image_file.convert("L")
    patches.append(augment_digit_cell(patch, rng, strong=False))

  cell_width = rng.randint(54, 94)
  cell_height = rng.randint(62, 116)
  window = Image.new("L", (cell_width * 4, cell_height), color=255)

  for index, patch in enumerate(patches):
    cell = fit_patch_into_cell(patch, cell_width, cell_height, rng)
    x0 = index * cell_width
    window.paste(cell, (x0, 0))

  window = apply_perspective_quad(window, rng, jitter_ratio=0.04)
  window = apply_brightness_contrast(window, rng)
  if rng.random() < 0.55:
    window = window.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, 0.7)))
  if rng.random() < 0.5:
    window = apply_noise(window, rng)
  if rng.random() < 0.22:
    window = apply_edge_clip(window, rng)

  return window, choices, source_paths


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent
  dataset_root = resolve_path(base_dir, args.dataset_root)
  output_root = resolve_path(base_dir, args.output_root)

  if not dataset_root.exists():
    raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
  if args.direct_per_real < 0:
    raise ValueError("--direct-per-real must be >= 0.")
  if args.compose_window_count < 0:
    raise ValueError("--compose-window-count must be >= 0.")

  if args.clean and output_root.exists():
    shutil.rmtree(output_root)
  ensure_output_structure(output_root)

  rng = random.Random(args.seed)
  np.random.seed(args.seed)
  train_samples = collect_train_samples(dataset_root)
  if not train_samples:
    raise RuntimeError(f"No train samples found under: {dataset_root / 'train'}")

  pools: dict[str, list[SourceSample]] = {digit: [] for digit in DEFAULT_DIGITS}
  for sample in train_samples:
    pools[sample.digit].append(sample)

  windows_dir = output_root / "windows" if args.save_composed_windows else None
  if windows_dir:
    windows_dir.mkdir(parents=True, exist_ok=True)

  manifest_rows: list[dict[str, str]] = []
  direct_count = 0
  composed_count = 0
  digit_counts = {digit: 0 for digit in DEFAULT_DIGITS}

  for sample in train_samples:
    with Image.open(sample.path) as image_file:
      original = image_file.convert("L")
    stem = sample.path.stem
    for replicate in range(args.direct_per_real):
      synthetic = augment_digit_cell(original, rng, strong=True)
      file_name = f"{stem}__direct_r{replicate:02d}.png"
      out_path = output_root / "train" / sample.digit / file_name
      synthetic.save(out_path)
      direct_count += 1
      digit_counts[sample.digit] += 1
      manifest_rows.append({
        "kind": "direct",
        "digit": sample.digit,
        "output_path": str(out_path.relative_to(output_root)),
        "source_path": str(sample.path.relative_to(dataset_root)),
        "window_id": "",
        "cell_index": "",
        "sequence": sample.digit
      })

  for window_index in range(args.compose_window_count):
    window, sequence, source_paths = compose_synthetic_window(pools, rng)
    sections = split_equispaced_major(window, 4)
    if len(sections) != 4:
      continue

    if windows_dir:
      window_name = f"synthetic_window_{window_index:05d}.png"
      window.save(windows_dir / window_name)

    for cell_index, section in enumerate(sections):
      digit = sequence[cell_index]
      file_name = f"synthetic_window_{window_index:05d}_s{cell_index}_d{digit}.png"
      out_path = output_root / "train" / digit / file_name
      section.save(out_path)
      composed_count += 1
      digit_counts[digit] += 1
      manifest_rows.append({
        "kind": "composed",
        "digit": digit,
        "output_path": str(out_path.relative_to(output_root)),
        "source_path": "|".join([
          str(Path(path).relative_to(dataset_root))
          for path in source_paths
        ]),
        "window_id": str(window_index),
        "cell_index": str(cell_index),
        "sequence": "".join(sequence)
      })

  manifest_path = output_root / "manifests" / "synthetic_cells.csv"
  with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
    writer = csv.DictWriter(
      manifest_file,
      lineterminator="\n",
      fieldnames=[
        "kind",
        "digit",
        "output_path",
        "source_path",
        "window_id",
        "cell_index",
        "sequence"
      ]
    )
    writer.writeheader()
    writer.writerows(manifest_rows)

  summary = {
    "dataset_root": str(dataset_root),
    "output_root": str(output_root),
    "seed": args.seed,
    "train_source_count": len(train_samples),
    "direct_per_real": args.direct_per_real,
    "compose_window_count": args.compose_window_count,
    "direct_cells": direct_count,
    "composed_cells": composed_count,
    "total_cells": direct_count + composed_count,
    "digit_counts": digit_counts,
    "manifest_path": str(manifest_path)
  }
  summary_path = output_root / "manifests" / "summary.json"
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

  print(f"Train source cells: {len(train_samples)}")
  print(f"Synthetic direct cells: {direct_count}")
  print(f"Synthetic composed cells: {composed_count}")
  print(f"Synthetic total cells: {direct_count + composed_count}")
  print("Synthetic digit counts:")
  for digit in DEFAULT_DIGITS:
    print(f"  {digit}: {digit_counts[digit]}")
  print(f"Manifest: {manifest_path}")
  print(f"Summary: {summary_path}")


if __name__ == "__main__":
  main()
