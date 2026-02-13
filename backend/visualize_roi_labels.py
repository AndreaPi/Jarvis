from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class RenderedPreview:
  split: str
  image_path: Path
  preview_path: Path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Render YOLO label overlays for quick visual QA."
  )
  parser.add_argument(
    "--dataset-root",
    default="data/roi_dataset",
    help="YOLO dataset root (relative to backend/ by default)."
  )
  parser.add_argument(
    "--out-dir",
    default="data/roi_dataset/qa_previews",
    help="Directory to write per-image previews."
  )
  parser.add_argument(
    "--skip-contact-sheet",
    action="store_true",
    help="Skip generating the combined contact sheet."
  )
  return parser.parse_args()


def resolve_path(base_dir: Path, value: str) -> Path:
  path = Path(value)
  if path.is_absolute():
    return path
  return (base_dir / path).resolve()


def find_image_for_label(image_dir: Path, stem: str) -> Path | None:
  for extension in SUPPORTED_IMAGE_EXTENSIONS:
    candidate = image_dir / f"{stem}{extension}"
    if candidate.exists():
      return candidate

  lower_lookup = {path.stem.lower(): path for path in image_dir.iterdir() if path.is_file()}
  return lower_lookup.get(stem.lower())


def parse_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
  labels: list[tuple[int, float, float, float, float]] = []
  raw = label_path.read_text(encoding="utf-8").strip()
  if not raw:
    return labels

  for line in raw.splitlines():
    tokens = line.split()
    if len(tokens) != 5:
      raise ValueError(f"Invalid YOLO line in {label_path}: {line}")
    class_id = int(tokens[0])
    xc, yc, width, height = map(float, tokens[1:])
    labels.append((class_id, xc, yc, width, height))

  return labels


def draw_preview(image_path: Path, labels: list[tuple[int, float, float, float, float]], out_path: Path) -> None:
  try:
    from PIL import Image, ImageDraw
  except ImportError as error:
    raise RuntimeError("Pillow is required. Install backend/requirements.txt first.") from error

  with Image.open(image_path).convert("RGB") as image:
    width, height = image.size
    draw = ImageDraw.Draw(image)
    line_width = max(2, width // 350)

    for class_id, xc, yc, box_w, box_h in labels:
      pixel_w = box_w * width
      pixel_h = box_h * height
      x1 = (xc * width) - (pixel_w * 0.5)
      y1 = (yc * height) - (pixel_h * 0.5)
      x2 = x1 + pixel_w
      y2 = y1 + pixel_h

      draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 255), width=line_width)
      draw.text((x1 + 4, max(2, y1 - 14)), f"class {class_id}", fill=(0, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=95)


def build_contact_sheet(previews: list[RenderedPreview], output_path: Path) -> None:
  if not previews:
    return

  try:
    from PIL import Image, ImageDraw, ImageOps
  except ImportError as error:
    raise RuntimeError("Pillow is required. Install backend/requirements.txt first.") from error

  thumb_width = 420
  thumb_height = 320
  label_height = 40
  padding = 16
  columns = 3
  rows = (len(previews) + columns - 1) // columns

  canvas_width = padding + columns * (thumb_width + padding)
  canvas_height = padding + rows * (thumb_height + label_height + padding)
  canvas = Image.new("RGB", (canvas_width, canvas_height), (22, 22, 24))
  draw = ImageDraw.Draw(canvas)

  for index, preview in enumerate(previews):
    row = index // columns
    column = index % columns
    x = padding + column * (thumb_width + padding)
    y = padding + row * (thumb_height + label_height + padding)

    with Image.open(preview.preview_path).convert("RGB") as image:
      thumb = ImageOps.contain(image, (thumb_width, thumb_height))
      frame = Image.new("RGB", (thumb_width, thumb_height), (40, 40, 44))
      offset_x = (thumb_width - thumb.width) // 2
      offset_y = (thumb_height - thumb.height) // 2
      frame.paste(thumb, (offset_x, offset_y))

    canvas.paste(frame, (x, y))
    label = f"{preview.split}/{preview.image_path.name}"
    draw.text((x, y + thumb_height + 10), label, fill=(230, 230, 230))

  output_path.parent.mkdir(parents=True, exist_ok=True)
  canvas.save(output_path, quality=95)


def main() -> None:
  args = parse_args()
  base_dir = Path(__file__).resolve().parent
  dataset_root = resolve_path(base_dir, args.dataset_root)
  out_dir = resolve_path(base_dir, args.out_dir)

  if not dataset_root.exists():
    raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

  previews: list[RenderedPreview] = []
  missing_images: list[str] = []
  empty_labels: list[str] = []

  for split in ("train", "val", "test"):
    label_dir = dataset_root / "labels" / split
    image_dir = dataset_root / "images" / split
    split_out = out_dir / split

    if not label_dir.exists():
      continue

    for label_path in sorted(label_dir.glob("*.txt")):
      image_path = find_image_for_label(image_dir, label_path.stem)
      if image_path is None:
        missing_images.append(f"{split}/{label_path.name}")
        continue

      labels = parse_yolo_labels(label_path)
      if not labels:
        empty_labels.append(f"{split}/{label_path.name}")
        continue

      preview_path = split_out / f"{label_path.stem}_qa.jpg"
      draw_preview(image_path=image_path, labels=labels, out_path=preview_path)
      previews.append(
        RenderedPreview(split=split, image_path=image_path, preview_path=preview_path)
      )

  if not args.skip_contact_sheet and previews:
    contact_sheet_path = out_dir / "qa_contact_sheet.jpg"
    build_contact_sheet(previews, contact_sheet_path)
    print(f"Contact sheet: {contact_sheet_path}")

  print(f"Rendered previews: {len(previews)} in {out_dir}")
  if missing_images:
    print("Missing source images:")
    for item in missing_images:
      print(f"  - {item}")
  if empty_labels:
    print("Empty label files:")
    for item in empty_labels:
      print(f"  - {item}")


if __name__ == "__main__":
  main()
