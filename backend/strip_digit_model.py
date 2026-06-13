from __future__ import annotations

import numpy as np
from PIL import Image

DEFAULT_STRIP_WIDTH = 520
DEFAULT_STRIP_HEIGHT = 160
DEFAULT_SEQUENCE_LENGTH = 4
DEFAULT_NUM_CLASSES = 10
DEFAULT_CLASS_NAMES = [str(index) for index in range(DEFAULT_NUM_CLASSES)]


def _resolve_torch():
  try:
    import torch
    from torch import nn
  except ImportError as error:
    raise RuntimeError("torch is required for strip digit reader training and inference.") from error
  return torch, nn


def build_strip_digit_reader(
  sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
  num_classes: int = DEFAULT_NUM_CLASSES
):
  torch, nn = _resolve_torch()

  class StripDigitReaderCnn(nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.sequence_length = sequence_length
      self.num_classes = num_classes
      self.features = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=5, padding=2),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(128, 192, kernel_size=3, padding=1),
        nn.BatchNorm2d(192),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(192, 192, kernel_size=3, padding=1),
        nn.BatchNorm2d(192),
        nn.ReLU(inplace=True)
      )
      self.pool = nn.AdaptiveAvgPool2d((4, 16))
      self.embedding = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(p=0.28),
        nn.Linear(192 * 4 * 16, 384),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.18)
      )
      self.heads = nn.ModuleList([
        nn.Linear(384, num_classes)
        for _ in range(sequence_length)
      ])

    def forward(self, inputs):
      features = self.pool(self.features(inputs))
      embedding = self.embedding(features)
      return torch.stack([head(embedding) for head in self.heads], dim=1)

  return StripDigitReaderCnn()


def build_strip_digit_reader_23xx(num_classes: int = DEFAULT_NUM_CLASSES):
  torch, nn = _resolve_torch()

  class StripDigitReader23xxCnn(nn.Module):
    def __init__(self) -> None:
      super().__init__()
      self.fixed_prefix = "23"
      self.num_classes = num_classes
      self.features = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=5, padding=2),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(128, 192, kernel_size=3, padding=1),
        nn.BatchNorm2d(192),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(192, 192, kernel_size=3, padding=1),
        nn.BatchNorm2d(192),
        nn.ReLU(inplace=True)
      )
      self.pool = nn.AdaptiveAvgPool2d((4, 16))
      self.embedding = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(p=0.28),
        nn.Linear(192 * 4 * 16, 384),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.18)
      )
      self.guard_head = nn.Linear(384, 2)
      self.suffix_heads = nn.ModuleList([
        nn.Linear(384, num_classes)
        for _ in range(2)
      ])

    def forward(self, inputs):
      features = self.pool(self.features(inputs))
      embedding = self.embedding(features)
      return {
        "guard_logits": self.guard_head(embedding),
        "suffix_logits": torch.stack([head(embedding) for head in self.suffix_heads], dim=1)
      }

  return StripDigitReader23xxCnn()


def letterbox_strip_image(
  image: Image.Image,
  width: int = DEFAULT_STRIP_WIDTH,
  height: int = DEFAULT_STRIP_HEIGHT
) -> Image.Image:
  if image.mode != "L":
    image = image.convert("L")
  scale = min(width / max(1, image.width), height / max(1, image.height))
  resized_width = max(1, int(round(image.width * scale)))
  resized_height = max(1, int(round(image.height * scale)))
  resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
  resized = image.resize((resized_width, resized_height), resample)
  canvas = Image.new("L", (width, height), color=255)
  left = (width - resized_width) // 2
  top = (height - resized_height) // 2
  canvas.paste(resized, (left, top))
  return canvas


def prepare_strip_tensor(
  image: Image.Image,
  width: int = DEFAULT_STRIP_WIDTH,
  height: int = DEFAULT_STRIP_HEIGHT
):
  torch, _ = _resolve_torch()
  prepared = letterbox_strip_image(image, width=width, height=height)
  pixels = np.asarray(prepared, dtype=np.float32) / 255.0
  return torch.from_numpy(pixels).unsqueeze(0)
