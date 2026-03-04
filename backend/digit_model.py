from __future__ import annotations

import numpy as np

DEFAULT_IMAGE_SIZE = 64
DEFAULT_NUM_CLASSES = 10
DEFAULT_CLASS_NAMES = [str(index) for index in range(DEFAULT_NUM_CLASSES)]


def _resolve_torch():
  try:
    import torch
    from torch import nn
  except ImportError as error:
    raise RuntimeError("torch is required for digit classifier training and inference.") from error
  return torch, nn


def build_digit_cnn(num_classes: int = DEFAULT_NUM_CLASSES):
  _, nn = _resolve_torch()
  return nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
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
    nn.Conv2d(128, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Dropout(p=0.22),
    nn.Linear(128, num_classes)
  )


def prepare_digit_tensor(image, image_size: int = DEFAULT_IMAGE_SIZE):
  try:
    from PIL import Image
  except ImportError as error:
    raise RuntimeError("Pillow is required for digit classifier preprocessing.") from error
  torch, _ = _resolve_torch()

  if image.mode != "L":
    image = image.convert("L")
  if hasattr(Image, "Resampling"):
    resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
  else:
    resized = image.resize((image_size, image_size), Image.BILINEAR)
  pixels = np.asarray(resized, dtype=np.float32) / 255.0
  return torch.from_numpy(pixels).unsqueeze(0)
