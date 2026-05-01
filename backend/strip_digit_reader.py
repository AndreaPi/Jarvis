from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
  from .strip_digit_model import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_STRIP_HEIGHT,
    DEFAULT_STRIP_WIDTH,
    build_strip_digit_reader,
    prepare_strip_tensor
  )
except ImportError:
  from strip_digit_model import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_STRIP_HEIGHT,
    DEFAULT_STRIP_WIDTH,
    build_strip_digit_reader,
    prepare_strip_tensor
  )


class StripDigitReaderUnavailableError(RuntimeError):
  """Raised when strip digit reader dependencies or weights are unavailable."""


@dataclass
class StripDigitPrediction:
  value: str
  confidence: float
  digits: list[str]
  digit_confidences: list[float]
  top_k_by_position: list[list[dict[str, float]]]


class StripDigitReader:
  def __init__(self, weights_path: Path, device: str | None = None) -> None:
    self.weights_path = Path(weights_path)
    if not self.weights_path.exists():
      raise StripDigitReaderUnavailableError(
        f"Strip digit reader weights not found at {self.weights_path}. Train first or set STRIP_DIGIT_MODEL_PATH."
      )

    try:
      import torch
    except ImportError as error:
      raise StripDigitReaderUnavailableError(
        "torch is not installed. Install backend/requirements.txt first."
      ) from error

    self._torch = torch
    self.device = self._resolve_device(device)
    self._checkpoint = self._load_checkpoint()
    self.class_names = self._resolve_class_names(self._checkpoint)
    self.sequence_length = self._resolve_sequence_length(self._checkpoint)
    self.image_width, self.image_height = self._resolve_image_size(self._checkpoint)
    state_dict = self._resolve_state_dict(self._checkpoint)

    model = build_strip_digit_reader(
      sequence_length=self.sequence_length,
      num_classes=len(self.class_names)
    )
    model.load_state_dict(state_dict)
    model.eval()
    model.to(self.device)
    self._model = model

  def _resolve_device(self, raw: str | None) -> str:
    if raw is None:
      return "cpu"
    normalized = str(raw).strip()
    if not normalized:
      return "cpu"
    if normalized.lower() == "auto":
      return "cuda:0" if self._torch.cuda.is_available() else "cpu"
    return normalized

  def _load_checkpoint(self) -> dict:
    try:
      payload = self._torch.load(str(self.weights_path), map_location="cpu")
    except Exception as error:
      raise StripDigitReaderUnavailableError(
        f"Unable to load strip digit reader checkpoint at {self.weights_path}: {error}"
      ) from error
    if not isinstance(payload, dict):
      raise StripDigitReaderUnavailableError(
        f"Unexpected checkpoint payload at {self.weights_path}; expected dict."
      )
    return payload

  @staticmethod
  def _resolve_class_names(checkpoint: dict) -> list[str]:
    names = checkpoint.get("class_names")
    if isinstance(names, list) and names:
      return [str(item) for item in names]
    return list(DEFAULT_CLASS_NAMES)

  @staticmethod
  def _resolve_sequence_length(checkpoint: dict) -> int:
    value = checkpoint.get("sequence_length")
    try:
      parsed = int(value)
    except (TypeError, ValueError):
      parsed = DEFAULT_SEQUENCE_LENGTH
    return max(1, parsed)

  @staticmethod
  def _resolve_image_size(checkpoint: dict) -> tuple[int, int]:
    width = checkpoint.get("image_width")
    height = checkpoint.get("image_height")
    try:
      parsed_width = int(width)
    except (TypeError, ValueError):
      parsed_width = DEFAULT_STRIP_WIDTH
    try:
      parsed_height = int(height)
    except (TypeError, ValueError):
      parsed_height = DEFAULT_STRIP_HEIGHT
    return max(32, parsed_width), max(32, parsed_height)

  @staticmethod
  def _resolve_state_dict(checkpoint: dict) -> dict:
    state_dict = checkpoint.get("state_dict")
    if isinstance(state_dict, dict) and state_dict:
      return state_dict
    raise StripDigitReaderUnavailableError("Checkpoint is missing a valid state_dict.")

  @property
  def model_name(self) -> str:
    return self.weights_path.name

  @property
  def device_name(self) -> str:
    return self.device

  def predict(self, image_rgb: np.ndarray, top_k: int = 3) -> StripDigitPrediction:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
      raise ValueError("Expected RGB image with shape HxWx3.")

    pil_image = Image.fromarray(image_rgb).convert("L")
    tensor = prepare_strip_tensor(
      pil_image,
      width=self.image_width,
      height=self.image_height
    ).unsqueeze(0)
    tensor = tensor.to(self.device)

    with self._torch.no_grad():
      logits = self._model(tensor)
      probabilities = self._torch.softmax(logits, dim=2)[0]

    bounded_top_k = max(1, min(int(top_k), len(self.class_names)))
    top_k_by_position: list[list[dict[str, float]]] = []
    digits: list[str] = []
    digit_confidences: list[float] = []
    for position in range(self.sequence_length):
      position_probabilities = probabilities[position]
      confidence, index = self._torch.max(position_probabilities, dim=0)
      predicted_index = int(index.item())
      digits.append(self.class_names[predicted_index])
      digit_confidences.append(float(confidence.item()))

      top_values, top_indices = self._torch.topk(position_probabilities, k=bounded_top_k)
      position_top_k = []
      for rank in range(bounded_top_k):
        class_index = int(top_indices[rank].item())
        position_top_k.append({
          "digit": self.class_names[class_index],
          "confidence": float(top_values[rank].item())
        })
      top_k_by_position.append(position_top_k)

    return StripDigitPrediction(
      value="".join(digits),
      confidence=sum(digit_confidences) / max(1, len(digit_confidences)),
      digits=digits,
      digit_confidences=digit_confidences,
      top_k_by_position=top_k_by_position
    )
