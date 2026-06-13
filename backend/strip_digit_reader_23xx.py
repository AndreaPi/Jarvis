from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
  from .strip_digit_model import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_STRIP_HEIGHT,
    DEFAULT_STRIP_WIDTH,
    build_strip_digit_reader_23xx,
    prepare_strip_tensor
  )
except ImportError:
  from strip_digit_model import (
    DEFAULT_CLASS_NAMES,
    DEFAULT_STRIP_HEIGHT,
    DEFAULT_STRIP_WIDTH,
    build_strip_digit_reader_23xx,
    prepare_strip_tensor
  )


class StripDigitReader23xxUnavailableError(RuntimeError):
  """Raised when constrained strip-reader dependencies or weights are unavailable."""


@dataclass
class StripDigit23xxPrediction:
  accepted: bool
  value: str | None
  predicted_value: str
  fixed_prefix: str
  prefix_guard: dict[str, object]
  suffix_digits: list[str]
  confidence: float
  guard_confidence: float
  suffix_confidences: list[float]
  top_k_by_position: list[list[dict[str, float]]]


class StripDigitReader23xx:
  def __init__(self, weights_path: Path, device: str | None = None) -> None:
    self.weights_path = Path(weights_path)
    if not self.weights_path.exists():
      raise StripDigitReader23xxUnavailableError(
        f"23xx strip reader weights not found at {self.weights_path}. Train first or set STRIP_DIGIT_23XX_MODEL_PATH."
      )

    try:
      import torch
    except ImportError as error:
      raise StripDigitReader23xxUnavailableError(
        "torch is not installed. Install backend/requirements.txt first."
      ) from error

    self._torch = torch
    self.device = self._resolve_device(device)
    self._checkpoint = self._load_checkpoint()
    self.class_names = self._resolve_class_names(self._checkpoint)
    self.fixed_prefix = str(self._checkpoint.get("fixed_prefix") or "23")
    self.guard_threshold = self._resolve_float(self._checkpoint.get("guard_threshold"), 0.98)
    self.image_width, self.image_height = self._resolve_image_size(self._checkpoint)
    state_dict = self._resolve_state_dict(self._checkpoint)

    model = build_strip_digit_reader_23xx(num_classes=len(self.class_names))
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
      raise StripDigitReader23xxUnavailableError(
        f"Unable to load 23xx strip reader checkpoint at {self.weights_path}: {error}"
      ) from error
    if not isinstance(payload, dict):
      raise StripDigitReader23xxUnavailableError(
        f"Unexpected checkpoint payload at {self.weights_path}; expected dict."
      )
    return payload

  @staticmethod
  def _resolve_float(value: object, default: float) -> float:
    try:
      parsed = float(value)
    except (TypeError, ValueError):
      return default
    return parsed if np.isfinite(parsed) else default

  @staticmethod
  def _resolve_class_names(checkpoint: dict) -> list[str]:
    names = checkpoint.get("class_names")
    if isinstance(names, list) and names:
      return [str(item) for item in names]
    return list(DEFAULT_CLASS_NAMES)

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
    raise StripDigitReader23xxUnavailableError("Checkpoint is missing a valid state_dict.")

  @property
  def model_name(self) -> str:
    return self.weights_path.name

  @property
  def device_name(self) -> str:
    return self.device

  def predict(self, image_rgb: np.ndarray, top_k: int = 3, guard_threshold: float | None = None) -> StripDigit23xxPrediction:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
      raise ValueError("Expected RGB image with shape HxWx3.")

    threshold = self.guard_threshold if guard_threshold is None else float(guard_threshold)
    pil_image = Image.fromarray(image_rgb).convert("L")
    tensor = prepare_strip_tensor(
      pil_image,
      width=self.image_width,
      height=self.image_height
    ).unsqueeze(0)
    tensor = tensor.to(self.device)

    with self._torch.no_grad():
      outputs = self._model(tensor)
      guard_probabilities = self._torch.softmax(outputs["guard_logits"], dim=1)[0]
      suffix_probabilities = self._torch.softmax(outputs["suffix_logits"], dim=2)[0]

    guard_confidence = float(guard_probabilities[1].item())
    guard_is_3 = guard_confidence >= 0.5
    suffix_digits: list[str] = []
    suffix_confidences: list[float] = []
    top_k_by_position: list[list[dict[str, float]]] = []
    bounded_top_k = max(1, min(int(top_k), len(self.class_names)))

    for position in range(2):
      position_probabilities = suffix_probabilities[position]
      confidence, index = self._torch.max(position_probabilities, dim=0)
      predicted_index = int(index.item())
      suffix_digits.append(self.class_names[predicted_index])
      suffix_confidences.append(float(confidence.item()))

      top_values, top_indices = self._torch.topk(position_probabilities, k=bounded_top_k)
      position_top_k = []
      for rank in range(bounded_top_k):
        class_index = int(top_indices[rank].item())
        position_top_k.append({
          "digit": self.class_names[class_index],
          "confidence": float(top_values[rank].item())
        })
      top_k_by_position.append(position_top_k)

    predicted_value = f"{self.fixed_prefix}{''.join(suffix_digits)}"
    accepted = bool(guard_is_3 and guard_confidence >= threshold)
    confidence_parts = [guard_confidence, *suffix_confidences]
    confidence = sum(confidence_parts) / max(1, len(confidence_parts))
    return StripDigit23xxPrediction(
      accepted=accepted,
      value=predicted_value if accepted else None,
      predicted_value=predicted_value,
      fixed_prefix=self.fixed_prefix,
      prefix_guard={
        "label": "second_digit_is_3",
        "is_3": guard_is_3,
        "confidence": guard_confidence,
        "threshold": threshold
      },
      suffix_digits=suffix_digits,
      confidence=confidence,
      guard_confidence=guard_confidence,
      suffix_confidences=suffix_confidences,
      top_k_by_position=top_k_by_position
    )
