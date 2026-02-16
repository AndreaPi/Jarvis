from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
  from .digit_model import DEFAULT_CLASS_NAMES, DEFAULT_IMAGE_SIZE, build_digit_cnn, prepare_digit_tensor
except ImportError:
  from digit_model import DEFAULT_CLASS_NAMES, DEFAULT_IMAGE_SIZE, build_digit_cnn, prepare_digit_tensor


class DigitClassifierUnavailableError(RuntimeError):
  """Raised when digit classifier dependencies or weights are unavailable."""


@dataclass
class DigitPrediction:
  digit: str
  confidence: float
  top_k: list[dict[str, float]]


class DigitClassifier:
  def __init__(self, weights_path: Path, device: str | None = None) -> None:
    self.weights_path = Path(weights_path)
    if not self.weights_path.exists():
      raise DigitClassifierUnavailableError(
        f"Digit classifier weights not found at {self.weights_path}. Train first or set DIGIT_MODEL_PATH."
      )

    try:
      import torch
    except ImportError as error:
      raise DigitClassifierUnavailableError(
        "torch is not installed. Install backend/requirements.txt first."
      ) from error

    self._torch = torch
    self.device = self._resolve_device(device)
    self._checkpoint = self._load_checkpoint()
    self.class_names = self._resolve_class_names(self._checkpoint)
    self.image_size = self._resolve_image_size(self._checkpoint)
    state_dict = self._resolve_state_dict(self._checkpoint)

    model = build_digit_cnn(num_classes=len(self.class_names))
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
      raise DigitClassifierUnavailableError(
        f"Unable to load digit classifier checkpoint at {self.weights_path}: {error}"
      ) from error
    if not isinstance(payload, dict):
      raise DigitClassifierUnavailableError(
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
  def _resolve_image_size(checkpoint: dict) -> int:
    image_size = checkpoint.get("image_size")
    try:
      parsed = int(image_size)
    except (TypeError, ValueError):
      parsed = DEFAULT_IMAGE_SIZE
    return max(24, parsed)

  @staticmethod
  def _resolve_state_dict(checkpoint: dict) -> dict:
    state_dict = checkpoint.get("state_dict")
    if isinstance(state_dict, dict) and state_dict:
      return state_dict
    raise DigitClassifierUnavailableError("Checkpoint is missing a valid state_dict.")

  @property
  def model_name(self) -> str:
    return self.weights_path.name

  @property
  def device_name(self) -> str:
    return self.device

  def predict(self, image_rgb: np.ndarray, top_k: int = 3) -> DigitPrediction:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
      raise ValueError("Expected RGB image with shape HxWx3.")

    pil_image = Image.fromarray(image_rgb).convert("L")
    tensor = prepare_digit_tensor(pil_image, self.image_size).unsqueeze(0)
    tensor = tensor.to(self.device)

    with self._torch.no_grad():
      logits = self._model(tensor)
      probabilities = self._torch.softmax(logits, dim=1)[0]

    confidence, index = self._torch.max(probabilities, dim=0)
    predicted_index = int(index.item())
    predicted_digit = self.class_names[predicted_index]
    confidence_value = float(confidence.item())

    bounded_top_k = max(1, min(int(top_k), len(self.class_names)))
    top_values, top_indices = self._torch.topk(probabilities, k=bounded_top_k)
    top_payload = []
    for rank in range(bounded_top_k):
      class_index = int(top_indices[rank].item())
      top_payload.append({
        "digit": self.class_names[class_index],
        "confidence": float(top_values[rank].item())
      })

    return DigitPrediction(
      digit=predicted_digit,
      confidence=confidence_value,
      top_k=top_payload
    )
