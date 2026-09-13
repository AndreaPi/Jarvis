"""Tests for the full-image digit-detector shadow runtime."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

import backend.app as app_module
from backend.detector import RoiDetection
from backend.full_image_digit_shadow import (
  FullImageDigitShadow,
  build_rotation_candidates,
  crop_image,
  evaluate_roi_sanity,
)


class ArrayWrapper:
  def __init__(self, values):
    self.values = np.asarray(values)

  def cpu(self):
    return self

  def numpy(self):
    return self.values


class FakeBoxes:
  def __init__(self):
    self.xywhn = ArrayWrapper([
      [0.5, 0.8, 0.2, 0.1],
      [0.5, 0.6, 0.2, 0.1],
      [0.5, 0.4, 0.2, 0.1],
      [0.5, 0.2, 0.2, 0.1],
    ])
    self.conf = ArrayWrapper([0.9, 0.8, 0.7, 0.6])
    self.cls = ArrayWrapper([1, 2, 3, 4])

  def __len__(self):
    return 4


class FakeModel:
  def __init__(self):
    self.last_source_shape = None

  def predict(self, *, source, **_kwargs):
    self.last_source_shape = source.shape
    self.last_source_pixel = source[0, 0].tolist()
    return [type("Result", (), {"boxes": FakeBoxes()})()]


class FakeRoiDetector:
  model_name = "mock-roi.pt"

  def detect(self, *_args, **_kwargs):
    return RoiDetection(40, 35, 50, 50, 0.9, 0, "digit_window")


class FullImageDigitShadowTests(unittest.TestCase):
  def test_unloadable_optional_checkpoint_does_not_break_canonical_health(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      checkpoint = Path(directory) / "broken.pt"
      checkpoint.write_bytes(b"invalid checkpoint")
      for error in (RuntimeError("incompatible model"), EOFError("truncated"), OSError("unreadable")):
        with (
          self.subTest(error=type(error).__name__),
          patch("ultralytics.YOLO", side_effect=error),
          patch.object(app_module, "FULL_IMAGE_DIGIT_SHADOW_MODEL_PATH", checkpoint),
          patch.object(app_module, "_full_image_digit_shadow", None),
          patch.object(app_module, "_full_image_digit_shadow_error", None),
          patch.object(app_module, "get_detector"),
          patch.object(app_module, "get_digit_classifier"),
          patch.object(app_module, "get_strip_digit_reader"),
          patch.object(app_module, "get_strip_digit_reader_23xx"),
        ):
          payload = app_module.health()
          self.assertTrue(payload["ready"])
          self.assertTrue(payload["digit_ready"])
          self.assertTrue(payload["strip_digit_ready"])
          self.assertFalse(payload["full_image_digit_shadow_ready"])
          self.assertIn(str(error), payload["full_image_digit_shadow_error"])

  def test_rotation_candidates_preserve_all_four_reading_directions(self) -> None:
    detections = [
      {"class_id": digit, "x_center": 0.5, "y_center": y}
      for digit, y in zip([1, 2, 3, 4], [0.8, 0.6, 0.4, 0.2])
    ]

    candidates = {
      item["rotation"]: item["value"]
      for item in build_rotation_candidates(detections)
    }

    self.assertEqual(candidates[90], "1234")
    self.assertEqual(candidates[270], "4321")

  def test_roi_sanity_and_crop_match_runtime_geometry(self) -> None:
    accepted, status, _ = evaluate_roi_sanity({
      "x": 0.4,
      "y": 0.35,
      "width": 0.1,
      "height": 0.15,
    })
    self.assertTrue(accepted)
    self.assertEqual(status, "accepted")
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    cropped = crop_image(image, {"x": 0.25, "y": 0.2, "width": 0.5, "height": 0.4})
    self.assertIn(cropped.shape[0], {40, 41})
    self.assertEqual(cropped.shape[1:], (100, 3))

  def test_predict_returns_candidates_without_choosing_an_orientation(self) -> None:
    shadow = object.__new__(FullImageDigitShadow)
    shadow.weights_path = app_module.BASE_DIR / "mock-shadow.pt"
    shadow.device = "cpu"
    shadow._model = FakeModel()

    payload = shadow.predict(
      np.full((100, 100, 3), [255, 40, 10], dtype=np.uint8),
      FakeRoiDetector(),
    )

    self.assertTrue(payload["ok"])
    self.assertEqual(shadow._model.last_source_pixel, [10, 40, 255])
    self.assertEqual(payload["detection_count"], 4)
    self.assertEqual(payload["confidence"], 0.6)
    self.assertEqual(len(payload["candidates"]), 4)
    self.assertNotIn("value", payload)
    self.assertGreater(shadow._model.last_source_shape[0], 15)


if __name__ == "__main__":
  unittest.main()
