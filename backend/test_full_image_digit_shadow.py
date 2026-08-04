"""Tests for the full-image digit-detector shadow runtime."""

from __future__ import annotations

import unittest

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
    return [type("Result", (), {"boxes": FakeBoxes()})()]


class FakeRoiDetector:
  model_name = "mock-roi.pt"

  def detect(self, *_args, **_kwargs):
    return RoiDetection(40, 35, 50, 50, 0.9, 0, "digit_window")


class StubUpload:
  async def read(self, _size=-1):
    return b"test-image"


class FullImageDigitShadowTests(unittest.TestCase):
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
      np.zeros((100, 100, 3), dtype=np.uint8),
      FakeRoiDetector(),
    )

    self.assertTrue(payload["ok"])
    self.assertEqual(payload["detection_count"], 4)
    self.assertEqual(payload["confidence"], 0.6)
    self.assertEqual(len(payload["candidates"]), 4)
    self.assertNotIn("value", payload)
    self.assertGreater(shadow._model.last_source_shape[0], 15)


class ShadowEndpointTests(unittest.IsolatedAsyncioTestCase):
  async def test_endpoint_passes_through_diagnostic_payload(self) -> None:
    expected = {
      "ok": True,
      "candidates": [{"rotation": 90, "value": "2311"}],
    }

    class StubShadow:
      def predict(self, *_args, **_kwargs):
        return expected

    original_detector = app_module._detector
    original_shadow = app_module._full_image_digit_shadow
    original_loader = app_module._load_rgb_image
    try:
      app_module._detector = FakeRoiDetector()
      app_module._full_image_digit_shadow = StubShadow()
      app_module._load_rgb_image = lambda _payload: np.zeros((10, 10, 3), dtype=np.uint8)

      payload = await app_module.predict_full_image_digit_shadow(StubUpload())
    finally:
      app_module._detector = original_detector
      app_module._full_image_digit_shadow = original_shadow
      app_module._load_rgb_image = original_loader

    self.assertEqual(payload, expected)


if __name__ == "__main__":
  unittest.main()
