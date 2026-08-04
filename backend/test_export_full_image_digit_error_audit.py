"""Tests for the full-image digit-detector error-audit exporter."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from backend.detector import RoiDetection
from backend.export_full_image_digit_error_audit import (
  choose_aperture_detection,
  evaluate_runtime_roi_sanity,
  expand_normalized_bbox,
  failure_bucket,
  intersection_over_union,
  map_crop_detection_to_full_image,
  match_detections_by_iou,
  prepare_roi_cascade_images,
  predict_runtime_images_one_at_a_time,
  write_transition_review,
)


def box(
  digit: int,
  x_center: float,
  *,
  position: int = 0,
) -> dict[str, str]:
  return {
    "digit": str(digit),
    "position": str(position),
    "x_center": str(x_center),
    "y_center": "0.5",
    "width": "0.2",
    "height": "0.4",
    "transition_state": "unknown",
  }


def detection(digit: int, x_center: float, confidence: float = 0.9) -> dict[str, object]:
  return {
    "class_id": digit,
    "confidence": confidence,
    "x_center": x_center,
    "y_center": 0.5,
    "width": 0.2,
    "height": 0.4,
  }


class GeometryTests(unittest.TestCase):
  def test_intersection_over_union(self) -> None:
    self.assertEqual(intersection_over_union((0, 0, 1, 1), (2, 2, 3, 3)), 0)
    self.assertAlmostEqual(
      intersection_over_union((0, 0, 1, 1), (0.5, 0, 1.5, 1)),
      1 / 3,
    )

  def test_matches_by_geometry_instead_of_detection_order(self) -> None:
    truth = [box(1, 0.25, position=0), box(2, 0.75, position=1)]
    predictions = [detection(2, 0.75), detection(1, 0.25)]

    matches, missing_truth, extra_detections = match_detections_by_iou(
      truth,
      predictions,
      threshold=0.5,
    )

    self.assertEqual(matches[0][0], 1)
    self.assertEqual(matches[1][0], 0)
    self.assertEqual(missing_truth, [])
    self.assertEqual(extra_detections, [])

  def test_runtime_roi_sanity_matches_frontend_boundaries(self) -> None:
    accepted, status, geometry = evaluate_runtime_roi_sanity({
      "x": 0.38,
      "y": 0.42,
      "width": 0.08,
      "height": 0.12,
    })
    self.assertTrue(accepted)
    self.assertEqual(status, "accepted")
    self.assertAlmostEqual(geometry["area"], 0.0096)

    accepted, status, _ = evaluate_runtime_roi_sanity({
      "x": 0.02,
      "y": 0.42,
      "width": 0.08,
      "height": 0.12,
    })
    self.assertFalse(accepted)
    self.assertEqual(status, "invalid-center-x")

  def test_roi_expansion_clips_and_crop_detection_maps_back(self) -> None:
    expanded = expand_normalized_bbox(
      {"x": 0.02, "y": 0.10, "width": 0.20, "height": 0.30},
      0.25,
      0.50,
    )
    self.assertEqual(expanded["x"], 0.0)
    self.assertEqual(expanded["y"], 0.0)
    self.assertAlmostEqual(expanded["width"], 0.27)
    self.assertAlmostEqual(expanded["height"], 0.55)
    mapped = map_crop_detection_to_full_image(
      detection(4, 0.5),
      {"x": 0.2, "y": 0.3, "width": 0.4, "height": 0.2},
    )
    self.assertAlmostEqual(mapped["x_center"], 0.4)
    self.assertAlmostEqual(mapped["y_center"], 0.4)
    self.assertAlmostEqual(mapped["width"], 0.08)
    self.assertAlmostEqual(mapped["height"], 0.08)

  def test_prepare_roi_cascade_uses_detector_geometry_not_truth_crop(self) -> None:
    class FakeDetector:
      def detect(self, *_args, **_kwargs) -> RoiDetection:
        return RoiDetection(36, 36, 44, 48, 0.9, 0, "digit_window")

    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / "train").mkdir()
      Image.new("RGB", (100, 100), "white").save(root / "train" / "meter.jpg")
      rows = [
        {
          **box(digit, 0.38 + position * 0.015, position=position),
          "filename": "meter.jpg",
          "split": "train",
        }
        for position, digit in enumerate([1, 2, 3, 4])
      ]
      images, filenames, metadata = prepare_roi_cascade_images(
        [{"filename": "meter.jpg"}],
        {"meter.jpg": rows},
        root,
        FakeDetector(),
        SimpleNamespace(
          roi_confidence=0.05,
          roi_iou=0.5,
          roi_imgsz=960,
          roi_expand_x=0.26,
          roi_expand_y=0.16,
        ),
      )

    self.assertEqual(filenames, ["meter.jpg"])
    self.assertEqual(len(images), 1)
    self.assertEqual(metadata["meter.jpg"]["status"], "accepted")
    self.assertEqual(metadata["meter.jpg"]["bbox_norm"]["x"], 0.36)


class ClassificationTests(unittest.TestCase):
  def test_runtime_cascade_predicts_each_image_separately(self) -> None:
    class FakeModel:
      def __init__(self) -> None:
        self.sources = []

      def predict(self, *, source, **_kwargs):
        self.sources.append(source)
        return [f"result-{len(self.sources)}"]

    model = FakeModel()
    results = predict_runtime_images_one_at_a_time(
      model,
      [Image.new("RGB", (20, 30)), Image.new("RGB", (40, 50))],
      imgsz=1280,
      device="cpu",
      confidence=0.25,
      iou=0.7,
      max_detections=300,
    )

    self.assertEqual(results, ["result-1", "result-2"])
    self.assertEqual(len(model.sources), 2)
    self.assertEqual(model.sources[0].shape, (30, 20, 3))
    self.assertEqual(model.sources[1].shape, (50, 40, 3))

  def test_failure_buckets_distinguish_count_geometry_and_classification(self) -> None:
    base = {"exact_match": False, "detection_count": 4}
    self.assertEqual(
      failure_bucket({**base, "exact_match": True}, 4),
      "exact",
    )
    self.assertEqual(
      failure_bucket({**base, "detection_count": 3}, 3),
      "no-read-missing-detection",
    )
    self.assertEqual(
      failure_bucket({**base, "detection_count": 5}, 4),
      "no-read-extra-detection",
    )
    self.assertEqual(failure_bucket(base, 3), "readable-localization-error")
    self.assertEqual(failure_bucket(base, 4), "readable-classification-error")

  def test_aperture_selection_prefers_central_detection(self) -> None:
    selected = choose_aperture_detection([
      detection(8, 0.05, confidence=0.99),
      detection(4, 0.50, confidence=0.60),
    ])

    self.assertIsNotNone(selected)
    self.assertEqual(selected["class_id"], 4)
    self.assertIsNone(choose_aperture_detection([]))


class TransitionWorksheetTests(unittest.TestCase):
  def test_review_fields_are_blank_and_predictions_are_preserved(self) -> None:
    rows = [{
      "filename": "meter.JPEG",
      "fold": 2,
      "positions": [{
        "position": 0,
        "truth_digit": 4,
        "transition_state": "unknown",
        "full_image_predicted_digit": 1,
        "full_image_confidence": 0.8,
        "full_image_iou": 0.7,
        "aperture_oracle": {
          "predicted_digit": 4,
          "confidence": 0.9,
        },
      }],
    }]
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "transition-review.csv"
      write_transition_review(rows, path)
      with path.open(encoding="utf-8", newline="") as handle:
        exported = list(csv.DictReader(handle))

    self.assertEqual(len(exported), 1)
    self.assertEqual(exported[0]["truth_digit"], "4")
    self.assertEqual(exported[0]["full_image_predicted_digit"], "1")
    self.assertEqual(exported[0]["aperture_oracle_digit"], "4")
    self.assertEqual(exported[0]["reviewed_transition_state"], "")
    self.assertEqual(exported[0]["review_notes"], "")


if __name__ == "__main__":
  unittest.main()
