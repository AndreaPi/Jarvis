"""Tests for the full-image digit-detector error-audit exporter."""

from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from backend.detector import RoiDetection
from backend.export_full_image_digit_error_audit import (
  aggregate_summary,
  build_audit_rows,
  build_sequence_record,
  choose_aperture_detection,
  display_oriented_crop,
  evaluate_runtime_roi_sanity,
  expand_normalized_bbox,
  failure_bucket,
  intersection_over_union,
  map_crop_detection_to_full_image,
  match_detections_by_iou,
  prepare_roi_cascade_images,
  predict_runtime_images_one_at_a_time,
  run_oracles,
  write_transition_review,
  write_html_report,
  write_markdown_summary,
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
  def test_transition_findings_follow_actual_reviewed_states(self) -> None:
    cases = (
      (["unknown"] * 4, {"unknown": 4}),
      (["stable", "transitioning", "uncertain", "unknown"],
       {"stable": 1, "transitioning": 1, "uncertain": 1, "unknown": 1}),
      (["stable"] * 4, {"stable": 4}),
    )
    for states, counts in cases:
      with self.subTest(states=states):
        record = build_sequence_record("meter.jpg", "1234", 0, [])
        evaluations = [{"fold": 0, "predictions": [record]}]
        annotations = {"meter.jpg": [
          {**box(digit, 0.2 + position * 0.2, position=position), "transition_state": state}
          for position, (digit, state) in enumerate(zip((1, 2, 3, 4), states))
        ]}
        register = {"meter.jpg": record}
        cascade = {"meter.jpg": {**record, "roi": {
          "status": "accepted", "confidence": 0.9, "truth_register_coverage": 1.0,
        }}}
        rows = build_audit_rows(evaluations, annotations, register, {}, cascade)
        summary = aggregate_summary(evaluations, rows, register, {}, cascade)
        self.assertEqual(summary["transition_state_counts"], counts)
        finding = summary["decision"]["supporting_findings"][-1]
        for state, count in counts.items():
          self.assertIn(f"{count} {state}", finding)
        if "unknown" in counts:
          self.assertIn(f"Review the {counts['unknown']} unknown states", finding)
        else:
          self.assertIn("No audited state is unknown", finding)
          self.assertNotIn("remain unknown", summary["decision"]["promotion_status"])
        self.assertIn("not a locked external test", summary["decision"]["promotion_status"])
        with tempfile.TemporaryDirectory() as directory:
          output = Path(directory) / "summary.md"
          write_markdown_summary(summary, output, "test")
          self.assertIn(finding, output.read_text())

  def test_audit_crop_rotates_clockwise_like_the_review_and_runtime(self) -> None:
    source = Image.new("RGB", (2, 4), "red")
    source.paste("blue", (0, 2, 2, 4))
    for rotation, size, first_pixel in (
      (0, (2, 4), (255, 0, 0)), (90, (4, 2), (0, 0, 255)),
      (180, (2, 4), (0, 0, 255)), (270, (4, 2), (255, 0, 0)),
    ):
      with self.subTest(rotation=rotation):
        crop = display_oriented_crop(source, rotation)
        self.assertEqual(crop.size, size)
        self.assertEqual(crop.getpixel((0, 0)), first_pixel)
    with self.assertRaisesRegex(ValueError, "Unsupported display rotation"):
      display_oriented_crop(source, 45)

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
  def test_oracle_checkpoint_must_match_recorded_evaluation_hash(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / "best.pt"
      checkpoint.write_bytes(b"evaluated model")
      digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
      valid = {"fold": 0, "checkpoint": str(checkpoint), "checkpoint_sha256": digest, "predictions": []}
      args = SimpleNamespace(device="cpu", skip_oracles=True, skip_roi_cascade=True)
      for recorded_hash in (None, "", "different-model"):
        with self.subTest(recorded_hash=recorded_hash), patch("ultralytics.YOLO") as model:
          with self.assertRaisesRegex(ValueError, "Checkpoint SHA256"):
            run_oracles([valid, {**valid, "fold": 1, "checkpoint_sha256": recorded_hash}], {}, root, args)
          model.assert_not_called()
      checkpoint.write_bytes(b"replaced checkpoint at same path")
      with patch("ultralytics.YOLO") as model:
        with self.assertRaisesRegex(ValueError, "Checkpoint SHA256"):
          run_oracles([valid], {}, root, args)
        model.assert_not_called()
      checkpoint.write_bytes(b"evaluated model")
      with patch("ultralytics.YOLO") as model:
        self.assertEqual(run_oracles([valid], {}, root, args), ({}, {}, {}))
        model.assert_called_once_with(str(checkpoint))

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
      [Image.new("RGB", (20, 30), (255, 40, 10)), Image.new("RGB", (40, 50))],
      imgsz=1280,
      device="cpu",
      confidence=0.25,
      iou=0.7,
      max_detections=300,
    )

    self.assertEqual(results, ["result-1", "result-2"])
    self.assertEqual(len(model.sources), 2)
    self.assertEqual(model.sources[0].shape, (30, 20, 3))
    self.assertEqual(model.sources[0][0, 0].tolist(), [10, 40, 255])
    self.assertTrue(model.sources[0].flags.c_contiguous)
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


class ReportTests(unittest.TestCase):
  def test_oracle_conclusions_do_not_invent_a_failure_when_every_path_is_correct(self) -> None:
    record = build_sequence_record("meter.jpg", "1234", 0,
      [detection(digit, 0.2 + p * 0.2) for p, digit in enumerate((1, 2, 3, 4))])
    evaluations = [{"fold": 0, "predictions": [record]}]
    annotations = {"meter.jpg": [
      box(digit, 0.2 + p * 0.2, position=p) for p, digit in enumerate((1, 2, 3, 4))
    ]}
    register = {"meter.jpg": record}
    cascade = {"meter.jpg": {**record, "roi": {"status": "accepted", "truth_register_coverage": 1.0}}}
    for correct_count in (0, 2, 4):
      with self.subTest(correct_count=correct_count):
        apertures = {("meter.jpg", p): {
          "predicted_digit": digit if p < correct_count else 9,
          "correct": p < correct_count,
        } for p, digit in enumerate((1, 2, 3, 4))}
        rows = build_audit_rows(evaluations, annotations, register, apertures, cascade)
        summary = aggregate_summary(evaluations, rows, register, apertures, cascade)
        decision = summary["decision"]
        self.assertIn("1/1", decision["finding"])
        self.assertNotIn("bottleneck", decision["finding"])
        self.assertNotIn("gain", decision["finding"])
        finding = decision["supporting_findings"][3]
        self.assertIn(f"{correct_count}/4", finding)
        self.assertIn(f"{correct_count / 4:.1%}", finding)
        self.assertNotIn("should preserve whole-register context", finding)
        with tempfile.TemporaryDirectory() as directory:
          output = Path(directory) / "summary.md"
          write_markdown_summary(summary, output, "test")
          self.assertIn(finding, output.read_text())

  def test_partial_roi_rejections_are_not_diagnosed_as_digit_padding_failures(self) -> None:
    records = [
      build_sequence_record(f"meter-{index}.jpg", "1234", 0,
                            [detection(digit, 0.2 + p * 0.2) for p, digit in enumerate((1, 2, 3, 4))])
      for index in range(5)
    ]
    evaluations = [{"fold": 0, "predictions": records}]
    annotations = {record["filename"]: [
      box(digit, 0.2 + p * 0.2, position=p) for p, digit in enumerate((1, 2, 3, 4))
    ] for record in records}
    register = {record["filename"]: record for record in records}
    for rejection in ("no-detection", "invalid-center-x", "empty-crop"):
      with self.subTest(rejection=rejection):
        cascade = {record["filename"]: {
          **(record if index == 0 else build_sequence_record(record["filename"], "1234", 0, [])),
          "roi": {"status": "accepted", "truth_register_coverage": 1.0} if index == 0 else {"status": rejection},
        } for index, record in enumerate(records)}
        rows = build_audit_rows(evaluations, annotations, register, {}, cascade)
        summary = aggregate_summary(evaluations, rows, register, {}, cascade)
        self.assertEqual(summary["roi_cascade_diagnostics"]["roi_rejected_count"], 4)
        self.assertEqual(summary["roi_cascade_diagnostics"]["coverage_image_count"], 1)
        decision = summary["decision"]
        self.assertIn("4/5", decision["finding"])
        self.assertIn("ROI rejection", decision["recommended_next_step"])
        self.assertNotIn("padding", decision["recommended_next_step"])
        self.assertIn("1/5", decision["supporting_findings"][2])
        self.assertNotIn("every reviewed register", " ".join(decision["supporting_findings"]))

  def test_reports_handle_zero_or_one_readable_result_and_skipped_paths(self) -> None:
    for readable_count in (0, 1, 2):
      for mode in ("both", "register", "cascade", "skipped", "roi-rejected"):
        with self.subTest(readable=readable_count, mode=mode), tempfile.TemporaryDirectory() as directory:
          records = [
            build_sequence_record(
              f"meter-{index}.jpg", "1234", 0,
              [detection(digit, 0.2 + position * 0.2) for position, digit in enumerate((1, 2, 3, 5))]
              if index < readable_count else [],
            )
            for index in range(2)
          ]
          evaluations = [{"fold": 0, "predictions": records}]
          annotations = {
            record["filename"]: [box(digit, 0.2 + position * 0.2, position=position) for position, digit in enumerate((1, 2, 3, 4))]
            for record in records
          }
          register = {record["filename"]: record for record in records} if mode in ("both", "register", "roi-rejected") else {}
          cascade = {
            record["filename"]: {
              **(build_sequence_record(record["filename"], "1234", 0, []) if mode == "roi-rejected" else record),
              "roi": {"status": "not-detected" if mode == "roi-rejected" else "accepted",
                      "confidence": None if mode == "roi-rejected" else 0.9,
                      "truth_register_coverage": None if mode == "roi-rejected" else 1.0},
            }
            for record in records
          } if mode in ("both", "cascade", "roi-rejected") else {}
          rows = build_audit_rows(evaluations, annotations, register, {}, cascade)
          summary = aggregate_summary(evaluations, rows, register, {}, cascade)
          for key in ("register_oracle_diagnostics", "roi_cascade_diagnostics"):
            if key in summary:
              diagnostics = summary[key]
              effective_count = 0 if mode == "roi-rejected" and key.startswith("roi") else readable_count
              self.assertEqual(diagnostics["median_absolute_error"], 1.0 if effective_count else None)
              self.assertEqual(diagnostics["mean_absolute_error_excluding_worst"], 1.0 if effective_count == 2 else None)
              self.assertEqual(diagnostics["worst_absolute_error"], 1 if effective_count else None)
          if mode == "roi-rejected":
            self.assertIsNone(summary["roi_cascade_diagnostics"]["minimum_truth_register_coverage"])
            self.assertIn("no readable sequence", summary["decision"]["finding"])
          for row in rows:
            row["crop_hrefs"] = ["crop.jpg"] * 4
            row["overlay_href"] = "overlay.jpg"
          path = Path(directory) / "index.html"
          write_html_report(rows, summary, path, "test")
          write_markdown_summary(summary, Path(directory) / "summary.md", "test")
          self.assertIn("Jarvis Full-Image Digit Error Audit", path.read_text())
          if readable_count < 2 or mode == "roi-rejected":
            self.assertIn("n/a", path.read_text())


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
