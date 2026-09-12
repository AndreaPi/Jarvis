from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import backend.evaluate_full_image_digit_detector as evaluator
import backend.evaluate_full_image_digit_shadow_sensitivity as sensitivity

from backend.evaluate_full_image_digit_detector import (
  build_sequence_record,
  sort_detections_in_reading_order,
  summarize_sequence_records,
  validation_annotation_groups,
  validate_checkpoint_fold,
)


def detection(
  class_id: int,
  x_center: float,
  y_center: float,
) -> dict[str, float | int]:
  return {
    "class_id": class_id,
    "confidence": 0.9,
    "x_center": x_center,
    "y_center": y_center,
    "width": 0.1,
    "height": 0.1,
  }


class FullImageDigitDetectorEvaluationTests(unittest.TestCase):
  def test_rejects_unverified_checkpoint_fold_before_dataset_or_inference(self) -> None:
    cases = [
      ("wrong fold with default CLI fold", '{"selected_fold": 1}', "validation fold 1"),
      ("missing provenance", None, "original dataset_provenance.json"),
      ("malformed JSON", "{", "Invalid checkpoint training provenance JSON"),
      ("missing fold", "{}", "integer selected_fold"),
      ("invalid root", "[]", "integer selected_fold"),
      ("boolean fold", '{"selected_fold": false}', "integer selected_fold"),
      ("string fold", '{"selected_fold": "0"}', "integer selected_fold"),
      ("fractional fold", '{"selected_fold": 0.5}', "integer selected_fold"),
      ("out of range", '{"selected_fold": 5}', "integer selected_fold"),
    ]
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / "weights" / "best.pt"
      checkpoint.parent.mkdir()
      checkpoint.write_bytes(b"unused checkpoint")
      provenance = root / "dataset_provenance.json"
      folds = root / "folds.csv"
      folds.write_text("filename,fold\nold-validation.jpg,0\nold-training.jpg,1\n")
      original_hash = evaluator.file_sha256(folds)
      cases.extend([
        ("missing membership", '{"selected_fold": 0}', "original cv_folds_sha256"),
        ("invalid membership hash", '{"selected_fold": 0, "cv_folds_sha256": 123}',
         "original cv_folds_sha256"),
        ("reassigned training image", json.dumps({
          "selected_fold": 0, "cv_folds_sha256": original_hash,
        }), "does not match"),
      ])
      # Keep fold 0 present, but swap its validation source with a training source.
      folds.write_text("filename,fold\nold-validation.jpg,1\nold-training.jpg,0\n")
      for label, contents, message in cases:
        with self.subTest(label=label):
          if contents is None:
            provenance.unlink(missing_ok=True)
          else:
            provenance.write_text(contents, encoding="utf-8")
          for runner in (evaluator, sensitivity):
            with (
              self.subTest(runner=runner.__name__),
              patch("sys.argv", ["evaluate", "--checkpoint", str(checkpoint),
                                 "--fold", "0", "--folds", str(folds)]),
              patch.object(runner, "read_csv_rows") as read_dataset,
              patch.object(evaluator.tempfile, "mkdtemp") as materialize,
              patch.object(sensitivity, "RoiDetector") as roi_model,
              patch.object(sensitivity, "FullImageDigitShadow") as digit_model,
              patch.object(sensitivity.Image, "open") as open_image,
            ):
              with self.assertRaisesRegex((ValueError, FileNotFoundError), message):
                runner.main()
              read_dataset.assert_not_called()
              materialize.assert_not_called()
              roi_model.assert_not_called()
              digit_model.assert_not_called()
              open_image.assert_not_called()

  def test_accepts_recorded_fold_independently_of_checkpoint_directory_name(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      run_dir = Path(directory) / "misleading-fold0"
      run_dir.mkdir()
      provenance = run_dir / "dataset_provenance.json"
      folds = run_dir / "folds.csv"
      folds.write_text("filename,fold\nvalidation.jpg,1\ntraining.jpg,0\n")
      folds_hash = evaluator.file_sha256(folds)
      provenance.write_text(json.dumps({
        "selected_fold": 1, "cv_folds_sha256": folds_hash,
      }), encoding="utf-8")
      for checkpoint in (run_dir / "weights" / "best.pt", run_dir / "best.pt"):
        with self.subTest(checkpoint=str(checkpoint)):
          result = validate_checkpoint_fold(checkpoint, 1, folds)
          self.assertEqual(result, {
            "path": str(provenance),
            "sha256": evaluator.file_sha256(provenance),
            "selected_fold": 1,
            "cv_folds_path": str(folds),
            "cv_folds_sha256": folds_hash,
            "fold_assignments": {"validation.jpg": 1, "training.jpg": 0},
          })

  def test_verified_fold_membership_is_retained_after_manifest_changes(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / "best.pt"
      folds = root / "folds.csv"
      folds.write_text("filename,fold\nvalidation.jpg,0\ntraining.jpg,1\n")
      (root / "dataset_provenance.json").write_text(json.dumps({
        "selected_fold": 0, "cv_folds_sha256": evaluator.file_sha256(folds),
      }))
      verified = validate_checkpoint_fold(checkpoint, None, folds)
      folds.write_text("filename,fold\nvalidation.jpg,1\ntraining.jpg,0\n")
      groups = {filename: [{"split": "train", "position": "0"}] for filename in verified["fold_assignments"]}
      self.assertEqual(list(validation_annotation_groups(
        groups, verified["fold_assignments"], verified["selected_fold"],
      )), ["validation.jpg"])

  def test_sorts_detections_for_each_supported_rotation(self) -> None:
    horizontal = [
      detection(3, 0.7, 0.5),
      detection(1, 0.1, 0.5),
      detection(4, 0.9, 0.5),
      detection(2, 0.3, 0.5),
    ]
    vertical = [
      detection(2, 0.5, 0.3),
      detection(4, 0.5, 0.9),
      detection(1, 0.5, 0.1),
      detection(3, 0.5, 0.7),
    ]

    self.assertEqual(
      [item["class_id"] for item in sort_detections_in_reading_order(horizontal, 0)],
      [1, 2, 3, 4],
    )
    self.assertEqual(
      [item["class_id"] for item in sort_detections_in_reading_order(horizontal, 180)],
      [4, 3, 2, 1],
    )
    self.assertEqual(
      [item["class_id"] for item in sort_detections_in_reading_order(vertical, 270)],
      [1, 2, 3, 4],
    )
    self.assertEqual(
      [item["class_id"] for item in sort_detections_in_reading_order(vertical, 90)],
      [4, 3, 2, 1],
    )

  def test_summarizes_exact_wrong_and_no_read_predictions(self) -> None:
    exact = build_sequence_record(
      "meter_exact.JPEG",
      "1234",
      0,
      [
        detection(1, 0.1, 0.5),
        detection(2, 0.3, 0.5),
        detection(3, 0.7, 0.5),
        detection(4, 0.9, 0.5),
      ],
    )
    wrong = build_sequence_record(
      "meter_wrong.JPEG",
      "1234",
      0,
      [
        detection(1, 0.1, 0.5),
        detection(2, 0.3, 0.5),
        detection(3, 0.7, 0.5),
        detection(5, 0.9, 0.5),
      ],
    )
    no_read = build_sequence_record(
      "meter_no_read.JPEG",
      "1234",
      0,
      [
        detection(1, 0.1, 0.5),
        detection(2, 0.3, 0.5),
        detection(3, 0.7, 0.5),
      ],
    )

    summary = summarize_sequence_records([exact, wrong, no_read])
    self.assertEqual(summary["image_count"], 3)
    self.assertEqual(summary["readable_count"], 2)
    self.assertEqual(summary["no_read_count"], 1)
    self.assertAlmostEqual(summary["no_read_rate"], 1 / 3)
    self.assertEqual(summary["exact_match_count"], 1)
    self.assertAlmostEqual(summary["exact_match_rate"], 1 / 3)
    self.assertEqual(summary["readable_exact_match_rate"], 0.5)
    self.assertEqual(summary["readable_digit_accuracy"], 0.875)
    self.assertEqual(summary["readable_mae"], 0.5)

  def test_selects_only_the_requested_train_source_fold(self) -> None:
    grouped = {
      "meter_a.JPEG": [{"filename": "meter_a.JPEG", "split": "train", "position": "0"}],
      "meter_b.JPEG": [{"filename": "meter_b.JPEG", "split": "train", "position": "0"}],
      "meter_test.JPEG": [
        {"filename": "meter_test.JPEG", "split": "test", "position": "0"}
      ],
    }
    folds = {
      "meter_a.JPEG": 0,
      "meter_b.JPEG": 1,
    }

    selected = validation_annotation_groups(grouped, folds, selected_fold=1)
    self.assertEqual(list(selected), ["meter_b.JPEG"])
    self.assertNotIn("meter_test.JPEG", selected)


if __name__ == "__main__":
  unittest.main()
