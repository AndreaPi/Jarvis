"""Tests for the bounded full-image shadow sensitivity runner."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import backend.evaluate_full_image_digit_shadow_sensitivity as sensitivity
from backend.test_train_full_image_digit_detector import annotation_rows

from backend.evaluate_full_image_digit_shadow_sensitivity import (
  parse_float_grid,
  rank_setting,
)


class FullImageDigitShadowSensitivityTests(unittest.TestCase):
  def test_excluded_pending_sources_never_reach_sensitivity_inference(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      checkpoint = root / "weights/best.pt"
      checkpoint.parent.mkdir()
      checkpoint.write_bytes(b"mock model")
      folds = root / "folds.csv"
      folds.write_text("filename,fold\nactive.jpg,4\nexcluded.jpg,4\n")
      (root / "dataset_provenance.json").write_text(json.dumps({
        "selected_fold": 4, "cv_folds_sha256": sensitivity.file_sha256(folds),
      }))
      exclusions = root / "exclusions.csv"
      exclusions.write_text(
        "filename,scope,reason,retention\n"
        "excluded.jpg,full_image_digit_detection,legacy example,legacy_stress\n"
      )
      rows = annotation_rows("active.jpg", "1234", "train")
      rows += annotation_rows("excluded.jpg", "1234", "train", review_status="pending")
      annotations = root / "annotations.csv"
      with annotations.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
      image_root = root / "images"
      (image_root / "train").mkdir(parents=True)
      Image.new("RGB", (10, 10)).save(image_root / "train/active.jpg")
      original = annotations.read_bytes(), exclusions.read_bytes()
      argv = ["sensitivity", "--checkpoint", str(checkpoint), "--folds", str(folds),
              "--annotations", str(annotations), "--source-exclusions", str(exclusions),
              "--source-images", str(image_root), "--roi-model", str(checkpoint),
              "--target", "active.jpg", "--confidences", "0.25", "--ious", "0.7",
              "--output-root", str(root / "output")]
      with (
        patch("sys.argv", argv), patch.object(sensitivity, "RoiDetector"),
        patch.object(sensitivity, "FullImageDigitShadow") as model,
        patch.object(sensitivity, "render_target_grid"), patch("builtins.print"),
      ):
        model.return_value.predict.return_value = {"detections": []}
        sensitivity.main()
        model.return_value.predict.assert_called_once()
      payload = json.loads(next((root / "output").glob("*/summary.json")).read_text())
      self.assertEqual([row["filename"] for row in payload["settings"][0]["predictions"]], ["active.jpg"])
      self.assertEqual(payload["excluded_sources"], ["excluded.jpg"])
      self.assertEqual(payload["source_exclusions_sha256"], sensitivity.file_sha256(exclusions))
      self.assertEqual((annotations.read_bytes(), exclusions.read_bytes()), original)
      # Selecting an excluded source itself must fail before opening images.
      argv[argv.index("--target") + 1] = "excluded.jpg"
      with patch("sys.argv", argv), patch.object(sensitivity.Image, "open") as open_image:
        with self.assertRaisesRegex(ValueError, "Target is not assigned"):
          sensitivity.main()
        open_image.assert_not_called()

  def test_grid_parsing_and_ranking_guardrails(self) -> None:
    self.assertEqual(
      parse_float_grid("0.25,0.10,0.25", minimum=0.0, maximum=1.0),
      [0.10, 0.25],
    )
    with self.assertRaises(ValueError):
      parse_float_grid("1.2", minimum=0.0, maximum=1.0)

    exact_with_no_read = {
      "confidence": 0.25,
      "iou": 0.7,
      "sequence_metrics": {
        "no_read_count": 1,
        "exact_match_count": 5,
        "readable_mae": 1.0,
      },
    }
    readable_setting = {
      "confidence": 0.20,
      "iou": 0.7,
      "sequence_metrics": {
        "no_read_count": 0,
        "exact_match_count": 4,
        "readable_mae": 2.0,
      },
    }
    shared = {
      "confidence": 0.20,
      "sequence_metrics": {
        "no_read_count": 0,
        "exact_match_count": 5,
        "readable_mae": 15.0,
      },
    }
    ranking_cases = (
      (
        "avoid no-reads before maximizing exact matches",
        readable_setting,
        exact_with_no_read,
      ),
      (
        "prefer the baseline IoU when metrics tie",
        {**shared, "iou": 0.70},
        {**shared, "iou": 0.50},
      ),
    )
    for label, preferred, rejected in ranking_cases:
      with self.subTest(label=label):
        self.assertGreater(rank_setting(preferred), rank_setting(rejected))


if __name__ == "__main__":
  unittest.main()
