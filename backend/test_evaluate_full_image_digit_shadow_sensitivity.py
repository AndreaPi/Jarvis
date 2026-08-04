"""Tests for the bounded full-image shadow sensitivity runner."""

from __future__ import annotations

import unittest

from backend.evaluate_full_image_digit_shadow_sensitivity import (
  parse_float_grid,
  rank_setting,
)


class FullImageDigitShadowSensitivityTests(unittest.TestCase):
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
