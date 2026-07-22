from __future__ import annotations

import unittest

from PIL import Image

from backend.runtime_digit_pipeline import build_cell_rects, resolve_crop_rect


class CropRectTests(unittest.TestCase):
  def setUp(self) -> None:
    self.image = Image.new("RGB", (100, 50))

  def test_resolve_crop_rect_intersects_each_image_edge(self) -> None:
    cases = [
      ({"x": -10, "y": 5, "width": 30, "height": 20}, (0, 5, 20, 25)),
      ({"x": 90, "y": 5, "width": 20, "height": 20}, (90, 5, 100, 25)),
      ({"x": 10, "y": -5, "width": 20, "height": 15}, (10, 0, 30, 10)),
      ({"x": 10, "y": 45, "width": 20, "height": 10}, (10, 45, 30, 50)),
    ]

    for rect, expected in cases:
      with self.subTest(rect=rect):
        crop = resolve_crop_rect(self.image, rect)
        self.assertEqual((crop.left, crop.top, crop.right, crop.bottom), expected)

  def test_resolve_crop_rect_keeps_fully_external_crop_non_empty(self) -> None:
    left = resolve_crop_rect(self.image, {"x": -20, "y": 0, "width": 5, "height": 10})
    right = resolve_crop_rect(self.image, {"x": 120, "y": 0, "width": 5, "height": 10})

    self.assertEqual((left.left, left.right), (0, 1))
    self.assertEqual((right.left, right.right), (99, 100))

  def test_cell_overlap_preserves_trained_boundary_geometry(self) -> None:
    rects = build_cell_rects(self.image, count=4, overlap_ratio=0.1)

    self.assertEqual((rects[0].left, rects[0].right), (0, 30))
    self.assertEqual((rects[1].left, rects[1].right), (22, 52))
    self.assertEqual((rects[3].left, rects[3].right), (72, 100))


if __name__ == "__main__":
  unittest.main()
