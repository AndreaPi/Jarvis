from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from backend.app import _load_rgb_image


class ImageLoadingTests(unittest.TestCase):
  def test_load_rgb_image_applies_exif_orientation(self) -> None:
    source = np.zeros((20, 40, 3), dtype=np.uint8)
    source[:, :20] = (240, 20, 20)
    source[:, 20:] = (20, 20, 240)

    image = Image.fromarray(source)
    exif = image.getexif()
    exif[274] = 6
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=100, subsampling=0, exif=exif)

    loaded = _load_rgb_image(buffer.getvalue())

    self.assertEqual(loaded.shape, (40, 20, 3))
    top_mean = loaded[:20].mean(axis=(0, 1))
    bottom_mean = loaded[20:].mean(axis=(0, 1))
    self.assertGreater(top_mean[0], top_mean[2])
    self.assertGreater(bottom_mean[2], bottom_mean[0])


if __name__ == "__main__":
  unittest.main()
