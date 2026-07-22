from __future__ import annotations

import io
import unittest

import numpy as np
from fastapi import HTTPException
from PIL import Image

from backend.app import _load_rgb_image, _read_upload_bytes


class StubUpload:
  def __init__(self, payload: bytes):
    self.payload = payload
    self.read_sizes: list[int] = []

  async def read(self, size: int = -1) -> bytes:
    self.read_sizes.append(size)
    return self.payload if size < 0 else self.payload[:size]


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


class UploadLimitTests(unittest.IsolatedAsyncioTestCase):
  async def test_read_upload_bytes_accepts_payload_at_limit(self) -> None:
    upload = StubUpload(b"1234")

    payload = await _read_upload_bytes(upload, max_bytes=4)

    self.assertEqual(payload, b"1234")
    self.assertEqual(upload.read_sizes, [5])

  async def test_read_upload_bytes_rejects_payload_over_limit(self) -> None:
    upload = StubUpload(b"12345")

    with self.assertRaises(HTTPException) as context:
      await _read_upload_bytes(upload, max_bytes=4)

    self.assertEqual(context.exception.status_code, 413)
    self.assertEqual(upload.read_sizes, [5])


if __name__ == "__main__":
  unittest.main()
