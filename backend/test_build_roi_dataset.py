from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


BACKEND_DIR = Path(__file__).resolve().parent
BUILD_SCRIPT = BACKEND_DIR / "build_roi_dataset.py"


def write_image(path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  Image.new("RGB", (16, 12), color=(120, 90, 60)).save(path, "JPEG")


def write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["filename", "value"])
    writer.writerows(rows)


def write_manifest(path: Path, entries: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


class BuildRoiDatasetTests(unittest.TestCase):
  def run_builder(self, root: Path) -> None:
    subprocess.run(
      [
        sys.executable,
        str(BUILD_SCRIPT),
        "--csv",
        str(root / "meter_readings.csv"),
        "--roi-json",
        str(root / "roi_boxes_manifest.json"),
        "--assets-dir",
        str(root / "assets"),
        "--out-dir",
        str(root / "roi_dataset"),
        "--preview-dir",
        str(root / "roi_dataset" / "previews"),
        "--splits-json",
        str(root / "roi_dataset" / "splits.json"),
      ],
      cwd=BACKEND_DIR,
      check=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
    )

  def test_adding_new_roi_preserves_existing_generated_label_and_split(self) -> None:
    with tempfile.TemporaryDirectory(prefix="jarvis-roi-build-") as temp_dir:
      root = Path(temp_dir)
      assets_dir = root / "assets"
      csv_path = root / "meter_readings.csv"
      manifest_path = root / "roi_boxes_manifest.json"
      out_dir = root / "roi_dataset"

      old_filename = "meter_20260524.JPEG"
      new_filename = "meter_20260603.JPEG"
      old_manifest_entry = {
        "filename": old_filename,
        "rectNorm": {
          "x": 0.4006175,
          "y": 0.4152445,
          "width": 0.068259,
          "height": 0.126767,
        },
      }
      new_manifest_entry = {
        "filename": new_filename,
        "rectNorm": {
          "x": 0.3789185,
          "y": 0.3967565,
          "width": 0.070631,
          "height": 0.139459,
        },
      }

      write_image(assets_dir / old_filename)
      write_csv(csv_path, [(old_filename, "2335")])
      write_manifest(manifest_path, [old_manifest_entry])
      self.run_builder(root)

      old_label_path = out_dir / "labels" / "train" / "meter_20260524.txt"
      old_image_path = out_dir / "images" / "train" / old_filename
      split_path = out_dir / "splits.json"

      old_label_before = old_label_path.read_text(encoding="utf-8")
      old_image_before = old_image_path.read_bytes()
      old_split_before = json.loads(split_path.read_text(encoding="utf-8"))["assignments"][old_filename]

      write_image(assets_dir / new_filename)
      write_csv(csv_path, [(old_filename, "2335"), (new_filename, "2336")])
      write_manifest(manifest_path, [old_manifest_entry, new_manifest_entry])
      self.run_builder(root)

      old_label_after = old_label_path.read_text(encoding="utf-8")
      old_image_after = old_image_path.read_bytes()
      split_payload = json.loads(split_path.read_text(encoding="utf-8"))["assignments"]

      self.assertEqual(old_label_after, old_label_before)
      self.assertEqual(old_image_after, old_image_before)
      self.assertEqual(split_payload[old_filename], old_split_before)
      self.assertEqual(split_payload[new_filename], "train")
      self.assertTrue((out_dir / "labels" / "train" / "meter_20260603.txt").exists())


if __name__ == "__main__":
  unittest.main()
