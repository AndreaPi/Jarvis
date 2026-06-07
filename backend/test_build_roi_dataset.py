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
  def run_builder(self, root: Path, new_split: str | None = None) -> None:
    command = [
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
    ]
    if new_split is not None:
      command.extend(["--new-split", new_split])

    subprocess.run(
      command,
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

  def test_changing_existing_roi_updates_generated_label(self) -> None:
    with tempfile.TemporaryDirectory(prefix="jarvis-roi-build-") as temp_dir:
      root = Path(temp_dir)
      assets_dir = root / "assets"
      csv_path = root / "meter_readings.csv"
      manifest_path = root / "roi_boxes_manifest.json"
      out_dir = root / "roi_dataset"

      filename = "meter_20260524.JPEG"
      original_manifest_entry = {
        "filename": filename,
        "rectNorm": {
          "x": 0.1,
          "y": 0.2,
          "width": 0.3,
          "height": 0.4,
        },
      }
      corrected_manifest_entry = {
        "filename": filename,
        "rectNorm": {
          "x": 0.2,
          "y": 0.3,
          "width": 0.2,
          "height": 0.2,
        },
      }

      write_image(assets_dir / filename)
      write_csv(csv_path, [(filename, "2335")])
      write_manifest(manifest_path, [original_manifest_entry])
      self.run_builder(root)

      label_path = out_dir / "labels" / "train" / "meter_20260524.txt"
      self.assertEqual(label_path.read_text(encoding="utf-8"), "0 0.250000 0.400000 0.300000 0.400000\n")

      write_manifest(manifest_path, [corrected_manifest_entry])
      self.run_builder(root)

      self.assertEqual(label_path.read_text(encoding="utf-8"), "0 0.300000 0.400000 0.200000 0.200000\n")

  def test_removing_roi_row_prunes_generated_image_label_and_preview(self) -> None:
    with tempfile.TemporaryDirectory(prefix="jarvis-roi-build-") as temp_dir:
      root = Path(temp_dir)
      assets_dir = root / "assets"
      csv_path = root / "meter_readings.csv"
      manifest_path = root / "roi_boxes_manifest.json"
      out_dir = root / "roi_dataset"

      kept_filename = "meter_20260524.JPEG"
      removed_filename = "meter_20260603.JPEG"
      kept_manifest_entry = {
        "filename": kept_filename,
        "rectNorm": {
          "x": 0.1,
          "y": 0.2,
          "width": 0.3,
          "height": 0.4,
        },
      }
      removed_manifest_entry = {
        "filename": removed_filename,
        "rectNorm": {
          "x": 0.2,
          "y": 0.3,
          "width": 0.2,
          "height": 0.2,
        },
      }

      write_image(assets_dir / kept_filename)
      write_image(assets_dir / removed_filename)
      write_csv(csv_path, [(kept_filename, "2335"), (removed_filename, "2336")])
      write_manifest(manifest_path, [kept_manifest_entry, removed_manifest_entry])
      self.run_builder(root)

      removed_image_path = out_dir / "images" / "train" / removed_filename
      removed_label_path = out_dir / "labels" / "train" / "meter_20260603.txt"
      removed_preview_path = out_dir / "previews" / "meter_20260603_bbox.jpg"
      self.assertTrue(removed_image_path.exists())
      self.assertTrue(removed_label_path.exists())
      self.assertTrue(removed_preview_path.exists())

      write_csv(csv_path, [(kept_filename, "2335")])
      write_manifest(manifest_path, [kept_manifest_entry])
      self.run_builder(root)

      self.assertFalse(removed_image_path.exists())
      self.assertFalse(removed_label_path.exists())
      self.assertFalse(removed_preview_path.exists())
      self.assertTrue((out_dir / "images" / "train" / kept_filename).exists())
      self.assertTrue((out_dir / "labels" / "train" / "meter_20260524.txt").exists())
      self.assertTrue((out_dir / "previews" / "meter_20260524_bbox.jpg").exists())

  def test_new_split_applies_only_to_genuinely_new_rows(self) -> None:
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
          "x": 0.1,
          "y": 0.2,
          "width": 0.3,
          "height": 0.4,
        },
      }
      new_manifest_entry = {
        "filename": new_filename,
        "rectNorm": {
          "x": 0.2,
          "y": 0.3,
          "width": 0.2,
          "height": 0.2,
        },
      }

      write_image(assets_dir / old_filename)
      write_csv(csv_path, [(old_filename, "2335")])
      write_manifest(manifest_path, [old_manifest_entry])
      self.run_builder(root)

      write_image(assets_dir / new_filename)
      write_csv(csv_path, [(old_filename, "2335"), (new_filename, "2336")])
      write_manifest(manifest_path, [old_manifest_entry, new_manifest_entry])
      self.run_builder(root, new_split="val")

      split_payload = json.loads((out_dir / "splits.json").read_text(encoding="utf-8"))["assignments"]
      self.assertEqual(split_payload[old_filename], "train")
      self.assertEqual(split_payload[new_filename], "val")
      self.assertTrue((out_dir / "labels" / "train" / "meter_20260524.txt").exists())
      self.assertTrue((out_dir / "labels" / "val" / "meter_20260603.txt").exists())

  def test_existing_split_assignments_survive_csv_order_changes(self) -> None:
    with tempfile.TemporaryDirectory(prefix="jarvis-roi-build-") as temp_dir:
      root = Path(temp_dir)
      assets_dir = root / "assets"
      csv_path = root / "meter_readings.csv"
      manifest_path = root / "roi_boxes_manifest.json"
      out_dir = root / "roi_dataset"

      filenames = [
        "meter_20260524.JPEG",
        "meter_20260603.JPEG",
        "meter_20260604.JPEG",
        "meter_20260605.JPEG",
      ]
      manifest_entries = [
        {
          "filename": filename,
          "rectNorm": {
            "x": 0.1,
            "y": 0.2,
            "width": 0.3,
            "height": 0.4,
          },
        }
        for filename in filenames
      ]

      for filename in filenames:
        write_image(assets_dir / filename)
      write_csv(csv_path, [(filename, f"233{index}") for index, filename in enumerate(filenames)])
      write_manifest(manifest_path, manifest_entries)
      self.run_builder(root)

      split_path = out_dir / "splits.json"
      split_payload_before = json.loads(split_path.read_text(encoding="utf-8"))["assignments"]
      self.assertEqual(split_payload_before[filenames[0]], "train")
      self.assertEqual(split_payload_before[filenames[1]], "train")
      self.assertEqual(split_payload_before[filenames[2]], "val")
      self.assertEqual(split_payload_before[filenames[3]], "test")

      reversed_filenames = list(reversed(filenames))
      write_csv(csv_path, [(filename, f"233{index}") for index, filename in enumerate(reversed_filenames)])
      self.run_builder(root)

      split_payload_after = json.loads(split_path.read_text(encoding="utf-8"))["assignments"]
      self.assertEqual(split_payload_after, split_payload_before)
      self.assertTrue((out_dir / "labels" / "train" / "meter_20260524.txt").exists())
      self.assertTrue((out_dir / "labels" / "train" / "meter_20260603.txt").exists())
      self.assertTrue((out_dir / "labels" / "val" / "meter_20260604.txt").exists())
      self.assertTrue((out_dir / "labels" / "test" / "meter_20260605.txt").exists())


if __name__ == "__main__":
  unittest.main()
