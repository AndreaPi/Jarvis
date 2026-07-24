from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from backend.build_full_image_digit_dataset import (
  build_digit_boxes,
  seed_or_preserve_cv_folds,
)
from backend.import_full_image_digit_annotations import merge_reviewed_export


BACKEND_DIR = Path(__file__).resolve().parent
BUILD_SCRIPT = BACKEND_DIR / "build_full_image_digit_dataset.py"


def write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)


class FullImageDigitDatasetTests(unittest.TestCase):
  def test_horizontal_and_vertical_reading_order(self) -> None:
    horizontal_roi = (0.5, 0.5, 0.4, 0.2)
    vertical_roi = (0.5, 0.5, 0.2, 0.4)

    left_to_right = build_digit_boxes("1234", horizontal_roi, 0, 0.0, 0.0)
    right_to_left = build_digit_boxes("1234", horizontal_roi, 180, 0.0, 0.0)
    bottom_to_top = build_digit_boxes("1234", vertical_roi, 90, 0.0, 0.0)
    top_to_bottom = build_digit_boxes("1234", vertical_roi, 270, 0.0, 0.0)

    self.assertEqual(
      [round(float(box["x_center"]), 2) for box in left_to_right],
      [0.35, 0.45, 0.55, 0.65],
    )
    self.assertEqual(
      [round(float(box["x_center"]), 2) for box in right_to_left],
      [0.65, 0.55, 0.45, 0.35],
    )
    self.assertEqual(
      [round(float(box["y_center"]), 2) for box in bottom_to_top],
      [0.65, 0.55, 0.45, 0.35],
    )
    self.assertEqual(
      [round(float(box["y_center"]), 2) for box in top_to_bottom],
      [0.35, 0.45, 0.55, 0.65],
    )

  def test_builder_preserves_existing_review_annotations(self) -> None:
    with tempfile.TemporaryDirectory(prefix="jarvis-full-digit-build-") as temp_dir:
      root = Path(temp_dir)
      roi_dataset = root / "roi_dataset"
      image_path = roi_dataset / "images" / "train" / "meter_test.JPEG"
      label_path = roi_dataset / "labels" / "train" / "meter_test.txt"
      readings_path = root / "meter_readings.csv"
      canonical_path = root / "canonical_windows.csv"
      out_dir = root / "full_image_digit_dataset"
      review_dir = root / "review"

      image_path.parent.mkdir(parents=True, exist_ok=True)
      label_path.parent.mkdir(parents=True, exist_ok=True)
      Image.new("RGB", (100, 200), color=(220, 210, 190)).save(image_path, "JPEG")
      label_path.write_text("0 0.500000 0.500000 0.200000 0.600000\n", encoding="utf-8")
      write_csv(readings_path, ["filename", "value"], [["meter_test.JPEG", "1234"]])
      write_csv(
        canonical_path,
        [
          "split",
          "filename",
          "applied_rotation",
          "canonical_rotate_degrees",
        ],
        [["train", "meter_test.JPEG", "90", "0"]],
      )

      command = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--readings",
        str(readings_path),
        "--roi-dataset",
        str(roi_dataset),
        "--canonical-manifest",
        str(canonical_path),
        "--out-dir",
        str(out_dir),
        "--review-dir",
        str(review_dir),
      ]
      subprocess.run(command, cwd=BACKEND_DIR, check=True, capture_output=True, text=True)

      annotations_path = out_dir / "manifests" / "annotations.csv"
      with annotations_path.open("r", encoding="utf-8") as handle:
        annotations = list(csv.DictReader(handle))
      self.assertEqual(len(annotations), 4)
      self.assertTrue(all(row["review_status"] == "pending" for row in annotations))

      annotations[0]["x_center"] = "0.49000000"
      annotations[0]["review_status"] = "reviewed"
      annotations[0]["annotation_source"] = "human-makesense"
      with annotations_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(annotations[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(annotations)

      subprocess.run(command, cwd=BACKEND_DIR, check=True, capture_output=True, text=True)

      with annotations_path.open("r", encoding="utf-8") as handle:
        rebuilt = list(csv.DictReader(handle))
      self.assertEqual(rebuilt[0]["x_center"], "0.49000000")
      self.assertEqual(rebuilt[0]["review_status"], "reviewed")
      self.assertTrue((out_dir / "labels" / "train" / "meter_test.txt").exists())
      self.assertTrue((review_dir / "digit-box-contact-sheet.jpg").exists())
      with zipfile.ZipFile(review_dir / "makesense-yolo-labels.zip") as archive:
        self.assertEqual(
          sorted(archive.namelist()),
          ["labels.txt", "meter_test.txt"],
        )

      summary = json.loads(
        (out_dir / "manifests" / "summary.json").read_text(encoding="utf-8")
      )
      self.assertEqual(summary["images"], 1)
      self.assertEqual(summary["annotations"], 4)
      self.assertEqual(summary["review_counts"], {"pending": 3, "reviewed": 1})

  def test_builder_retains_legacy_stress_annotations_but_excludes_active_outputs(
    self,
  ) -> None:
    with tempfile.TemporaryDirectory(prefix="jarvis-full-digit-exclude-") as temp_dir:
      root = Path(temp_dir)
      roi_dataset = root / "roi_dataset"
      readings_path = root / "meter_readings.csv"
      canonical_path = root / "canonical_windows.csv"
      out_dir = root / "full_image_digit_dataset"
      review_dir = root / "review"
      filenames = ("meter_active.JPEG", "meter_legacy.JPEG")

      for filename in filenames:
        image_path = roi_dataset / "images" / "train" / filename
        label_path = roi_dataset / "labels" / "train" / f"{Path(filename).stem}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 200), color=(220, 210, 190)).save(
          image_path,
          "JPEG",
        )
        label_path.write_text(
          "0 0.500000 0.500000 0.200000 0.600000\n",
          encoding="utf-8",
        )

      write_csv(
        readings_path,
        ["filename", "value"],
        [
          ["meter_active.JPEG", "1234"],
          ["meter_legacy.JPEG", "5678"],
        ],
      )
      write_csv(
        canonical_path,
        [
          "split",
          "filename",
          "applied_rotation",
          "canonical_rotate_degrees",
        ],
        [
          ["train", "meter_active.JPEG", "90", "0"],
          ["train", "meter_legacy.JPEG", "90", "0"],
        ],
      )
      write_csv(
        out_dir / "manifests" / "source_exclusions.csv",
        ["filename", "scope", "reason", "retention", "notes"],
        [[
          "meter_legacy.JPEG",
          "full_image_digit_detection",
          "severe defocus",
          "legacy_stress",
          "retain reviewed history",
        ]],
      )

      command = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--readings",
        str(readings_path),
        "--roi-dataset",
        str(roi_dataset),
        "--canonical-manifest",
        str(canonical_path),
        "--out-dir",
        str(out_dir),
        "--review-dir",
        str(review_dir),
      ]
      subprocess.run(command, cwd=BACKEND_DIR, check=True, capture_output=True, text=True)

      with (out_dir / "manifests" / "annotations.csv").open(
        "r",
        encoding="utf-8",
      ) as handle:
        annotation_rows = list(csv.DictReader(handle))
      self.assertEqual(
        {row["filename"] for row in annotation_rows},
        set(filenames),
      )
      self.assertTrue(
        (out_dir / "labels" / "train" / "meter_active.txt").exists()
      )
      self.assertFalse(
        (out_dir / "labels" / "train" / "meter_legacy.txt").exists()
      )

      with (out_dir / "manifests" / "cv_folds.csv").open(
        "r",
        encoding="utf-8",
      ) as handle:
        fold_rows = list(csv.DictReader(handle))
      self.assertEqual(
        [row["filename"] for row in fold_rows],
        ["meter_active.JPEG"],
      )
      with zipfile.ZipFile(review_dir / "makesense-yolo-labels.zip") as archive:
        self.assertEqual(
          sorted(archive.namelist()),
          ["labels.txt", "meter_active.txt"],
        )

      summary = json.loads(
        (out_dir / "manifests" / "summary.json").read_text(encoding="utf-8")
      )
      self.assertEqual(summary["images"], 1)
      self.assertEqual(summary["source_exclusions"]["count"], 1)
      self.assertEqual(
        summary["source_exclusions"]["retained_annotation_images"],
        2,
      )
      self.assertEqual(summary["source_exclusions"]["active_images"], 1)

  def test_import_maps_boxes_by_orientation_and_rejects_wrong_classes(self) -> None:
    rows = []
    for position, digit in enumerate("1234"):
      rows.append({
        "split": "train",
        "filename": "meter_test.JPEG",
        "reading": "1234",
        "position": str(position),
        "digit": digit,
        "class_id": digit,
        "x_center": "0.50000000",
        "y_center": f"{0.65 - position * 0.10:.8f}",
        "width": "0.20000000",
        "height": "0.08000000",
        "x0_px": "40",
        "y0_px": "0",
        "x1_px": "60",
        "y1_px": "0",
        "image_width": "100",
        "image_height": "200",
        "direction_rotation": "90",
        "annotation_source": "bootstrap-roi-split",
        "review_status": "pending",
        "transition_state": "unknown",
        "notes": "",
      })

    export = {
      "meter_test": "\n".join([
        "4 0.500000 0.350000 0.180000 0.070000",
        "2 0.500000 0.550000 0.180000 0.070000",
        "1 0.500000 0.650000 0.180000 0.070000",
        "3 0.500000 0.450000 0.180000 0.070000",
      ])
    }
    merged = merge_reviewed_export(rows, export)
    self.assertEqual([row["digit"] for row in merged], list("1234"))
    self.assertEqual([row["review_status"] for row in merged], ["reviewed"] * 4)
    self.assertEqual([row["annotation_source"] for row in merged], ["human-makesense"] * 4)
    self.assertEqual([row["y_center"] for row in merged], [
      "0.65000000",
      "0.55000000",
      "0.45000000",
      "0.35000000",
    ])

    wrong_classes = {
      "meter_test": export["meter_test"].replace("4 0.500000", "9 0.500000")
    }
    with self.assertRaisesRegex(ValueError, "verified reading is 1234"):
      merge_reviewed_export(rows, wrong_classes)

    legacy_rows = []
    for position, digit in enumerate("5678"):
      legacy = dict(rows[position])
      legacy.update({
        "filename": "meter_legacy.JPEG",
        "reading": "5678",
        "position": str(position),
        "digit": digit,
        "class_id": digit,
      })
      legacy_rows.append(legacy)
    merged_with_legacy = merge_reviewed_export(
      rows + legacy_rows,
      export,
      excluded_filenames={"meter_legacy.JPEG"},
    )
    retained_legacy = [
      row
      for row in merged_with_legacy
      if row["filename"] == "meter_legacy.JPEG"
    ]
    self.assertEqual(retained_legacy, legacy_rows)

  def test_cv_folds_are_balanced_persistent_and_exclude_test(self) -> None:
    annotation_rows = []
    readings = [
      "0123",
      "1234",
      "2345",
      "3456",
      "4567",
      "5678",
      "6789",
      "7890",
      "8901",
      "9012",
    ]
    for image_index, reading in enumerate(readings):
      split = "test" if image_index == len(readings) - 1 else "train"
      for position, digit in enumerate(reading):
        annotation_rows.append({
          "split": split,
          "filename": f"meter_{image_index:02d}.JPEG",
          "reading": reading,
          "position": str(position),
          "digit": digit,
        })

    with tempfile.TemporaryDirectory(prefix="jarvis-full-digit-folds-") as temp_dir:
      path = Path(temp_dir) / "cv_folds.csv"
      rows, seeded = seed_or_preserve_cv_folds(path, annotation_rows, fold_count=3)
      self.assertEqual(seeded, 9)
      self.assertEqual(len(rows), 9)
      self.assertNotIn("meter_09.JPEG", {row["filename"] for row in rows})
      fold_sizes = {
        fold: sum(int(row["fold"]) == fold for row in rows)
        for fold in range(3)
      }
      self.assertEqual(fold_sizes, {0: 3, 1: 3, 2: 3})

      preserved = {row["filename"]: row["fold"] for row in rows}
      rebuilt, seeded = seed_or_preserve_cv_folds(path, annotation_rows, fold_count=3)
      self.assertEqual(seeded, 0)
      self.assertEqual(
        {row["filename"]: row["fold"] for row in rebuilt},
        preserved,
      )

      excluded_filename = "meter_00.JPEG"
      active_rows = [
        row
        for row in annotation_rows
        if row["filename"] != excluded_filename
      ]
      rebuilt, seeded = seed_or_preserve_cv_folds(
        path,
        active_rows,
        fold_count=3,
        excluded_filenames={excluded_filename},
      )
      self.assertEqual(seeded, 0)
      self.assertNotIn(
        excluded_filename,
        {row["filename"] for row in rebuilt},
      )


if __name__ == "__main__":
  unittest.main()
