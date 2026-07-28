from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from backend.train_full_image_digit_detector import (
  materialize_fold_dataset,
  validate_resume_checkpoint,
)


def annotation_rows(
  filename: str,
  reading: str,
  split: str,
  review_status: str = "reviewed",
) -> list[dict[str, str]]:
  rows = []
  for position, digit in enumerate(reading):
    rows.append({
      "split": split,
      "filename": filename,
      "reading": reading,
      "position": str(position),
      "digit": digit,
      "class_id": digit,
      "x_center": f"{0.2 + position * 0.2:.8f}",
      "y_center": "0.50000000",
      "width": "0.10000000",
      "height": "0.20000000",
      "direction_rotation": "0",
      "review_status": review_status,
    })
  return rows


def write_sample(
  image_root: Path,
  label_root: Path,
  filename: str,
  reading: str,
  split: str,
) -> None:
  image_path = image_root / split / filename
  label_path = label_root / split / f"{Path(filename).stem}.txt"
  image_path.parent.mkdir(parents=True, exist_ok=True)
  label_path.parent.mkdir(parents=True, exist_ok=True)
  Image.new("RGB", (100, 100), color=(220, 210, 190)).save(image_path, "JPEG")
  lines = [
    f"{digit} {0.2 + position * 0.2:.6f} 0.500000 0.100000 0.200000"
    for position, digit in enumerate(reading)
  ]
  label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class FullImageDigitDetectorTrainingTests(unittest.TestCase):
  def test_validates_resumable_checkpoint_configuration(self) -> None:
    with tempfile.TemporaryDirectory(prefix="jarvis-full-digit-resume-") as temp_dir:
      run_dir = Path(temp_dir) / "runs" / "balanced-fold1"
      checkpoint_path = run_dir / "weights" / "last.pt"
      checkpoint = {
        "epoch": 59,
        "optimizer": {"state": {}},
        "train_args": {
          "epochs": 120,
          "imgsz": 1280,
          "batch": 4,
          "seed": 42,
        },
      }
      args = type("Args", (), {
        "epochs": 120,
        "imgsz": 1280,
        "batch": 4,
        "seed": 42,
      })()

      completed_epochs = validate_resume_checkpoint(
        checkpoint_path,
        checkpoint,
        args,
        run_dir,
      )

      self.assertEqual(completed_epochs, 60)

  def test_rejects_resume_without_optimizer_state(self) -> None:
    with tempfile.TemporaryDirectory(prefix="jarvis-full-digit-resume-") as temp_dir:
      run_dir = Path(temp_dir) / "runs" / "balanced-fold1"
      checkpoint_path = run_dir / "weights" / "last.pt"
      checkpoint = {
        "epoch": 59,
        "optimizer": None,
        "train_args": {
          "epochs": 120,
          "imgsz": 1280,
          "batch": 4,
          "seed": 42,
        },
      }
      args = type("Args", (), {
        "epochs": 120,
        "imgsz": 1280,
        "batch": 4,
        "seed": 42,
      })()

      with self.assertRaisesRegex(ValueError, "no optimizer state"):
        validate_resume_checkpoint(
          checkpoint_path,
          checkpoint,
          args,
          run_dir,
        )

  def test_materializes_fold_without_test_leakage(self) -> None:
    with tempfile.TemporaryDirectory(prefix="jarvis-full-digit-train-") as temp_dir:
      root = Path(temp_dir)
      image_root = root / "source_images"
      label_root = root / "source_labels"
      rows = []
      folds = {}
      samples = [
        ("meter_a.JPEG", "1234", "train", 0),
        ("meter_b.JPEG", "2345", "train", 1),
        ("meter_c.JPEG", "3456", "train", 2),
        ("meter_d.JPEG", "4567", "train", 3),
        ("meter_e.JPEG", "5678", "train", 4),
        ("meter_test.JPEG", "6789", "test", None),
      ]
      for filename, reading, split, fold in samples:
        rows.extend(annotation_rows(filename, reading, split))
        write_sample(image_root, label_root, filename, reading, split)
        if fold is not None:
          folds[filename] = fold

      yaml_path, summary = materialize_fold_dataset(
        root / "materialized",
        rows,
        folds,
        image_root,
        label_root,
        selected_fold=2,
      )
      self.assertTrue(yaml_path.exists())
      self.assertEqual(summary["split_images"], {"train": 4, "val": 1, "test": 1})
      self.assertTrue(
        (root / "materialized" / "images" / "val" / "meter_c.JPEG").is_symlink()
      )
      self.assertTrue(
        (root / "materialized" / "images" / "test" / "meter_test.JPEG").is_symlink()
      )
      self.assertFalse(
        (root / "materialized" / "images" / "train" / "meter_test.JPEG").exists()
      )

  def test_rejects_pending_annotations(self) -> None:
    rows = annotation_rows("meter_pending.JPEG", "1234", "train", "pending")
    with tempfile.TemporaryDirectory(prefix="jarvis-full-digit-pending-") as temp_dir:
      root = Path(temp_dir)
      with self.assertRaisesRegex(ValueError, "requires every annotation to be reviewed"):
        materialize_fold_dataset(
          root / "materialized",
          rows,
          {"meter_pending.JPEG": 0},
          root / "images",
          root / "labels",
          selected_fold=0,
        )

  def test_register_crops_are_generated_for_training_images_only(self) -> None:
    with tempfile.TemporaryDirectory(prefix="jarvis-full-digit-crops-") as temp_dir:
      root = Path(temp_dir)
      image_root = root / "source_images"
      label_root = root / "source_labels"
      rows = []
      folds = {}
      samples = [
        ("meter_a.JPEG", "1234", "train", 0),
        ("meter_b.JPEG", "2345", "train", 1),
        ("meter_c.JPEG", "3456", "train", 2),
        ("meter_d.JPEG", "4567", "train", 3),
        ("meter_e.JPEG", "5678", "train", 4),
        ("meter_test.JPEG", "6789", "test", None),
      ]
      for filename, reading, split, fold in samples:
        rows.extend(annotation_rows(filename, reading, split))
        write_sample(image_root, label_root, filename, reading, split)
        if fold is not None:
          folds[filename] = fold

      _, summary = materialize_fold_dataset(
        root / "materialized",
        rows,
        folds,
        image_root,
        label_root,
        selected_fold=2,
        train_register_crops=True,
        register_crop_context=0.25,
      )

      self.assertEqual(summary["split_images"], {"train": 8, "val": 1, "test": 1})
      self.assertEqual(
        summary["train_register_crops"],
        {
          "enabled": True,
          "context_ratio": 0.25,
          "generated_images": 4,
        },
      )
      crop_image = (
        root
        / "materialized"
        / "images"
        / "train"
        / "meter_a__register_crop.JPEG"
      )
      crop_label = (
        root
        / "materialized"
        / "labels"
        / "train"
        / "meter_a__register_crop.txt"
      )
      self.assertTrue(crop_image.exists())
      self.assertFalse(crop_image.is_symlink())
      self.assertEqual(len(crop_label.read_text(encoding="utf-8").splitlines()), 4)
      self.assertFalse(
        (
          root
          / "materialized"
          / "images"
          / "val"
          / "meter_c__register_crop.JPEG"
        ).exists()
      )
      self.assertFalse(
        (
          root
          / "materialized"
          / "images"
          / "test"
          / "meter_test__register_crop.JPEG"
        ).exists()
      )

  def test_balanced_digit_crops_use_training_fold_sources_only(self) -> None:
    with tempfile.TemporaryDirectory(prefix="jarvis-full-digit-balanced-") as temp_dir:
      root = Path(temp_dir)
      image_root = root / "source_images"
      label_root = root / "source_labels"
      rows = []
      folds = {}
      samples = [
        ("meter_val.JPEG", "0123", "train", 0),
        ("meter_a.JPEG", "4567", "train", 1),
        ("meter_b.JPEG", "8901", "train", 2),
        ("meter_c.JPEG", "2345", "train", 3),
        ("meter_d.JPEG", "6789", "train", 4),
        ("meter_test.JPEG", "6789", "test", None),
      ]
      for filename, reading, split, fold in samples:
        rows.extend(annotation_rows(filename, reading, split))
        write_sample(image_root, label_root, filename, reading, split)
        if fold is not None:
          folds[filename] = fold

      _, summary = materialize_fold_dataset(
        root / "materialized",
        rows,
        folds,
        image_root,
        label_root,
        selected_fold=0,
        train_balanced_digit_target=3,
      )

      balanced = summary["train_balanced_digit_crops"]
      self.assertEqual(balanced["generated_images"], 14)
      self.assertEqual(
        summary["split_class_counts"]["train"],
        {str(digit): 3 for digit in range(10)},
      )
      self.assertEqual(summary["split_images"], {"train": 18, "val": 1, "test": 1})
      generated_images = list(
        (root / "materialized" / "images" / "train").glob("*__digit_*.JPEG")
      )
      generated_labels = list(
        (root / "materialized" / "labels" / "train").glob("*__digit_*.txt")
      )
      self.assertEqual(len(generated_images), 14)
      self.assertEqual(len(generated_labels), 14)
      self.assertFalse(any("meter_val" in path.name for path in generated_images))
      self.assertFalse(any("meter_test" in path.name for path in generated_images))
      self.assertTrue(
        all(len(path.read_text(encoding="utf-8").splitlines()) == 1 for path in generated_labels)
      )
      self.assertFalse(
        list((root / "materialized" / "images" / "val").glob("*__digit_*.JPEG"))
      )
      self.assertFalse(
        list((root / "materialized" / "images" / "test").glob("*__digit_*.JPEG"))
      )


if __name__ == "__main__":
  unittest.main()
