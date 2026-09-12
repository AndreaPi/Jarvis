from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import backend.train_full_image_digit_detector as training
from backend.train_full_image_digit_detector import (
  materialize_fold_dataset,
  materialized_dataset_sha256,
  validate_resume_checkpoint,
  validate_resume_provenance,
)


def resume_provenance() -> dict:
  return {
    "selected_fold": 1,
    "annotations_sha256": "annotations",
    "cv_folds_sha256": "folds",
    "source_exclusions_sha256": None,
    "materialized_dataset_sha256": "artifacts",
    "train_register_crops": {"enabled": True, "context_ratio": 0.75},
    "train_balanced_digit_crops": {"target_class_count": 24},
    "augmentation": dict(training.TRAIN_AUGMENT_KWARGS),
    "ultralytics_version": "test-version",
  }


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
  def test_materialization_checks_class_box_pairs_against_reviewed_annotations(self) -> None:
    for case in ("rounded-reordered", "stale-geometry", "swapped-classes", "nan"):
      with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = annotation_rows("meter.JPEG", "1234", "train")
        rows += annotation_rows("holdout.JPEG", "1234", "test")
        rows += annotation_rows("train.JPEG", "1234", "train")
        for filename, split in (("meter.JPEG", "train"), ("holdout.JPEG", "test"), ("train.JPEG", "train")):
          write_sample(root / "images", root / "labels", filename, "1234", split)
        path = root / "labels/train/meter.txt"
        lines = path.read_text().splitlines()
        if case == "rounded-reordered":
          rows[0]["x_center"] = "0.20000049"
          lines.reverse()
        elif case == "stale-geometry":
          lines[0] = lines[0].replace("0.500000", "0.700000")
        elif case == "swapped-classes":
          lines[0] = "2" + lines[0][1:]
          lines[1] = "1" + lines[1][1:]
        else:
          lines[0] = lines[0].replace("0.500000", "nan")
        path.write_text("\n".join(lines) + "\n")

        def materialize():
          return materialize_fold_dataset(
            root / "out", rows, {"meter.JPEG": 0, "train.JPEG": 1}, root / "images", root / "labels", 0,
          )

        if case == "rounded-reordered":
          materialize()
        else:
          with self.assertRaises(ValueError):
            materialize()

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
      provenance = resume_provenance()
      run_dir.mkdir(parents=True)
      (run_dir / "dataset_provenance.json").write_text(json.dumps(provenance))

      completed_epochs = validate_resume_checkpoint(
        checkpoint_path,
        checkpoint,
        args,
        run_dir,
        provenance,
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
          resume_provenance(),
        )

  def test_resume_rejects_changed_or_missing_provenance(self) -> None:
    original = resume_provenance()
    changed_values = {
      "selected_fold": 0,
      "annotations_sha256": "changed annotations",
      "cv_folds_sha256": "changed folds",
      "source_exclusions_sha256": "new exclusion",
      "materialized_dataset_sha256": "changed pixels or labels",
      "train_register_crops": {"enabled": False, "context_ratio": 0.75},
      "train_balanced_digit_crops": {"target_class_count": 48},
      "augmentation": {"degrees": 90},
      "ultralytics_version": "changed-version",
    }
    with tempfile.TemporaryDirectory() as directory:
      run_dir = Path(directory)
      path = run_dir / "dataset_provenance.json"
      path.write_text(json.dumps(original))
      original_bytes = path.read_bytes()
      for key, value in changed_values.items():
        with self.subTest(changed=key):
          requested = copy.deepcopy(original)
          requested[key] = value
          with self.assertRaisesRegex(ValueError, key):
            validate_resume_provenance(run_dir, requested)
          self.assertEqual(path.read_bytes(), original_bytes)
      for contents in (None, "{", "[]", "{}", json.dumps({**original, "selected_fold": True})):
        with self.subTest(contents=contents):
          if contents is None:
            path.unlink()
          else:
            path.write_text(contents)
          with self.assertRaises(ValueError):
            validate_resume_provenance(run_dir, original)

  def test_dataset_fingerprint_detects_pixel_and_label_changes_across_temp_roots(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      for name in ("first", "second"):
        write_sample(root / name / "images", root / name / "labels", "a.JPEG", "1234", "train")
      original = materialized_dataset_sha256(root / "first")
      self.assertEqual(original, materialized_dataset_sha256(root / "second"))
      (root / "first/labels/train.cache").write_bytes(b"Ultralytics-generated cache")
      self.assertEqual(original, materialized_dataset_sha256(root / "first"))
      label = root / "first/labels/train/a.txt"
      label.write_text(label.read_text().replace("0.500000", "0.600000"))
      self.assertNotEqual(original, materialized_dataset_sha256(root / "first"))
      Image.new("RGB", (100, 100), "black").save(root / "second/images/train/a.JPEG")
      self.assertNotEqual(original, materialized_dataset_sha256(root / "second"))

  def test_interrupted_training_keeps_original_provenance_and_blocks_changed_resume(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      annotations, folds = root / "annotations.csv", root / "folds.csv"
      annotations.write_text("unchanged annotations")
      folds.write_text("unchanged folds")
      actual_run = root / "runs/suffixed-run2"
      actual_run.mkdir(parents=True)
      checkpoint = actual_run / "weights/last.pt"
      checkpoint.parent.mkdir()
      checkpoint.write_bytes(b"mock checkpoint")
      args = SimpleNamespace(
        annotations=str(annotations), folds=str(folds), source_exclusions=str(root / "none.csv"),
        source_images=str(root / "images"), labels=str(root / "labels"), project=str(root / "runs"),
        copy_to="", resume_from="", name="suffixed-run", pretrained_model="", base_model="yolov8n.pt",
        fold=1, train_register_crops=True, register_crop_context=0.75, train_balanced_digit_target=24,
        epochs=120, imgsz=1280, batch=4, patience=25, seed=42, device="cpu", workers=0,
        validate_only=False,
      )

      def materialize(destination, *_args, **_kwargs):
        destination.mkdir(parents=True)
        (destination / "dataset.yaml").write_text("test")
        return destination / "dataset.yaml", {
          key: value for key, value in resume_provenance().items()
          if key in {"selected_fold", "train_register_crops", "train_balanced_digit_crops"}
        }

      class InterruptedModel:
        ckpt = {"epoch": 59, "optimizer": {}, "train_args": {
          "epochs": 120, "imgsz": 1280, "batch": 4, "seed": 42,
        }}

        def add_callback(self, event, callback):
          self.event, self.callback = event, callback

        def train(self, **_kwargs):
          self.callback(SimpleNamespace(
            save_dir=actual_run, args=SimpleNamespace(data=_kwargs["data"]),
          ))
          raise KeyboardInterrupt("simulated interruption before first epoch")

      model = InterruptedModel()
      with (
        patch.object(training, "parse_args", return_value=args),
        patch.object(training, "read_csv_rows", return_value=[]),
        patch.object(training, "read_fold_assignments", return_value={}),
        patch.object(training, "materialize_fold_dataset", side_effect=materialize),
        patch("ultralytics.YOLO", return_value=model) as load_model,
        patch("builtins.print"),
      ):
        with self.assertRaises(KeyboardInterrupt):
          training.main()
        self.assertEqual(model.event, "on_pretrain_routine_start")
        provenance_path = actual_run / "dataset_provenance.json"
        original_bytes = provenance_path.read_bytes()
        self.assertEqual(json.loads(original_bytes)["selected_fold"], 1)
        self.assertFalse((root / "runs/suffixed-run/dataset_provenance.json").exists())

        args.resume_from, args.name = str(checkpoint), actual_run.name
        with self.assertRaises(KeyboardInterrupt):
          training.main()
        self.assertEqual(provenance_path.read_bytes(), original_bytes)

        actual_data = root / "surviving-dataset"
        write_sample(actual_data / "images", actual_data / "labels", "stale.JPEG", "1234", "train")
        with self.assertRaisesRegex(ValueError, "Ultralytics selected a resume dataset"):
          model.callback(SimpleNamespace(
            save_dir=actual_run, args=SimpleNamespace(data=str(actual_data / "dataset.yaml")),
          ))
        self.assertEqual(provenance_path.read_bytes(), original_bytes)

        folds.write_text("changed fold assignments")
        load_model.reset_mock()
        with self.assertRaisesRegex(ValueError, "cv_folds_sha256"):
          training.main()
        load_model.assert_not_called()
        self.assertEqual(provenance_path.read_bytes(), original_bytes)

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
