"""Verify original CV membership before running a checkpoint evaluation.

Standard-library only: the UI runner uses the same verifier via this module's CLI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

CV_FOLD_COUNT = 5


def checkpoint_run_dir(checkpoint_path: Path) -> Path:
  parent = checkpoint_path.parent
  return parent.parent if parent.name == "weights" else parent


def validate_checkpoint_fold(
  checkpoint_path: Path,
  selected_fold: int | None,
  folds_path: Path,
) -> dict[str, object]:
  provenance_path = checkpoint_run_dir(checkpoint_path) / "dataset_provenance.json"
  try:
    provenance_bytes = provenance_path.read_bytes()
    provenance = json.loads(provenance_bytes)
  except FileNotFoundError as error:
    raise FileNotFoundError(
      f"Missing checkpoint training provenance: {provenance_path}. "
      "Evaluation requires the original dataset_provenance.json from the training run."
    ) from error
  except json.JSONDecodeError as error:
    raise ValueError(f"Invalid checkpoint training provenance JSON: {provenance_path}") from error

  recorded_fold = provenance.get("selected_fold") if isinstance(provenance, dict) else None
  if type(recorded_fold) is not int or recorded_fold not in range(CV_FOLD_COUNT):
    raise ValueError(
      f"Checkpoint training provenance must record an integer selected_fold "
      f"in 0..{CV_FOLD_COUNT - 1}: {provenance_path}"
    )
  if selected_fold is not None and (
    type(selected_fold) is not int or selected_fold not in range(CV_FOLD_COUNT)
  ):
    raise ValueError("Requested validation fold must be an integer in 0..4.")
  if selected_fold is not None and recorded_fold != selected_fold:
    raise ValueError(
      f"Checkpoint was trained with validation fold {recorded_fold}, "
      f"but --fold {selected_fold} was requested. "
      f"Use --fold {recorded_fold} to avoid evaluating a training fold."
    )
  recorded_hash = provenance.get("cv_folds_sha256")
  if (
    not isinstance(recorded_hash, str) or len(recorded_hash) != 64
    or any(character not in "0123456789abcdef" for character in recorded_hash)
  ):
    raise ValueError("Checkpoint provenance must record the original cv_folds_sha256.")
  folds_bytes = folds_path.read_bytes()
  if hashlib.sha256(folds_bytes).hexdigest() != recorded_hash:
    raise ValueError(
      "CV fold manifest does not match the checkpoint's original cv_folds_sha256. "
      "Use the original training manifest via --folds (UI: "
      "FULL_IMAGE_DIGIT_SHADOW_CV_FOLDS_PATH); do not reconstruct provenance "
      "from current assignments."
    )

  # Parse the verified bytes once; callers must use these assignments even if
  # a concurrent dataset rebuild replaces the manifest during inference.
  assignments: dict[str, int] = {}
  reader = csv.DictReader(io.StringIO(folds_bytes.decode("utf-8")))
  if not {"filename", "fold"}.issubset(reader.fieldnames or []):
    raise ValueError(f"Invalid CV fold manifest columns: {folds_path}")
  for row in reader:
    filename = row.get("filename")
    if not filename or filename in assignments:
      raise ValueError(f"Invalid duplicate CV fold row: {filename!r}")
    try:
      fold = int(row.get("fold", ""))
    except (ValueError, TypeError) as error:
      raise ValueError(f"Invalid fold for {filename}") from error
    if fold not in range(CV_FOLD_COUNT):
      raise ValueError(f"Fold for {filename} must be in 0..4")
    assignments[filename] = fold
  if recorded_fold not in assignments.values():
    raise ValueError(f"No images in checkpoint validation fold {recorded_fold}.")
  return {
    "path": str(provenance_path),
    "sha256": hashlib.sha256(provenance_bytes).hexdigest(),
    "selected_fold": recorded_fold,
    "cv_folds_path": str(folds_path),
    "cv_folds_sha256": recorded_hash,
    "fold_assignments": assignments,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--folds", type=Path, required=True)
  parser.add_argument("--fold", type=int, choices=range(CV_FOLD_COUNT))
  args = parser.parse_args()
  print(json.dumps(validate_checkpoint_fold(args.checkpoint, args.fold, args.folds)))


if __name__ == "__main__":
  main()
