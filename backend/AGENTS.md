# Backend Guidelines

## Scope
- This file covers `backend/` runtime, API, dataset, and training guidance.

## Setup and Run
- Create the backend virtualenv with `cd backend && uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt`.
- For any Python task that depends on computer-vision packages or image tooling (for example `ultralytics`, `opencv`, or `Pillow`), use `backend/.venv` rather than the system Python.
- Run the API with `cd backend && source .venv/bin/activate && uvicorn app:app --host 127.0.0.1 --port 8001 --reload`.
- Run backend regression tests from the repo root with `npm run test:backend`; it auto-discovers `backend/test_*.py`.
- In this Codex environment, if the service must be consumed by the DevTools browser, starting `uvicorn` inside the sandbox may not be reachable from the page. Prefer restarting it with escalated permissions when browser fetches to `127.0.0.1:8001` fail.

## API and Runtime Expectations
- Frontend/backend default ports are `127.0.0.1:8000` and `127.0.0.1:8001`.
- Health check: `GET /health` should report `ready: true`, `roi_ready: true`, `digit_ready: true`, `strip_digit_ready: true`, `strip_digit_23xx_ready: true`, `max_upload_bytes`, and the expected model paths when all checkpoints are present.
- Image endpoints read at most `MAX_UPLOAD_BYTES + 1` bytes per uploaded file and return HTTP `413` above the configured limit; the default limit is 20 MiB.
- Default ROI endpoint: `http://127.0.0.1:8001/roi/detect`
- Default digit endpoint: `http://127.0.0.1:8001/digit/predict-cells`
- Default strip-reader shadow endpoint: `http://127.0.0.1:8001/digit/predict-strip`
- Default constrained `23xx` strip-reader shadow endpoint: `http://127.0.0.1:8001/digit/predict-strip-23xx`
- Optional full-image digit-detector shadow endpoint: `http://127.0.0.1:8001/digit/predict-full-image-shadow`
- Backend default ROI model is pinned to `backend/models/roi-rotaug-e30-640.pt` (override with `ROI_MODEL_PATH`). This checkpoint was refreshed on June 9, 2026 after the retrained ROI detector improved end-to-end OCR `MAE` without exact-match or no-read regressions.
- Digit classifier path is `backend/models/digit_classifier.pt`.
- Strip digit reader path is `backend/models/digit_strip_reader.pt`.
- Constrained `23xx` strip digit reader path is `backend/models/digit_strip_reader_23xx.pt`.

## Dataset and Training Commands
- `cd backend && source .venv/bin/activate && python train_roi.py --data data/roi_dataset.yaml --base-model yolov8n.pt --rotation-angles 90,180,270,360 --heavy-augment`: Fine-tune the ROI detector.
- `cd backend && source .venv/bin/activate && python extract_digit_windows.py --clean`: Rebuild split-wise digit windows from ROI labels.
- `cd backend && source .venv/bin/activate && python split_digit_windows.py --clean`: Canonicalize and split digit windows into 4 equispaced sections.
- `cd backend && source .venv/bin/activate && python label_digit_sections.py --clean`: Build labeled section datasets.
- `cd backend && source .venv/bin/activate && python validate_digit_dataset.py`: Validate the current windows/canonical/sections digit dataset.
- `cd backend && source .venv/bin/activate && python build_full_image_digit_dataset.py`: Bootstrap full-image digit-wheel boxes, regenerate derived YOLO labels, and build the Make Sense review package.
- `cd backend && source .venv/bin/activate && python import_full_image_digit_annotations.py /path/to/makesense-export.zip`: Guardedly import reviewed Make Sense boxes into the canonical full-image annotation manifest.
- `cd backend && source .venv/bin/activate && python train_full_image_digit_detector.py --validate-only --fold 0`: Validate and materialize one reviewed full-image digit-detector CV fold without training.
- `cd backend && source .venv/bin/activate && python train_full_image_digit_detector.py --fold 0 --device cpu`: Train one full-image digit-detector CV fold without promoting its checkpoint. Add `--train-register-crops --train-balanced-digit-target 24` for the controlled train-only mixed-scale and rare-class-balancing recipe.
- Resume an interrupted full-image run with the same fold, crop-recipe flags, and `--name`, plus `--resume-from runs/<run>/weights/last.pt`; the trainer rematerializes the temporary fold and preserves optimizer, scheduler, and early-stopping state.
- `cd backend && source .venv/bin/activate && python evaluate_full_image_digit_detector.py --checkpoint runs/<run>/weights/best.pt --fold 0`: Reproduce detection metrics and evaluate complete four-digit readings on the selected full-image validation fold only.
- `npm run qa:full-image-digit-errors`: Export a timestamped visual audit from the frozen balanced48 out-of-fold evaluation artifacts, including the production ROI-to-register-crop cascade, register-crop and single-aperture oracles, and a non-mutating transition-review worksheet.
- `npm run qa:full-image-digit-shadow`: Start disposable frontend/backend services, run one explicitly configured full-image checkpoint as a non-selecting shadow on the complete UI test set, and report both the development-wide comparison and its leakage-safe validation-fold slice.
- `npm run qa:full-image-digit-shadow-sensitivity`: Run the exact single-image ROI cascade on the selected CV fold across the bounded confidence/NMS grid and render the target failure. Treat the fold as tuning data after this command.
- `cd backend && source .venv/bin/activate && python generate_synthetic_digit_dataset.py --clean --direct-per-real 6 --compose-window-count 180`: Generate synthetic train-only digit sections.
- `cd backend && source .venv/bin/activate && python plan_digit_expansion.py --target-train-per-digit 12 --priority-digits 4,5,6,9`: Refresh the targeted capture checklist.
- `cd backend && source .venv/bin/activate && python train_digit_classifier.py --device cpu`: Train the real-only digit classifier.
- `cd backend && source .venv/bin/activate && python train_digit_classifier.py --device cpu --synthetic-root data/digit_dataset/sections_synthetic --synthetic-target-ratio 2.0`: Train on mixed real + synthetic data while keeping val/test real-only.
- `cd backend && source .venv/bin/activate && python train_strip_digit_reader.py --device cpu`: Train the fixed four-head whole-strip reader from `data/digit_dataset/windows_canonical`.
- `cd backend && source .venv/bin/activate && python train_strip_digit_reader_23xx.py --device cpu`: Train the guarded house-specific `23xx` shadow reader from `data/digit_dataset/windows_canonical`.
- `npm run qa:digit-classifier-cv`: Run grouped source-image CV on the train pool for digit-classifier recipe checks. This replaces the old one-image validation split as the default experiment evaluator.

## House-Specific Strip Reader Shortcut
- The constrained `23xx` strip reader is implemented as a shadow-only experiment. It uses a binary guard for whether the second digit is `3` plus two suffix digit heads; it only emits `23xx` when the guard confidence reaches `0.98`.
- This is valid only while the local water meter is expected to remain in the `2300`-`2399` range; review the assumption at least yearly and whenever readings approach `2390`.
- Keep benchmark comparison against the unconstrained four-head reader and current primary OCR path. The first checkpoint is diagnostic-only: cross-validation looked conservative (`0` guard false positives, `19` guard false negatives), but runtime QA still found accepted wrong predictions. Lowering the guard from `0.98` to `0.80` accepted more wrong values, so threshold tuning is not enough.

## Backend Policy
- Keep `backend/test_*.py` fast and deterministic: use mocks for model inference and temporary directories for generated data. Cover durable API/dataset/training invariants and confirmed regressions, but leave real training, checkpoint comparison, and visual inspection to the `qa:*` workflows.
- Prefer one table-driven test for equivalent helper inputs instead of one method per value. Do not add tests solely for report formatting or trivial pass-through code unless they reproduce a user-visible regression.
- `train_roi.py` should keep heavy augmentation and rotation expansion `90,180,270,360`; weaker runs require explicit override with `--allow-no-augment-policy`.
- Treat promoted checkpoints under `backend/models/*.pt` as must-retain artifacts and keep DVC state up to date when models or datasets change.
- Keep host/CORS scoped to localhost unless there is an explicit deployment task.
- Digit-model experiments use grouped CV by source image. The digit dataset currently has no validation split; `meter_20260323.JPEG` is train and `meter_20260327.JPEG` remains a fixed hard test holdout. UI **Run test set** remains the promotion gate.
- Full-image digit detection uses `data/full_image_digit_dataset/manifests/annotations.csv` as its canonical human-review layer and `cv_folds.csv` as its persistent five-fold assignment over train sources. Bootstrap output must never overwrite existing reviewed rows, training must wait until no annotation is `pending`, and the one-image historical sanity holdout must not participate in model selection or be presented as a generalization estimate. Evaluate every checkpoint with `evaluate_full_image_digit_detector.py`; mAP improvements are insufficient when complete-reading exact match and no-read do not improve. Freeze the selected recipe before evaluating it once on a newly collected locked external full-image test set.
- `data/full_image_digit_dataset/manifests/source_exclusions.csv` is the active-scope boundary for the full-image detector. Preserve excluded source photos, readings, bootstrap rows, and reviewed annotations as `legacy_stress`, while omitting them from active labels, folds, review packages, training, and evaluation.
- Balanced full-image digit crops must be generated only from source images assigned to training for the selected fold. Keep validation and test as unchanged full images, and record per-class generated-crop and source-image counts in run provenance.
- Full-image error audits must preserve the recorded out-of-fold predictions and hashes. The ROI cascade must use the promoted ROI checkpoint and production sanity/expansion geometry without digit-box leakage, and must infer each crop separately to mirror the one-upload runtime endpoint; batching can change Ultralytics padding and hide no-reads. Oracle crops are diagnostic only. Review the generated transition worksheet before any canonical `transition_state` edits.
- Keep the full-image runtime endpoint optional: a missing `backend/models/full_image_digit_detector.pt` must not make the canonical ROI API unready. The frontend shadow stays disabled by default and may only log candidates. A single fold checkpoint's complete UI result includes training overlap; promotion decisions must use its matching validation-fold slice or a separately locked external test set.
- July 23, 2026 retraining on the expanded 36-source corpus promoted no models. The ROI challenger regressed exact match/no-read, the CV-winning digit-classifier challenger failed the UI gate, and both strip-reader challengers remained non-competitive. Keep all four `backend/models/*.pt` defaults unchanged.

## Digit Dataset Expansion Loop (`4/5/6/9`)
1. Refresh capture planning with `python plan_digit_expansion.py --target-train-per-digit 12 --priority-digits 4,5,6,9`.
2. Add labeled captures with QA previews.
3. Confirm `data/digit_dataset/manifests/{windows.csv,canonical_windows.csv,sections.csv,section_labels.csv}` are regenerated and consistent with current splits.
4. Retrain the per-cell classifier only after class coverage improves.
5. Retrain the strip reader after canonical windows change; keep it shadow-only until UI benchmark exact-match and `MAE` beat the current primary path.
6. Promote new checkpoints only when benchmarked OCR `MAE` improves without exact-match or no-read regressions.

## Canonical Strip QA Overrides
- `split_digit_windows.py` reads optional reviewed corrections from `data/digit_dataset/manifests/canonical_overrides.csv`.
- Use canonical overrides only for small, visually accepted dataset-generation fixes such as side trimming, retaining more top/bottom source-window context, or extra deskew on specific filenames.
- After changing canonical overrides, regenerate sections, labels, synthetic train sections, validate the dataset, and run `npm run qa:strip-dataset` for visual acceptance before any retraining.
