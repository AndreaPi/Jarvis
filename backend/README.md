# ROI Backend

Python service for neural ROI detection (digit window), per-cell digit-classifier inference, whole-strip shadow-reader inference, and optional ROI-cropped full-image digit-detector shadow inference.

## 1) Install

```bash
cd backend
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

For CPU-only environments (for example Vercel), install:

```bash
uv pip install -r requirements-cpu.txt
```

## 2) Fine-tune on your dataset

Prepare a YOLO dataset and YAML file. A template is available at `data/roi_dataset.example.yaml`.

Expected labels: one class (`digit_window`) with normalized YOLO boxes.

`build_roi_dataset.py` now keeps a persistent split manifest at `data/roi_dataset/splits.json`.
Existing samples keep their assigned split on rebuild, and newly ingested images default to `train`
unless you explicitly edit `splits.json`.

```bash
cd backend
source .venv/bin/activate
python train_roi.py \
  --data data/roi_dataset.yaml \
  --base-model yolov8n.pt \
  --rotation-angles 90,180,270,360 \
  --heavy-augment
```

The training script enforces this augmentation policy by default (heavy online augmentation + 90/180/270/360 train rotations). You can bypass it only for explicit ablations with `--allow-no-augment-policy`.

```bash
python train_roi.py --data data/roi_dataset.yaml --base-model yolov8n.pt --allow-no-augment-policy --no-heavy-augment --rotation-angles 90,180
```

After training, best weights are copied to `backend/models/roi.pt`.
The API default is pinned to `backend/models/roi-rotaug-e30-640.pt`; use `ROI_MODEL_PATH` to explicitly test/use `roi.pt` or another checkpoint.
The default checkpoint was refreshed on June 9, 2026 after the retrained ROI detector improved browser OCR `MAE` from `388.00` to `106.83` with exact match unchanged at `11/31` and no-read unchanged at `1/31`.

Promoted checkpoints in `backend/models/*.pt` are Tier 1 artifacts. Track them with DVC and push them to your configured DVC remote:

```bash
cd ..
uv pip install --python backend/.venv/bin/python "dvc[s3]"
dvc add backend/models/*.pt
scripts/dvc-push-safe.sh backend/models/*.pt
```

If your remote is not configured yet, add one once. For Backblaze B2, use its S3-compatible endpoint:

```bash
source backend/.venv/bin/activate
dvc remote add -d b2 s3://<bucket-name>/jarvis-dvc
dvc remote modify b2 endpointurl https://s3.<region>.backblazeb2.com
dvc remote modify --local b2 access_key_id <key-id>
dvc remote modify --local b2 secret_access_key <application-key>
```

## Build a digit OCR dataset from ROI labels

Current recommended flow:

1. Extract ROI windows by split.
2. Canonicalize orientation and split each window into 4 equispaced sections.
3. Label each section from the 4-digit reading string.

```bash
cd backend
source .venv/bin/activate
python extract_digit_windows.py --clean
python split_digit_windows.py --clean
python label_digit_sections.py --clean
```

This creates:

- `data/digit_dataset/windows/{train,val,test}`
- `data/digit_dataset/windows_canonical/{train,val,test}`
- `data/digit_dataset/sections/{train,val,test}`
- `data/digit_dataset/sections_labeled/{train,val,test}/{0..9}`
- manifests under `data/digit_dataset/manifests`

`split_digit_windows.py` also reads optional per-image canonical strip corrections from
`data/digit_dataset/manifests/canonical_overrides.csv`. Use that manifest for small,
reviewed dataset-generation fixes such as trimming excess side padding, retaining a little
more top/bottom context from the source ROI window, or applying an extra deskew correction.
After changing canonical overrides, regenerate sections, labels, synthetic train sections,
and the strip QA page before retraining.

Keep `data/digit_dataset/manifests/**` in Git. The bulk image trees above are DVC-managed via:

- `data/digit_dataset/windows.dvc`
- `data/digit_dataset/windows_canonical.dvc`
- `data/digit_dataset/sections.dvc`
- `data/digit_dataset/sections_labeled.dvc`

Validate the current windows/canonical/sections dataset:

```bash
python validate_digit_dataset.py
```

`build_digit_dataset.py` is deprecated and writes the retired strips/cells dataset shape.

Generate a prioritized capture checklist for underrepresented digits:

```bash
python plan_digit_expansion.py --target-train-per-digit 12 --priority-digits 4,5,6,9
```

## Build the full-image digit-box dataset

The replacement detector dataset labels the four individual digit-wheel
apertures directly on each full meter image with classes `0` through `9`.
Bootstrap the boxes from the reviewed register ROI, verified reading, and
canonical orientation metadata:

```bash
cd backend
source .venv/bin/activate
python build_full_image_digit_dataset.py
```

This writes the canonical review manifest and derived YOLO labels under
`data/full_image_digit_dataset/`. It also creates a disposable Make Sense
review package under `../output/full-image-digit-review/`.

`data/full_image_digit_dataset/manifests/source_exclusions.csv` retains
legacy-stress sources for diagnostics while omitting them from active labels,
CV folds, review packages, training, and evaluation. The source photo,
trusted reading, bootstrap rows, and reviewed annotations remain preserved.

After reviewing and exporting YOLO annotations from Make Sense, import them
with:

```bash
python import_full_image_digit_annotations.py /path/to/makesense-export.zip
python build_full_image_digit_dataset.py
```

The importer rejects missing images by default, requires exactly four boxes
per image, restores reading order from orientation metadata, and verifies that
the imported classes equal the trusted reading. Reviewed coordinates are
preserved on later bootstrap runs. Do not use the dataset for training while
`manifests/summary.json` reports any `pending` annotations.

The builder also maintains `manifests/cv_folds.csv`: five persistent,
image-level folds over train sources. Validate a fold before training:

```bash
python train_full_image_digit_detector.py --validate-only --fold 0
```

Train one fold without promoting its checkpoint:

```bash
python train_full_image_digit_detector.py \
  --fold 0 \
  --base-model yolov8n.pt \
  --device cpu
```

If training is interrupted, resume from the run's unstripped `last.pt` while
repeating the same fold, recipe, and run name:

```bash
python train_full_image_digit_detector.py \
  --fold 0 \
  --device cpu \
  --name full-image-digit-detector-fold0 \
  --resume-from runs/full-image-digit-detector-fold0/weights/last.pt
```

The trainer rematerializes the temporary fold, rejects mismatched core
arguments or a stripped/completed checkpoint, and lets Ultralytics restore the
epoch, optimizer, scheduler, and early-stopping state. Repeat any crop-recipe
flags from the original command exactly.

Evaluate its best checkpoint on that same full-image validation fold:

```bash
python evaluate_full_image_digit_detector.py \
  --checkpoint runs/full-image-digit-detector-fold0/weights/best.pt \
  --fold 0
```

This writes a JSON artifact beside the run with reproducible detection metrics,
complete-reading exact match, no-read, readable digit accuracy, readable
`MAE`, and per-image predictions. It uses the selected CV validation fold only
and never reads the historical sanity holdout or a source listed in
`manifests/source_exclusions.csv`.

After all folds for one recipe have been evaluated, export the visual error
audit from the repository root:

```bash
npm run qa:full-image-digit-errors
```

The exporter discovers the balanced48 fold evaluation artifacts by default
and writes a timestamped report under
`output/full-image-digit-error-audit/`. It preserves the frozen full-image
predictions, runs the promoted ROI detector with production sanity and crop
expansion, and applies each image's out-of-fold digit checkpoint to that crop.
Each cascade crop is inferred separately, matching the deployed endpoint;
batching differently shaped crops can alter Ultralytics padding and produced
optimistic historical cascade results.
It also compares diagnostic ground-truth-derived register and single-aperture
crops. Its
`transition-review.csv` is a worksheet only: review the blank fields before
making any canonical annotation changes. Pass repeated `--evaluation` values
to the Python script when auditing another recipe.

To exercise one checkpoint through the real browser pipeline without changing
Jarvis's selected reading, run from the repository root:

```bash
npm run qa:full-image-digit-shadow
```

The command records the checkpoint path and SHA-256, compares the shadow with
the current OCR on the complete live test set, and separately reports the
checkpoint's leakage-safe validation-fold slice. The August 4, 2026 fold-4
run improved that seven-image slice from production `3/7` exact and `MAE
40.00` to shadow `4/7` and `17.67`, but added one no-read. It therefore remains
disabled and unpromoted.

For a bounded single-image runtime sensitivity check on fold 4:

```bash
npm run qa:full-image-digit-shadow-sensitivity
```

The August 4 run found that confidence `0.20` with the existing NMS IoU `0.70`
recovers `meter_20260423.JPEG` by retaining its final `7` at confidence
`0.217`. Fold 4 then measures `5/7` exact, `0` no-reads, and `MAE 15.14`.
However, the paired complete UI diagnostic also accepts a wrong `5348` for
`meter_20260724.JPEG` at nearly identical minimum confidence `0.218`, worsening
shadow `MAE` from `312.37` to `386.59`. Keep the runtime default at `0.25`;
threshold tuning alone cannot separate these cases.

After correcting the error audit to exact one-image runtime inference, the
28-image out-of-fold cascade is `12/28` exact with `7` no-reads and readable
`MAE 15.71`, versus the register-context oracle's `17/28`, `1`, and `213.22`.
Every production-expanded crop covers 100% of its reviewed register. The gap
therefore points to single-image detector padding/scale sensitivity and digit
classification, not insufficient ROI expansion.

The selected fold becomes validation, the other four folds become train, and
the historical one-image sanity holdout remains test-only. It is not a
statistically meaningful external test set or a promotion gate.

To compare a mixed-scale recipe while retaining full-image evaluation, add
`--train-register-crops`. It creates one register-context crop for each
training image only; validation and test inputs remain full images:

```bash
python train_full_image_digit_detector.py \
  --fold 0 \
  --base-model yolov8n.pt \
  --train-register-crops \
  --device cpu
```

The controlled rare-class balancing recipe additionally uses digit-centred
crops from training-fold images only:

```bash
python train_full_image_digit_detector.py \
  --fold 0 \
  --base-model yolov8n.pt \
  --train-register-crops \
  --train-balanced-digit-target 24 \
  --device cpu
```

This raises each rare digit to a minimum of 24 training-box exposures without
augmenting validation or test. Every added image is a real crop around one
reviewed aperture; validation and test remain full-image-only. The run
provenance reports generated counts and distinct source-image counts by class.

Repeat folds `0` through `4` only after a recipe passes the initial fold
comparison on both detection and complete-reading metrics. Cross-validation
checkpoints stay under `runs/`. Freeze the recipe, collect a new locked
external full-image test set, and evaluate it once before using
`--copy-to models/full_image_digit_detector.pt`. Promotion still requires
explicit approval.

The complete annotation policy and review checklist are in
[`../docs/full-image-digit-dataset.md`](../docs/full-image-digit-dataset.md).

## Train a dedicated digit classifier

This trains a small per-cell CNN on real labeled sections (`data/digit_dataset/sections_labeled`) and writes:
- weights: `backend/models/digit_classifier.pt`
- training summary: `backend/runs/digit-classifier/digit_classifier_summary.json`

```bash
cd backend
source .venv/bin/activate
python train_digit_classifier.py --device cpu
```

### Recover and fine-tune the promoted digit classifier safely

`backend/models/digit_classifier.pt` is the safety baseline for the primary OCR path. Do not overwrite it with a scratch retrain unless the challenger beats the restored checkpoint on the UI **Run test set**. The current clean canonical section dataset is small and imbalanced, and a clean/synthetic scratch retrain regressed badly on browser runtime crops.

Restore the promoted checkpoints before experiments:

```bash
cd ..
env DVC_SITE_CACHE_DIR=/tmp/dvc-site-cache \
  backend/.venv/bin/python -m dvc checkout --force \
  backend/models/digit_classifier.pt.dvc \
  backend/models/digit_strip_reader.pt.dvc
env DVC_SITE_CACHE_DIR=/tmp/dvc-site-cache \
  backend/.venv/bin/python -m dvc status \
  backend/models/digit_classifier.pt.dvc \
  backend/models/digit_strip_reader.pt.dvc
```

To reconstruct the runtime-failure crop dataset from the restored classifier and current ROI model:

```bash
cd backend
source .venv/bin/activate
python export_runtime_digit_failure_set.py
cd ..
npm run qa:runtime-failure-dataset
npm run qa:runtime-failure-dataset:selected
```

The runtime-failure dataset is generated under `backend/data/runtime_failure_dataset/` and is intentionally train-only. It is ignored by Git unless a future promoted recipe explicitly decides to DVC-track it. Use the selected-only QA view when reviewing the candidates the UI actually chose; the full QA view is mainly for comparing alternate candidates.

Use grouped CV before training a challenger:

```bash
npm run qa:digit-classifier-cv
```

The CV folds are grouped by source image filename to avoid sibling-cell leakage. By default, CV uses the train pool only; `meter_20260327.JPEG` remains a fixed hard test holdout. It compares the restored checkpoint, clean-cell fine-tuning, and clean + runtime-failure fine-tuning. This is evidence for candidate recipes, not a promotion gate by itself.

As of May 27, 2026, the digit dataset intentionally has no one-image validation split: `meter_20260323.JPEG` moved into train because a single validation image was too noisy to guide model selection. If `train_digit_classifier.py` sees no validation samples, checkpoint selection falls back to train loss; judge recipes with grouped CV and the UI **Run test set** before promotion.

Write challengers outside `backend/models/`:

```bash
cd backend
source .venv/bin/activate
python train_digit_classifier.py \
  --device cpu \
  --init-checkpoint models/digit_classifier.pt \
  --extra-train-root data/runtime_failure_dataset/sections_labeled \
  --synthetic-root data/digit_dataset/sections_synthetic \
  --synthetic-target-ratio 2.0 \
  --synthetic-selection-strategy balanced \
  --project runs \
  --name digit-classifier-finetune-runtime \
  --copy-to runs/digit-classifier-finetune-runtime/digit_classifier.pt
```

May 24, 2026 recovery result: grouped CV improved from restored baseline `51.8%` per-cell accuracy to clean-only `61.6%` and clean + curated-runtime-failure `66.1%`, but the final challenger failed the then-28-image UI promotion gate (`3/28` exact, `MAE 139.11`, `1` no-read) versus the restored checkpoint with the conservative geometry ranker (`10/28` exact, `MAE 61.22`, `1` no-read). The promoted checkpoint therefore remains `backend/models/digit_classifier.pt`. The active UI baseline is tracked in `src/ocr/AGENTS.md` and `docs/ocr-tuning-playbook.md`; May 2026 numbers are historical snapshots only.

July 23, 2026 recovery result: after expanding the digit manifests to `36` sources, clean + balanced-synthetic fine-tuning improved grouped source-image CV from `48.6%` for the restored checkpoint to `67.9%`, but failed the 36-image UI gate (`0/36` exact, `MAE 3063.31`, `1` no-read). This is further evidence that clean-section CV does not represent browser runtime crop behavior; keep the restored checkpoint promoted.

## Train the whole-strip shadow reader

This trains a fixed four-head CNN on canonical ROI windows (`data/digit_dataset/windows_canonical`) and writes:

- weights: `backend/models/digit_strip_reader.pt`
- training summary: `backend/runs/strip-digit-reader/strip_digit_reader_summary.json`

```bash
cd backend
source .venv/bin/activate
python train_strip_digit_reader.py --device cpu
```

The trainer letterboxes canonical windows to `520x160` so horizontal digit geometry is preserved. When validation has too few samples for reliable checkpoint selection, the default `--selection-split auto` falls back to train-set selection and leaves the UI test set as the promotion gate.

The July 23, 2026 fine-tune on the expanded `35`-source train split remained `0/1` exact on the fixed hard holdout and only `1/7` exact in focused runtime QA. It remains a rejected shadow challenger.

## Train the guarded `23xx` shadow reader

This trains a house-specific constrained CNN on canonical ROI windows. It uses a binary guard for whether the second digit is `3`, then predicts only the final two suffix digits. It writes:

- weights: `backend/models/digit_strip_reader_23xx.pt`
- training summary: `backend/runs/strip-digit-reader-23xx/strip_digit_reader_23xx_summary.json`

```bash
cd backend
source .venv/bin/activate
python train_strip_digit_reader_23xx.py --device cpu
```

House-specific constrained-reader assumption: for this local water meter, the fixed prefix `23` is an intentional shortcut based on the expectation that the meter will remain below `2400` cubic meters while this project is used in this home. Review the assumption at least yearly or whenever readings approach `2390`. The reader stays shadow-only and only accepts a forced `23xx` value when its second-digit-is-`3` guard reaches the configured threshold. The first checkpoint is diagnostic-only: cross-validation looked conservative (`0` guard false positives, `19` guard false negatives), but runtime QA still found accepted wrong predictions. Lowering the guard from `0.98` to `0.80` accepted more wrong values, so threshold tuning is not enough.

The July 23, 2026 retrain was rejected with `0` guard false positives, `33` false negatives, and no accepted CV or focused-runtime predictions at the `0.98` threshold. Keep the existing shadow checkpoint; do not infer that a lower threshold would be safer.

Optional: generate synthetic **train-only** sections from real train patches, then mix real + synthetic in training.
Val/test remain strictly real-only.

```bash
cd backend
source .venv/bin/activate
python generate_synthetic_digit_dataset.py --clean --direct-per-real 6 --compose-window-count 180
python train_digit_classifier.py \
  --device cpu \
  --synthetic-root data/digit_dataset/sections_synthetic \
  --synthetic-target-ratio 2.0 \
  --name digit-classifier-synth-v1
```

Keep `data/digit_dataset/sections_synthetic/manifests/**` in Git and track the synthetic image tree with `data/digit_dataset/sections_synthetic/train.dvc`.

After promoting a new ROI or digit checkpoint, run `dvc add` for the changed artifacts, push with `scripts/dvc-push-safe.sh`, then package Tier 1 artifacts and publish them as Release assets so the raw photos, labels, manifests, and promoted weights are recoverable off-machine.

## 3) Start the API

```bash
cd backend
source .venv/bin/activate
uvicorn app:app --host 127.0.0.1 --port 8001 --reload
```

In the Codex/DevTools environment, starting `uvicorn` inside the sandbox may leave the service unreachable from the browser even when shell `curl` works. If browser requests to `127.0.0.1:8001` fail with `ERR_CONNECTION_REFUSED` or `Failed to fetch`, restart the backend outside the sandbox with escalated permissions.

Readiness check:

```bash
curl -s http://127.0.0.1:8001/health
```

Run backend regression tests from the repo root:

```bash
npm run test:backend
```

This command auto-discovers `backend/test_*.py`. These are fast unit, component,
and confirmed-regression checks that mock model inference; training runs and
checkpoint/model-quality comparisons remain separate `qa:*` workflows.

## 4) Endpoints

- `GET /health`: model readiness (`ready`, `roi_ready`, `digit_ready`, `strip_digit_ready`, `strip_digit_23xx_ready`, `full_image_digit_shadow_ready`) + effective model/device config and `max_upload_bytes`. Missing optional full-image weights do not make the canonical ROI service unready.
- `POST /roi/detect`: multipart upload (`image`) and returns normalized bbox + confidence.
- `POST /digit/predict`: multipart upload (`image`) and returns the predicted digit + confidence.
- `POST /digit/predict-cells`: multipart upload (`images`, repeated field) for batch cell decoding.
- `POST /digit/predict-strip`: multipart upload (`image`) for direct fixed-length 4-digit strip decoding.
- `POST /digit/predict-strip-23xx`: multipart upload (`image`) for guarded house-specific `23xx` suffix decoding.
- `POST /digit/predict-full-image-shadow`: multipart upload (`image`) that reuses the promoted ROI detector, expands the register crop, detects four digit boxes, and returns ordered candidates for each right-angle rotation. It never chooses or promotes a reading.

All image fields default to a 20 MiB per-file limit and return HTTP `413` when exceeded. Override the limit with `MAX_UPLOAD_BYTES`.

Frontend integration defaults:
- ROI detection path is `http://127.0.0.1:8001/roi/detect` and is required for OCR.
- Digit classifier path is `http://127.0.0.1:8001/digit/predict-cells` and is only used when `OCR_CONFIG.digitClassifier.enabled=true`.
- Strip reader path is `http://127.0.0.1:8001/digit/predict-strip`; frontend OCR runs it shadow-only via `OCR_CONFIG.digitStripReader.shadowOnly=true` and logs results without changing final selection.
- Constrained `23xx` strip reader path is `http://127.0.0.1:8001/digit/predict-strip-23xx`; frontend OCR runs it shadow-only via `OCR_CONFIG.digitStripReader23xx.shadowOnly=true` and logs accepted/abstained diagnostics without changing final selection.
- Full-image digit shadow path is `http://127.0.0.1:8001/digit/predict-full-image-shadow`; it is frontend-disabled by default. An explicit run uses the primary OCR angle only to choose which returned rotation to log, and it cannot change final selection.
- Frontend ROI OCR prioritizes `90/270` edge candidates, but the primary pass also evaluates top base-strip rotations when present. A narrow `scan-roi` / base fallback rerun is only used when base candidates were not already evaluated and the edge evidence remains weak. Final confidence gates can still reject weak edge-only reads.

## Environment Variables

- `MAX_UPLOAD_BYTES`: maximum bytes read from each uploaded image (default: `20971520`).
- `ROI_MODEL_PATH`: path to `.pt` weights (default: `backend/models/roi-rotaug-e30-640.pt`)
- `ROI_DEFAULT_CONFIDENCE`: detection confidence threshold (default: `0.05`)
- `ROI_DEFAULT_IOU`: NMS IoU threshold (default: `0.5`)
- `ROI_DEFAULT_IMGSZ`: inference size (default: `960`)
- `ROI_CLASS_INDEX`: optional class id filter
- `ROI_DEVICE`: inference device (default: `cpu`).
  - Use `cpu` for CPU-only deploys (recommended on Vercel).
  - Use `auto` to let Ultralytics choose.
  - Use `0` or `cuda:0` to force GPU.
- `DIGIT_MODEL_PATH`: path to digit classifier checkpoint (default: `backend/models/digit_classifier.pt`)
- `DIGIT_DEVICE`: inference device for digit classifier (default follows `ROI_DEVICE`)
- `DIGIT_MIN_CONFIDENCE`: minimum accepted confidence for digit predictions (default: `0.0`)
- `DIGIT_TOP_K`: number of top classes returned by digit endpoints (default: `3`)
- `STRIP_DIGIT_MODEL_PATH`: path to strip reader checkpoint (default: `backend/models/digit_strip_reader.pt`)
- `STRIP_DIGIT_DEVICE`: inference device for strip reader (default follows `DIGIT_DEVICE`)
- `STRIP_DIGIT_MIN_CONFIDENCE`: minimum accepted average confidence for strip predictions (default: `0.0`)
- `STRIP_DIGIT_TOP_K`: number of top classes returned per strip position (default: `3`)
- `STRIP_DIGIT_23XX_MODEL_PATH`: path to constrained strip reader checkpoint (default: `backend/models/digit_strip_reader_23xx.pt`)
- `STRIP_DIGIT_23XX_DEVICE`: inference device for constrained strip reader (default follows `STRIP_DIGIT_DEVICE`)
- `STRIP_DIGIT_23XX_GUARD_THRESHOLD`: minimum second-digit-is-`3` guard confidence before accepting a forced `23xx` value (default: `0.98`)
- `STRIP_DIGIT_23XX_TOP_K`: number of top classes returned per constrained strip position (default: `3`)
- `FULL_IMAGE_DIGIT_SHADOW_MODEL_PATH`: optional full-image detector checkpoint (default: `backend/models/full_image_digit_detector.pt`; the file is intentionally absent until promotion)
- `FULL_IMAGE_DIGIT_SHADOW_DEVICE`: inference device (default follows `DIGIT_DEVICE`)
- `FULL_IMAGE_DIGIT_SHADOW_CONFIDENCE`: digit-box confidence threshold (default: `0.25`)
- `FULL_IMAGE_DIGIT_SHADOW_IOU`: digit-box NMS IoU threshold (default: `0.7`)
- `FULL_IMAGE_DIGIT_SHADOW_IMGSZ`: digit-detector inference size (default: `1280`)
- `FULL_IMAGE_DIGIT_SHADOW_MAX_DETECTIONS`: maximum digit detections (default: `300`)
- `FULL_IMAGE_DIGIT_SHADOW_ROI_EXPAND_X`: horizontal ROI expansion ratio (default: `0.26`)
- `FULL_IMAGE_DIGIT_SHADOW_ROI_EXPAND_Y`: vertical ROI expansion ratio (default: `0.16`)

## CPU-only vs GPU

- CPU-only install (recommended for Vercel/serverless):
  - `uv pip install -r requirements-cpu.txt`
- GPU-capable install:
  - install a CUDA-enabled PyTorch build, then `uv pip install -r requirements.txt`.

Training can also be pinned with `--device`:

```bash
python train_roi.py --data data/roi_dataset.yaml --base-model yolov8n.pt --device cpu --rotation-angles 90,180,270,360 --heavy-augment
```
