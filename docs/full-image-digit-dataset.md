# Full-Image Digit-Box Dataset

This dataset supports the replacement OCR approach: detect the four decimal
digit wheels directly in the original meter photo, with one object class for
each digit `0` through `9`.

It is deliberately separate from the existing register-ROI, per-cell
classifier, and whole-strip datasets. Building or reviewing it does not change
the active Jarvis OCR runtime.

## Annotation policy

- Draw exactly four boxes per full image.
- Each box covers one complete digit-wheel aperture, not only the dark glyph.
- Assign the class from the verified four-digit value in
  `assets/meter_readings.csv`.
- Preserve the fixed source split. `meter_20260327.JPEG` remains an untouched
  historical sanity holdout; other current sources remain in train. One image
  is not a statistically meaningful external test set and must not be used to
  claim generalization performance.
- A wheel between two digits is still one object. Keep the class from the
  verified reading and record its transition state separately in the canonical
  manifest.
- Do not train from annotations whose `review_status` is `pending`.
- `manifests/source_exclusions.csv` is the explicit active-scope boundary.
  Excluded sources retain their photo, reading, bootstrap rows, and reviewed
  canonical annotations, but are omitted from derived labels, CV folds, review
  packages, training, and evaluation.
- `meter_20201111.JPEG` is retained as `legacy_stress` but excluded from active
  full-image digit detection because severe defocus and its obsolete capture
  domain do not represent the intended input quality.

## Bootstrap the dataset

From the repository root:

```bash
cd backend
source .venv/bin/activate
python build_full_image_digit_dataset.py
```

The builder combines three existing reviewed facts:

1. the full-image register ROI;
2. the verified four-digit reading;
3. the canonical reading orientation.

It divides the register into four aperture boxes and writes:

- `data/full_image_digit_dataset/manifests/bootstrap_boxes.csv`: reproducible
  generated suggestions;
- `data/full_image_digit_dataset/manifests/annotations.csv`: canonical review
  layer, including source, review status, transition state, and notes;
- `data/full_image_digit_dataset/manifests/cv_folds.csv`: persistent
  image-level cross-validation assignments for train sources;
- `data/full_image_digit_dataset/manifests/source_exclusions.csv`: retained
  legacy-stress sources omitted from the active detector dataset;
- `data/full_image_digit_dataset/labels/{train,val,test}/*.txt`: derived YOLO
  labels;
- `data/full_image_digit_dataset/manifests/summary.json`: split, class, and
  review counts;
- `../output/full-image-digit-review/`: disposable images, previews, contact
  sheet, and Make Sense import ZIP.

Existing canonical annotation rows are preserved when the builder runs again.
This keeps human corrections separate from reproducible bootstrap geometry.
The summary reports active counts separately from the retained annotation
archive.

## Review in Make Sense

1. Open [Make Sense](https://www.makesense.ai/).
2. Upload every image from
   `output/full-image-digit-review/images/`.
3. Choose **Object Detection**.
4. At label setup, use **Load labels from file** with `labels.txt`, or create
   the labels `0` through `9` manually in exactly that order.
5. Start the project and choose **Actions > Import Annotations > YOLO**.
6. Unzip `output/full-image-digit-review/makesense-yolo-labels.zip`, then select
   all `meter_*.txt` files from the unzipped folder. Do not include
   `labels.txt` in the annotation import; it is only the class-list input from
   step 4.
7. For each image, confirm that there are four boxes, each box encloses a full
   aperture, and the four classes form the verified reading in reading order.
8. Correct the boxes if needed, then export all annotations in YOLO format.
   Keep the original image filenames.

Use `digit-box-contact-sheet.jpg` for a fast overview and
`previews/crops/` for larger per-image checks. The full-resolution source
images, not the previews, are the authoritative annotation surface.

## Import reviewed annotations

From `backend/`:

```bash
python import_full_image_digit_annotations.py /path/to/makesense-export.zip
python build_full_image_digit_dataset.py
```

The importer:

- rejects unknown or missing active images by default;
- requires exactly four in-bounds boxes per image;
- sorts boxes into reading order using the stored orientation;
- rejects class sequences that differ from the verified meter reading;
- marks imported rows as `human-makesense` and `reviewed`;
- preserves transition-state and notes fields.

Use `--allow-partial` only when intentionally importing a subset review. After
import, rerun the builder to regenerate YOLO labels, summary counts, and QA
previews from the canonical manifest.

## Training gate

Before any training:

1. Confirm `manifests/summary.json` contains no `pending` review status.
2. Inspect the regenerated contact sheet.
3. Run `npm run test:backend`.
4. Keep the historical sanity holdout out of all model-selection and
   augmentation inputs.

Validate the materialized split for each recipe:

```bash
cd backend
source .venv/bin/activate
python train_full_image_digit_detector.py --validate-only --fold 0
```

The trainer creates a temporary YOLO dataset: the selected CV fold is
validation, the other four folds are train, and the historical sanity holdout
remains test-only. It records annotation and fold-manifest hashes, the
source-exclusion manifest and hash, the Ultralytics version, and recipe
settings in every completed run.

Train one fold without promotion:

```bash
python train_full_image_digit_detector.py \
  --fold 0 \
  --base-model yolov8n.pt \
  --device cpu
```

If the run is interrupted, resume its unstripped `last.pt` checkpoint while
repeating the same fold, recipe flags, and run name:

```bash
python train_full_image_digit_detector.py \
  --fold 0 \
  --device cpu \
  --name full-image-digit-detector-fold0 \
  --resume-from runs/full-image-digit-detector-fold0/weights/last.pt
```

The trainer rebuilds the deleted temporary dataset before Ultralytics restores
the completed epoch, optimizer, scheduler, and early-stopping state. It rejects
core argument mismatches, a checkpoint from another run name, and stripped or
already-completed checkpoints.

Evaluate the best checkpoint on the same fold:

```bash
python evaluate_full_image_digit_detector.py \
  --checkpoint runs/full-image-digit-detector-fold0/weights/best.pt \
  --fold 0
```

The evaluator defaults to CPU and reads only uncropped full images from the
selected CV validation fold. It reproduces precision, recall, `mAP50`, and
`mAP50-95`, then applies class-agnostic NMS at a fixed confidence threshold.
Exactly four detections are required for a reading; any other count is a
no-read. The JSON artifact beside the run records complete-reading exact
match, no-read, readable digit accuracy, readable `MAE`, per-image
predictions, settings, hashes, and the Ultralytics version. It never evaluates
the historical sanity holdout or a source listed in
`manifests/source_exclusions.csv`.

## Audit cross-validation failures

Once every fold for a recipe has a sequence-evaluation JSON, create a visual
error audit from the repository root:

```bash
npm run qa:full-image-digit-errors
```

The default command audits the frozen balanced48 folds. It keeps their saved
full-image predictions intact, evaluates the promoted ROI detector followed by
the corresponding fold-specific digit detector, and compares that cascade with
two diagnostic views: ground-truth-derived register-context inference and
one-aperture-at-a-time inference. The
timestamped output under `output/full-image-digit-error-audit/` includes an
HTML report, contact sheet, machine-readable JSON, and a transition-review
worksheet. The worksheet does not edit `annotations.csv`; inspect each wheel
and explicitly classify it before changing any canonical transition state.
The cascade mirrors the production ROI confidence, sanity, and expansion
geometry; reviewed digit boxes are used only for scoring and overlays.
It also predicts each ROI crop separately to mirror the deployed endpoint.
Ultralytics can pad a mixed-shape batch differently, and the earlier batched
audit overstated runtime performance.

The corrected August 4, 2026 out-of-fold cascade is `12/28` exact with `7`
no-reads and readable `MAE 15.71`; the register-context oracle is `17/28`
exact with `1` no-read. Every expanded ROI crop covers 100% of its reviewed
register, so increasing ROI expansion is not the next lever. Investigate and
standardize the single-image inference canvas/padding behavior before another
model or crop-geometry decision.

For another recipe, invoke `backend/export_full_image_digit_error_audit.py`
with one `--evaluation` argument per fold artifact.

## Benchmark one deployable shadow checkpoint

After the cross-validation audit identifies a plausible checkpoint, exercise
that single checkpoint through the real browser pipeline without changing the
selected reading:

```bash
npm run qa:full-image-digit-shadow
```

The default development checkpoint is the balanced48 fold-4 `best.pt` file;
override `FULL_IMAGE_DIGIT_SHADOW_MODEL_PATH` and
`FULL_IMAGE_DIGIT_SHADOW_VALIDATION_FOLD` together for another checkpoint.
The command starts a disposable backend, records the checkpoint SHA-256, and
writes a timestamped report under `output/full-image-digit-shadow-qa/`.

The complete UI comparison is useful for finding runtime failures but is not a
generalization estimate because a fold checkpoint trained on the other active
folds. Judge unseen-image behavior from the report's matching validation-fold
slice. Keep the frontend shadow disabled and do not promote a model if that
slice increases no-read, even when exact match or MAE improves.

August 4, 2026 result for the balanced48 fold-4 checkpoint: the complete
38-image development comparison was `24/38` exact with `8` no-reads and
readable `MAE 312.37`, versus production `11/38`, `2`, and `183.83`. That
headline includes 29 mapped training-overlap images. On the leakage-safe
seven-image fold-4 slice, the shadow improved exact match from `3/7` to `4/7`
and readable `MAE` from `40.00` to `17.67`, but no-read regressed from `0` to
`1`. Keep it shadow-only.

The follow-up bounded sensitivity run used exact single-image runtime inference
at confidences `0.10` through `0.30` and NMS IoUs `0.50`, `0.70`, and `0.90`.
Confidence `0.20` with the unchanged IoU `0.70` recovers the missing final `7`
on `meter_20260423.JPEG` at confidence `0.217`, producing `5/7` exact, `0`
no-reads, and readable `MAE 15.14` on fold 4. But the complete UI diagnostic
then admits a wrong leading `5` at confidence `0.218` on
`meter_20260724.JPEG`, changing its result from no-read to `5348` and worsening
full shadow `MAE` from `312.37` to `386.59`. Retain the `0.25` runtime default;
confidence alone cannot separate the recovered true box from the new false
box. The sensitivity fold is now tuning data and cannot serve as fresh
promotion evidence.

For the controlled mixed-scale recipe, add one register-context crop for each
training image:

```bash
python train_full_image_digit_detector.py \
  --fold 0 \
  --base-model yolov8n.pt \
  --train-register-crops \
  --device cpu
```

The crop is derived only from images assigned to training for that fold.
Validation and test inputs remain uncropped full images. Use this option first
on one fold; run its checkpoint through the sequence evaluator and run all
five folds only if both detection and complete-reading metrics support it.

For the next controlled recipe, keep the register crops and raise rare-class
training exposure to a floor of 24 with digit-centred crops:

```bash
python train_full_image_digit_detector.py \
  --fold 0 \
  --base-model yolov8n.pt \
  --train-register-crops \
  --train-balanced-digit-target 24 \
  --device cpu
```

Each generated crop contains one reviewed digit aperture plus varying amounts
of real context from its source photo. The generator uses only images assigned
to training for the selected fold and labels only the selected aperture. It
does not generate crops from validation or test images, and those two splits
remain uncropped full images with unchanged labels. The floor augments rare
classes without duplicating already-abundant classes up to the dominant-class
count; this limits repeated exposure to the few rare source apertures.

Run the same command with `--validate-only` before training to inspect the
provenance summary. It reports generated crop counts and source-image counts
per class as well as the final class counts for every split.

## July 24, 2026 active-scope comparison

After `meter_20201111.JPEG` was classified as a retained legacy-stress source,
the same three already-trained fold-0 checkpoints were re-evaluated on the
same revised six-image active validation fold:

| Recipe | mAP50 | mAP50-95 | Readable | Exact | Digit accuracy | MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full images | 0.434 | 0.268 | 1/6 | 0/6 | 75.0% | 20.0 |
| Full + register crops | 0.639 | 0.464 | 5/6 | 0/6 | 60.0% | 21.6 |
| Full + register + balanced digit crops | 0.667 | 0.484 | 6/6 | 0/6 | 66.7% | 12.0 |

These are not newly trained models. Any difference from the earlier
seven-image reports is an evaluation-scope correction, not a model
improvement. The balanced checkpoint ranks best on this intended active scope,
but `0/6` exact readings still blocks promotion and a five-fold expansion.

Repeat folds `0` through `4` to compare recipes. Keep their checkpoints under
`runs/`; do not copy a checkpoint to
`models/full_image_digit_detector.pt` or refactor runtime OCR until
cross-validation supports freezing a recipe.

Before promotion, collect a new locked external set of independent full images
captured across varied dates and conditions. Evaluate the frozen recipe once
on that set. The existing single sanity holdout may remain a historical
diagnostic, but it is not a promotion gate.
