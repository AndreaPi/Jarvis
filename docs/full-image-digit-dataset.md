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
