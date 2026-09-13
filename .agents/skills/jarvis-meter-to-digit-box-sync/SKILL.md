---
name: jarvis-meter-to-digit-box-sync
description: "Continue an approved Jarvis meter-photo ingestion into the full-image digit-detector dataset: verify the trusted reading, ROI, canonical orientation, and retained artifacts; bootstrap four digit-aperture boxes with classes 0-9; prepare a Make Sense review batch; import corrected YOLO annotations; preserve source exclusions and persistent CV folds; regenerate labels and previews; and enforce the human-review training gate. Use after jarvis-meter-to-roi-sync for new canonical meter photos, or when existing full-image digit boxes need review or correction."
---

# Jarvis Meter To Digit Box Sync

Execute this workflow from the Jarvis repository root. Use `backend/.venv` for
every Python step.

This skill starts only after `jarvis-meter-to-roi-sync` has completed. It does
not ingest raw photos, decide trusted readings, approve register ROIs, train a
detector, or promote a model.

Run mutating sequences fail-fast. Record `git status --short` first and preserve
unrelated Git and DVC changes. Do not rebuild while
`train_full_image_digit_detector.py` is running.

## Workflow

1. Establish the approved batch.
   - Use the canonical `meter_YYYYMMDD.JPEG` or `.PNG` filenames handed off by
     `jarvis-meter-to-roi-sync`.
   - Do not silently add every unannotated file in the worktree to the batch.
   - For every target, verify:
     - the canonical photo exists in `assets/`;
     - `assets/meter_readings.csv` contains one trusted four-digit reading;
     - `backend/data/roi_boxes_manifest.json` contains the approved register
       ROI;
     - the matching image and one-box label exist in the persistent ROI split;
     - the photo and ROI image dataset DVC targets were published after the
       user approved the ROI overlay.
   - If any condition is missing, stop this workflow and finish
     `jarvis-meter-to-roi-sync`. ROI approval is not digit-box approval.

2. Verify canonical reading orientation.
   - Confirm every target has one row in
     `backend/data/digit_dataset/manifests/canonical_windows.csv`, with the same
     split and reading as the approved upstream data.
   - If the row is missing, incrementally refresh the existing digit
     preparation pipeline:

     ```bash
     backend/.venv/bin/python backend/extract_digit_windows.py
     backend/.venv/bin/python backend/split_digit_windows.py --clean
     backend/.venv/bin/python backend/label_digit_sections.py --clean
     backend/.venv/bin/python backend/validate_digit_dataset.py
     npm run qa:strip-dataset
     ```

   - Do not pass `--clean` to `extract_digit_windows.py` during additive
     ingestion. That flag removes the whole `data/digit_dataset` root,
     including unrelated synthetic artifacts and DVC pointers. Reserve it for
     an explicitly requested full regeneration that also rebuilds every
     affected derived dataset.
   - Run `npm run qa:strip-dataset` when no fresh QA artifact covers the target.
     Show the target canonical-strip QA previews to the user and require
     explicit confirmation that their left-to-right order matches the trusted
     reading, even when a canonical manifest row already existed. If the
     direction is wrong, correct
     `backend/data/digit_dataset/manifests/direction_overrides.csv`, regenerate
     from `split_digit_windows.py` onward, and repeat QA.
   - If orientation preparation changed the standard digit datasets, refresh
     and safely publish only their DVC targets:

     ```bash
     backend/.venv/bin/python -m dvc add \
       backend/data/digit_dataset/windows \
       backend/data/digit_dataset/windows_canonical \
       backend/data/digit_dataset/sections \
       backend/data/digit_dataset/sections_labeled
     scripts/dvc-push-safe.sh \
       backend/data/digit_dataset/windows.dvc \
       backend/data/digit_dataset/windows_canonical.dvc \
       backend/data/digit_dataset/sections.dvc \
       backend/data/digit_dataset/sections_labeled.dvc
     backend/.venv/bin/python -m dvc status \
       backend/data/digit_dataset/windows.dvc \
       backend/data/digit_dataset/windows_canonical.dvc \
       backend/data/digit_dataset/sections.dvc \
       backend/data/digit_dataset/sections_labeled.dvc
     ```

     If DVC needs an alternate cache, set
     `DVC_SITE_CACHE_DIR=/tmp/dvc-site-cache`. Never run raw `dvc push`.

3. Snapshot the full-image dataset boundary.
   - Read, do not infer, the existing:
     - `backend/data/full_image_digit_dataset/manifests/annotations.csv`;
     - `backend/data/full_image_digit_dataset/manifests/cv_folds.csv`;
     - `backend/data/full_image_digit_dataset/manifests/source_exclusions.csv`;
     - `backend/data/full_image_digit_dataset/manifests/summary.json`.
   - Record the target filenames, existing review states, exclusions, and CV
     assignments so the post-build comparison can detect unintended changes.
   - Preserve `source_exclusions.csv`. Do not exclude a new image because it is
     difficult without an explicit user decision.

4. Bootstrap the four digit boxes.
   - Run:

     ```bash
     backend/.venv/bin/python backend/build_full_image_digit_dataset.py
     ```

   - Never pass `--rebuild-cv-folds` during ingestion.
   - Confirm each new target seeded exactly four rows in `annotations.csv`, at
     positions `0..3`, with digits/classes equal to the trusted reading,
     `annotation_source=bootstrap-roi-split`, and
     `review_status=pending`.
   - Confirm existing reviewed rows, existing fold assignments, and exclusions
     are unchanged. A new train source may receive one new persistent CV fold;
     a test source must not receive one.

5. Present the review batch.
   - Inspect and show the target files under
     `output/full-image-digit-review/previews/crops/`. Use
     `digit-box-contact-sheet.jpg` only as an overview; full-resolution images
     are authoritative.
   - Lead every initial or corrective Make Sense annotation request with a
     prominent clickable Markdown link to
     `[Open Make Sense](https://www.makesense.ai/)`; localize the link text to
     the user's language when useful. Prefer the link over opening an external
     browser automatically, and never require the user to remember or type the
     site address.
   - In the same message, provide clickable local links for every target image,
     the class-list file, the matching bootstrap annotation files, and the
     exact ZIP or directory path expected from the user's export.
   - Ask the user to review the batch in Make Sense:
     1. Upload only the target files from
        `output/full-image-digit-review/images/`.
     2. Choose **Object Detection**.
     3. Load the class list from
        `output/full-image-digit-review/annotations/labels.txt`.
     4. Import only the matching `<meter-stem>.txt` files from that same
        annotations directory. Do not import `labels.txt` as annotations.
     5. Keep exactly four boxes per image. Each box must cover one complete
        digit-wheel aperture, not only the dark glyph.
     6. Verify that the classes form the trusted reading in reading order,
        correct boxes/classes when needed, retain the original filenames, and
        export YOLO annotations.
   - Even when the bootstrap looks correct, require a Make Sense export. That
     export is the durable evidence that the boxes were human-reviewed.
   - Stop and wait for the exported ZIP or directory path.

6. Import the reviewed subset.
   - For a normal new-image batch, run:

     ```bash
     backend/.venv/bin/python backend/import_full_image_digit_annotations.py \
       /path/to/makesense-export.zip \
       --allow-partial
     backend/.venv/bin/python backend/build_full_image_digit_dataset.py
     ```

   - Omit `--allow-partial` only when the export intentionally contains every
     active review-package image.
   - Let the importer reject unknown images, out-of-bounds coordinates, any
     count other than four boxes, and class sequences that do not match the
     trusted reading. Do not bypass these checks.
   - The import must update `annotations.csv` as the source of truth. Derived
     YOLO labels, `summary.json`, fold assignment, and previews come from the
     following builder run; do not edit generated label files directly.

7. Enforce the final gate.
   - Confirm every target now has exactly four rows with:
     - positions `0..3`;
     - `annotation_source=human-makesense`;
     - `review_status=reviewed`;
     - classes equal to the trusted reading.
   - Confirm the target's derived YOLO label exists in its active split,
     exclusions are unchanged, old CV assignments are unchanged, and each new
     train target retains its newly seeded fold.
   - Inspect the regenerated target crop previews and ask the user for explicit
     final approval.
   - Run `npm run test:backend`.
   - Declare the whole full-image dataset training-ready only when
     `summary.json` reports no active `pending` annotations and the user has
     approved the regenerated previews. If unrelated pending rows remain,
     report the target batch as reviewed but the dataset as not training-ready.

8. Summarize without training.
   - Report the target filenames and readings, imported row count, review
     status, regenerated labels/previews, new fold assignments, validation
     result, user approval state, and unrelated pre-existing changes.
   - Report any condition that prevents the training-ready claim.
   - Do not start training, evaluate checkpoints, promote models, rebuild CV
     folds, or publish release artifacts unless the user requests that as a
     separate next step.

## Invariants

- `annotations.csv` is the canonical human-review layer; generated labels are
  disposable derivatives.
- Bootstrap runs may seed missing rows but must never overwrite reviewed rows.
- `cv_folds.csv` is persistent. Do not recompute it during routine ingestion.
- `source_exclusions.csv` defines active scope while retaining legacy-stress
  artifacts.
- Validation and test images are never augmentation sources.
- Human Make Sense review and explicit preview approval remain mandatory.
- Routine additive ingestion must preserve the synthetic digit dataset and its
  DVC pointer unchanged.
