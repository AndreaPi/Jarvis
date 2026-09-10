---
name: jarvis-meter-to-roi-sync
description: "Ingest Jarvis meter photos into readings and the ROI dataset, or resume and correct their manual ROI annotation review."
---

# Jarvis Meter To ROI Sync

Execute this workflow from the Jarvis repository root.

Use `backend/.venv` for any Python step in this workflow. Do not rely on the system Python for CV or image dependencies such as `ultralytics` or `Pillow`.

Run mutating command sequences fail-fast. Use separate checked commands or `set -euo pipefail`; never let rename, conversion, validation, deletion, dataset rebuild, or DVC steps continue after an earlier failure. Record `git status --short` before starting and leave unrelated Git or DVC changes untouched.

## Workflow

Choose the entry point from the requested batch and verified artifact state:

- New photos: normalize, establish readings, then collect ROI labels.
- Supplied labels or a resumed review: verify the existing canonical photos,
  readings, manifest, and persistent splits, then continue at the first unmet step.
- ROI corrections: update only the requested targets, rebuild, and review the
  changed overlays. Do not repeat ingestion or change existing split assignments.

A manual export supplied by the user establishes human review of that ROI on
the annotated canonical image. A verified equivalent import and agent visual QA
satisfy the ROI review gate without a second human overlay confirmation. Reuse
that result on resumed work with unchanged inputs. Changed source/box geometry
or unresolved QA requires review of the affected targets; the checks below
define that boundary. Complete independent authorized work while input is pending.

1. Establish the batch and remove its Windows sidecars.
   - Record target filenames and existing split assignments. Delete only
     `:Zone.Identifier` sidecars belonging to this batch; preserve unrelated files.

2. Ingest new photos in `assets/`.
   - Use the user's named files when supplied. Otherwise scan `assets/` for
     import candidates; report the selected batch before normalization.
   - Suggested scan: `find assets -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.heic' -o -iname '*.heif' \) -print`
   - For discovery, treat `jpg|jpeg|png|heic|heif` (any case) files not already
     named `meter_*` or listed in `assets/meter_readings.csv` as candidates.
     These filters do not exclude explicitly selected canonical files from a
     resumed review. Do not add other discovered files to an explicit batch.
   - HEIC/HEIF is a Mac/iCloud import format only for this repo; do not keep HEIC/HEIF as canonical assets.

3. Normalize each candidate from capture metadata.
   - For raw imports, read [photo normalization](references/photo-normalization.md).
     Skip normalization for already verified canonical images.
   - Fully decode source pixels before renaming/conversion and validate the
     converted JPEG before deleting a HEIC/HEIF source. Keep a failed source.
   - Use the canonical JPEG/PNG filename in CSV, ROI, and DVC metadata.

4. Read and record the meter value.
   - Read the 4-digit black register only.
   - Ignore red fractional dials.
   - If needed, create temporary enhanced views (rotate/crop/contrast/resize) before reading.
   - For transition wheels, choose the conservative pre-roll digit unless rollover is clearly complete.
   - If you use the live app as an OCR assist, remember the current pipeline is neural-ROI + digit-classifier only and requires a healthy backend on `127.0.0.1:8001`. Treat OCR as a first pass only and confirm the reading manually before writing CSV rows.

5. Upsert `assets/meter_readings.csv`.
   - Keep header `filename,value`.
   - Add/update one row per renamed file.
   - Do not duplicate existing filenames.

6. Collect manual ROI labels for the new images.
   - Read [manual ROI labels](references/manual-roi-labels.md) when requesting,
     importing, or correcting labels. Do not auto-estimate boxes as the default.
   - Reuse matching labels already supplied for the batch; request only missing
     or corrected labels. Require one valid manual box per target, with an
     unambiguous filename mapping, before syncing the batch.

7. Sync uploaded ROI labels into the repo workflow.
   - Retain the original export coordinates and the annotated canonical image's
     hash, dimensions, and EXIF orientation in batch QA evidence before moving
     labels. Keep this evidence available for post-build and resumed checks.
   - Parse each uploaded YOLO TXT label in the batch and convert it to manifest rect format:
     - `x = x_center - width / 2`
     - `y = y_center - height / 2`
     - `width = width`
     - `height = height`
   - Upsert the converted entry in `backend/data/roi_boxes_manifest.json`:
     - `{"filename": "meter_YYYYMMDD.JPEG", "rectNorm": {"x": ..., "y": ..., "width": ..., "height": ...}}`
   - Move each consumed TXT label from `assets/` to
     `backend/data/roi_dataset/labels/<assigned-split>/<stem>.txt`; use the
     persistent split for existing images and `train` for new images.
   - Run label validation and moves from the repository root, or use absolute paths. Do not rely on a `cd backend` from an earlier command.
   - Do not rebuild the ROI dataset until every image in the new batch has a matching manifest entry.
   - Keep `backend/data/roi_boxes_manifest.json` as the source of truth; generated label files must stay aligned with it after rebuild.

8. Rebuild the ROI dataset from the current CSV + ROI manifest.
   - From the repository root, run `backend/.venv/bin/python backend/build_roi_dataset.py --roi-json data/roi_boxes_manifest.json`. The script resolves relative arguments from `backend/`, not from the shell's working directory.
   - The builder persists split assignments in `backend/data/roi_dataset/splits.json`.
   - Existing images keep their assigned split; new images default to `train` unless you edit `splits.json`.
   - The builder updates the ROI dataset to match the CSV + manifest without recomputing old splits from CSV order.
   - Treat imported boxes as pending validation until the consistency and
     agent visual checks below pass; a builder exit code alone is insufficient.

9. Review generated ROI previews.
   - Check `backend/data/roi_dataset/previews/*_bbox.jpg` for quick bounding-box QA.
   - Re-render full QA overlays from the repository root with `backend/.venv/bin/python backend/visualize_roi_labels.py`.
   - Compare the generated ROI dataset image to the reviewed canonical photo:
     filename mapping, file hash, dimensions, and orientation must agree.
   - Compare the original exported box, manifest rectangle, and generated YOLO
     label numerically, allowing only the label's six-decimal serialization
     rounding. Reject silent clipping, padding, coordinate swaps, or rotations.
   - Inspect target overlays under `backend/data/roi_dataset/qa_previews/` at
     full resolution and confirm the box covers the complete black register.
   - If the manual export, image identity, numerical checks, and agent visual QA
     agree, mark the ROI validated and continue authorized work. Show or link the
     preview for transparency without asking for a second human confirmation.
     Report manual annotation and agent QA accurately; do not claim the user
     approved a generated overlay they did not inspect.
   - If an import defect can be corrected to match the supplied export exactly,
     fix it and rerun affected checks. Ask for human review only when equivalence
     cannot be established, the annotated image/box geometry changes, or visual
     ambiguity remains. Explain the discrepancy and show the affected overlay.
   - Unresolved validation or required review blocks DVC publication, commit of
     affected annotations, and training-ready claims. Continue independent checks
     and report pending work. Annotation review does not authorize training,
     model promotion, or a previously unauthorized external upload.

10. Correct labels when needed.
   - Edit the source-of-truth entry in `backend/data/roi_boxes_manifest.json`, not the generated label file.
   - Keep rect format: `{"x": ..., "y": ..., "width": ..., "height": ...}` normalized to the full image.
   - Target only the 4-digit black register window.
   - Follow the manual-label reference for corrections and repeat the label
     sync step for the affected targets.
   - Re-run `build_roi_dataset.py` after any manifest correction so the generated labels stay aligned.
   - Re-run `visualize_roi_labels.py` after a correction and apply the checks in
     step 9. A corrected manual export supplies the human review for that box;
     agent-proposed geometry changes require human review before acceptance.
   - Remove any batch-specific `:Zone.Identifier` sidecars in the ROI outputs.

11. Refresh DVC-tracked artifacts.
   - Run `backend/.venv/bin/python -m dvc add backend/data/roi_dataset/images`.
   - Run `backend/.venv/bin/python -m dvc add assets/<new-meter-file>` for each newly ingested canonical photo.
   - For Mac/iCloud imports, DVC-track the converted `meter_YYYYMMDD.JPEG`, not the original `IMG_*.HEIC`/`IMG_*.HEIF`.
   - If DVC tries to use an unwritable system cache such as `/Library/Caches/dvc`, rerun with `DVC_SITE_CACHE_DIR=/tmp/dvc-site-cache`.
   - Run `scripts/dvc-push-safe.sh` only with a configured off-machine remote. The guard refuses plain local paths and `file://` URLs.
   - Push only the updated pointers for this batch, then run target-specific `dvc status` on those pointers. A global DVC status may expose unrelated dirty outputs; report them but do not repair or include them.
   - Proceed only after the ROI review gate in step 9 passes and publication is
     authorized. Resolving the annotation gate does not bypass upload permissions.

12. Validate and summarize.
   - Confirm no batch sidecars or successfully converted batch HEIC/HEIF sources
     remain. Report failed sources as pending; preserve unrelated imports.
   - Confirm every CSV filename exists in `assets/`.
   - Confirm every target has its manual label and manifest entry, generated
     labels agree with the manifest, and existing split assignments are unchanged.
   - Reuse passing checks for unchanged inputs and outputs; a user confirmation
     alone does not require rebuilding datasets or rerunning QA.
   - Report:
     - renamed files
     - CSV rows added/updated
     - ROI dataset rows/images rebuilt
     - preview images regenerated
     - final label files updated
     - DVC pointers pushed and their target-specific status
     - unrelated pre-existing Git or DVC changes, if any
     - manual-export provenance, consistency/agent visual QA results, and any
       exceptional human review or Make Sense correction still pending
   - After the ROI review gate passes and the batch's canonical photos and ROI
     images are published, hand off the canonical filenames to
     `jarvis-meter-to-digit-box-sync` when the new photos should join the
     full-image digit-detector dataset. The ROI review result does not replace
     the separate digit-box review.

For a full ingestion request, completion means the batch has manual ROI labels,
passes the ROI review gate, and its changed canonical photo/ROI binaries are safely
published through DVC. For a narrower preparation, review, or correction
request, complete the requested outputs and report remaining gates without
expanding into publication or downstream work outside that scope. When a gate
is pending, report completed work, exact review links or missing inputs, and
remaining steps. Git commit/push and training require authorization in the
user's request; reuse existing authorization once applicable gates pass.
