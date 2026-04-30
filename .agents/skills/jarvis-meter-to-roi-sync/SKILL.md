---
name: jarvis-meter-to-roi-sync
description: "Run a full Jarvis meter-image ingestion flow: clean assets sidecars, normalize JPEG/PNG/HEIC photos from capture metadata, read and confirm meter values, update assets/meter_readings.csv, then rebuild backend/data/roi_dataset from a browser-extracted ROI manifest with QA previews. Use when new water-meter photos are added to assets/ and should immediately be reflected in both CSV readings and ROI training data."
---

# Jarvis Meter To ROI Sync

Execute this workflow from the Jarvis repository root.

Use `backend/.venv` for any Python step in this workflow. Do not rely on the system Python for CV or image dependencies such as `ultralytics` or `Pillow`.

## Workflow

1. Remove Windows sidecars every run.
   - `find assets -type f -name '*:Zone.Identifier' -delete`

2. Ingest new photos in `assets/`.
   - Treat `jpg|jpeg|png|heic|heif` (any case) files not already named `meter_*` as candidates.
   - Exclude names already listed in `assets/meter_readings.csv`.
   - HEIC/HEIF is a Mac/iCloud import format only for this repo; do not keep HEIC/HEIF as canonical assets.

3. Normalize each candidate from capture metadata.
   - For JPEG/PNG, read EXIF when available: `identify -format '%[EXIF:DateTimeOriginal]\n' assets/<file>`.
   - If `identify` is unavailable on macOS, use `sips -g creation assets/<file>` as a fallback capture timestamp.
   - Rename JPEG/PNG to `meter_yyyymmdd` while preserving extension/case.
   - Convert HEIC/HEIF to a canonical JPEG named `meter_yyyymmdd.JPEG`. Use `backend/.venv` plus `pillow-heif` for conversion if macOS tooling cannot produce a normal JPEG that Pillow can reopen.
   - If target exists, append `_1`, `_2`, `_3`, ...
   - After HEIC/HEIF conversion, verify the target JPEG exists and can be opened by `backend/.venv` Pillow, then delete the original HEIC/HEIF file from `assets/`.
   - Never write HEIC/HEIF filenames to `assets/meter_readings.csv`, `backend/data/roi_boxes_manifest.json`, or DVC metadata; use the converted JPEG filename everywhere downstream.

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

6. Prepare or refresh the ROI manifest for the new images.
   - Current repo workflow requires a browser-extracted ROI JSON manifest with entries shaped like:
     - `{"filename": "meter_YYYYMMDD.JPG", "rectNorm": {"x": ..., "y": ..., "width": ..., "height": ...}}`
   - If the manifest does not yet include the new images, generate or extend it first.
   - Do not use the old external `jarvis-roi-dataset-sync` helper path; the repo now expects `backend/build_roi_dataset.py` with a manifest input.

7. Rebuild the ROI dataset from the current CSV + ROI manifest.
   - `cd /home/andrea/GitHubRepositories/Jarvis/backend`
   - `source .venv/bin/activate`
   - `python build_roi_dataset.py --roi-json <path-to-roi-json>`
   - The builder persists split assignments in `backend/data/roi_dataset/splits.json`.
   - Existing images keep their assigned split; new images default to `train` unless you edit `splits.json`.
   - The builder updates the ROI dataset to match the CSV + manifest without recomputing old splits from CSV order.
   - Treat any new or auto-estimated ROI boxes as tentative until the user reviews the generated previews.

8. Review generated ROI previews.
   - Check `backend/data/roi_dataset/previews/*_bbox.jpg` for quick bounding-box QA.
   - Re-render full QA overlays with `cd backend && source .venv/bin/activate && python visualize_roi_labels.py`.
   - Review outputs under `backend/data/roi_dataset/qa_previews/`.
   - Explicitly prompt the user to inspect the new image overlays before treating the labels as training-ready.
   - Do not continue to DVC push, final summary, commit, or promotion language until the user either approves the labels or asks for corrections.

9. Correct labels when needed.
   - Edit the source-of-truth entry in `backend/data/roi_boxes_manifest.json`, not the generated label file.
   - Keep rect format: `{"x": ..., "y": ..., "width": ..., "height": ...}` normalized to the full image.
   - Target only the 4-digit black register window.
   - Preferred correction path: use Make Sense (or another manual labeling tool), export the corrected box, then sync that correction back into `backend/data/roi_boxes_manifest.json`.
   - Re-run `build_roi_dataset.py` after any manifest correction so the generated labels stay aligned.
   - Re-run `visualize_roi_labels.py` after any correction and ask the user to confirm the updated overlay.
   - After the user approves the corrected overlay, scan for stray `:Zone.Identifier` files under `backend/data/roi_dataset/` and delete them before continuing.

10. Refresh DVC-tracked artifacts.
   - Run `dvc add backend/data/roi_dataset/images`
   - Run `dvc add assets/<new-meter-file>` for each newly ingested canonical photo.
   - For Mac/iCloud imports, DVC-track the converted `meter_YYYYMMDD.JPEG`, not the original `IMG_*.HEIC`/`IMG_*.HEIF`.
   - Run `scripts/dvc-push-safe.sh` if a DVC remote is configured
   - Only do this after the user has approved the ROI overlays for the new image(s).

11. Validate and summarize.
   - Confirm no sidecars remain.
   - Confirm no HEIC/HEIF files remain in `assets/` after conversion.
   - Confirm every CSV filename exists in `assets/`.
   - Confirm every newly ingested filename has a ROI manifest entry before rebuilding.
   - Report:
     - renamed files
     - CSV rows added/updated
     - ROI dataset rows/images rebuilt
     - preview images regenerated
     - final label files updated
     - whether the user explicitly approved the new ROI labels or whether further Make Sense correction is still pending

## Command Snippets

- Sidecars left:
  - `find assets -type f -name '*:Zone.Identifier' -print`
- HEIC/HEIF imports left after conversion:
  - `find assets -type f \( -iname '*.heic' -o -iname '*.heif' \) -print`
- ROI sidecars left:
  - `find backend/data/roi_dataset -type f -name '*:Zone.Identifier' -print`
- CSV to file consistency:
  - `awk -F, 'NR>1 {print $1}' assets/meter_readings.csv | while read -r f; do [ -f "assets/$f" ] || echo "missing: $f"; done`
- Rebuild ROI dataset:
  - `cd backend && source .venv/bin/activate && python build_roi_dataset.py --roi-json /absolute/path/to/roi_boxes.json`
- Re-render ROI QA overlays:
  - `cd backend && source .venv/bin/activate && python visualize_roi_labels.py`

## Notes

- The old external sync helper path is obsolete for this repo; use a browser-extracted ROI manifest plus `backend/build_roi_dataset.py`.
- Mac/iCloud HEIC/HEIF images should be treated as temporary import sources. Convert them to canonical JPEG assets, verify the JPEGs, then delete the HEIC/HEIF originals.
- The browser-assisted OCR path is not authoritative for CSV updates. Always confirm readings manually before writing `assets/meter_readings.csv`.
- New ROI labels are not training-ready until the user has reviewed the generated overlays and either approved them or corrected them in Make Sense.
- Canonical meter photos and ROI image binaries are retained with DVC; do not leave new ingested JPEG/PNG files outside DVC tracking.
