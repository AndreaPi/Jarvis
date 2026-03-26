---
name: jarvis-meter-to-roi-sync
description: "Run a full Jarvis meter-image ingestion flow: clean assets sidecars, rename new photos from EXIF, read and confirm meter values, update assets/meter_readings.csv, then rebuild backend/data/roi_dataset from a browser-extracted ROI manifest with QA previews. Use when new water-meter photos are added to assets/ and should immediately be reflected in both CSV readings and ROI training data."
---

# Jarvis Meter To ROI Sync

Execute this workflow from the Jarvis repository root.

Use `backend/.venv` for any Python step in this workflow. Do not rely on the system Python for CV or image dependencies such as `ultralytics` or `Pillow`.

## Workflow

1. Remove Windows sidecars every run.
   - `find assets -type f -name '*:Zone.Identifier' -delete`

2. Ingest new photos in `assets/`.
   - Treat `jpg|jpeg|png` (any case) files not already named `meter_*` as candidates.
   - Exclude names already listed in `assets/meter_readings.csv`.

3. Rename each candidate from EXIF date.
   - Read EXIF: `identify -format '%[EXIF:DateTimeOriginal]\n' assets/<file>`
   - Rename to `meter_yyyymmdd` while preserving extension/case.
   - If target exists, append `_1`, `_2`, `_3`, ...

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
   - This rebuilds `backend/data/roi_dataset` from the full CSV + manifest, not an incremental sync of only new files.

8. Review generated ROI previews.
   - Check `backend/data/roi_dataset/previews/*_bbox.jpg` for quick bounding-box QA.

9. Correct labels when needed.
   - Edit `backend/data/roi_dataset/labels/<split>/<name>.txt`.
   - Keep YOLO format: `0 x_center y_center width height` (normalized).
   - Target only the 4-digit black register window.

10. Re-render full ROI QA overlays.
   - `cd backend && source .venv/bin/activate && python visualize_roi_labels.py`
   - Review outputs under `backend/data/roi_dataset/qa_previews/`.

11. Refresh DVC-tracked artifacts.
   - Run `dvc add backend/data/roi_dataset/images`
   - Run `dvc add assets/<new-meter-file>` for each newly ingested raw photo
   - Run `scripts/dvc-push-safe.sh` if a DVC remote is configured

12. Validate and summarize.
   - Confirm no sidecars remain.
   - Confirm every CSV filename exists in `assets/`.
   - Confirm every newly ingested filename has a ROI manifest entry before rebuilding.
   - Report:
     - renamed files
     - CSV rows added/updated
     - ROI dataset rows/images rebuilt
     - preview images regenerated
     - final label files updated

## Command Snippets

- Sidecars left:
  - `find assets -type f -name '*:Zone.Identifier' -print`
- CSV to file consistency:
  - `awk -F, 'NR>1 {print $1}' assets/meter_readings.csv | while read -r f; do [ -f "assets/$f" ] || echo "missing: $f"; done`
- Rebuild ROI dataset:
  - `cd backend && source .venv/bin/activate && python build_roi_dataset.py --roi-json /absolute/path/to/roi_boxes.json`
- Re-render ROI QA overlays:
  - `cd backend && source .venv/bin/activate && python visualize_roi_labels.py`

## Notes

- The old external sync helper path is obsolete for this repo; use a browser-extracted ROI manifest plus `backend/build_roi_dataset.py`.
- The browser-assisted OCR path is not authoritative for CSV updates. Always confirm readings manually before writing `assets/meter_readings.csv`.
- Raw meter photos and ROI image binaries are retained with DVC; do not leave new ingested files outside DVC tracking.
