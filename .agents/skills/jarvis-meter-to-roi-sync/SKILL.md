---
name: jarvis-meter-to-roi-sync
description: "Run a full Jarvis meter-image ingestion in one flow: clean assets sidecars, rename new photos from EXIF, read meter values, update assets/meter_readings.csv, then sync and label the new meter_* files into backend/data/roi_dataset with QA previews. Use when new water-meter photos are added to assets/ and should immediately be reflected in both CSV readings and ROI training data."
---

# Jarvis Meter To ROI Sync

Execute this workflow from the Jarvis repository root.

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

5. Upsert `assets/meter_readings.csv`.
   - Keep header `filename,value`.
   - Add/update one row per renamed file.
   - Do not duplicate existing filenames.

6. Sync new `meter_*` files into ROI dataset.
   - `cd /home/andrea/GitHubRepositories/Jarvis`
   - `source backend/.venv/bin/activate`
   - `python /home/andrea/.codex/skills/jarvis-roi-dataset-sync/scripts/sync_roi_dataset.py --repo-root . --split train`

7. Review new-box QA previews.
   - Check `<name>_qa.jpg` and `<name>_zoom.jpg` under:
     - `backend/data/roi_dataset/qa_previews/new_boxes_sync/<timestamp>/`
   - Prioritize any `LOW_CONF` or `MISSING` output lines.

8. Correct labels when needed.
   - Edit `backend/data/roi_dataset/labels/<split>/<name>.txt`.
   - Keep YOLO format: `0 x_center y_center width height` (normalized).
   - Target only the 4-digit black register window.

9. Re-render full ROI QA overlays.
   - `cd backend && source .venv/bin/activate && python visualize_roi_labels.py`

10. Validate and summarize.
   - Confirm no sidecars remain.
   - Confirm every CSV filename exists in `assets/`.
   - Report:
     - renamed files
     - CSV rows added/updated
     - ROI images added count
     - `LOW_CONF` and `MISSING` counts
     - final label files updated

## Command Snippets

- Sidecars left:
  - `find assets -type f -name '*:Zone.Identifier' -print`
- CSV to file consistency:
  - `awk -F, 'NR>1 {print $1}' assets/meter_readings.csv | while read -r f; do [ -f "assets/$f" ] || echo "missing: $f"; done`
- Sync dry run (ROI stage only):
  - `source backend/.venv/bin/activate && python /home/andrea/.codex/skills/jarvis-roi-dataset-sync/scripts/sync_roi_dataset.py --repo-root . --split train --dry-run`

## Notes

- Default ROI model for sync helper: `backend/models/roi-rotaug-e30-640.pt`.
- Override model path with `--model-path backend/models/<other-model>.pt` when needed.
