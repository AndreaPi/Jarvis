# Photo normalization

Use from the repository root with `backend/.venv`, only for raw imports in the
selected batch. Keep a source-to-canonical filename mapping for resumed work.

- Read capture metadata before naming files. For JPEG/PNG, use EXIF when
  available: `identify -format '%[EXIF:DateTimeOriginal]\n' assets/<file>`.
  On macOS, `sips -g creation assets/<file>` is a fallback when `identify` is
  unavailable. If no reliable capture date is available, ask for the date;
  do not silently substitute the import date.
- Force a full Pillow pixel decode with `load()` before renaming or converting.
  For HEIC/HEIF, register the `pillow-heif` opener first. Metadata and dimensions
  from `sips` do not prove an iCloud-backed file contains usable pixels.
- If decoding fails, preserve the source and report the affected file for
  re-download/export. Searching and exporting the matching original from
  Photos is a recovery path only when explicitly authorized.
- Rename JPEG/PNG to `meter_yyyymmdd` while preserving extension/case.
- Convert HEIC/HEIF to `meter_yyyymmdd.JPEG` using Pillow plus `pillow-heif`,
  applying orientation consistently. Never overwrite an existing canonical
  target; append `_1`, `_2`, `_3`, etc. for name collisions.
- Write conversion output to a temporary sibling file. Reopen it, call
  `load()`, and verify positive dimensions before atomically placing it at the
  collision-free canonical path. Failed conversion/validation keeps the source.
- Delete the original batch HEIC/HEIF only after the canonical JPEG passes full
  pixel validation. This is required import cleanup, not annotation approval;
  if the execution environment requires deletion approval, request it and
  continue when granted. Do not delete unrelated imports.
- Use only the canonical JPEG/PNG filename in readings, ROI manifests, and DVC
  metadata. Finish successful batch conversion cleanup before downstream writes;
  a blocked cleanup may still be reported in a pending-work summary.
