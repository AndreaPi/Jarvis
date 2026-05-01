# OCR Guidelines

## Scope
- This file covers the OCR pipeline in `src/ocr/`, including current behavior, benchmark baselines, and promotion policy.

## Current Working State
- Neural ROI is mandatory in the frontend OCR flow; heuristic ROI fallback has been removed.
- On neural ROI failure, the UI shows an explicit reason and asks for manual measurement input.
- Digit-classifier inference is mandatory in the frontend OCR flow (`OCR_CONFIG.digitClassifier.enabled` defaults to `true`).
- The whole-strip digit reader (`OCR_CONFIG.digitStripReader`) runs in shadow mode only. Its predictions are logged for comparison and debug, but they must not change user-visible OCR selection until a benchmark explicitly promotes it.
- Frontend OCR evaluation is strip-only, classifier-first candidate decoding. The old Tesseract word-pass and sparse-scan stages are not part of the active path.
- Edge-derived candidate generation is enabled by default and can be toggled with `OCR_CONFIG.roiDeterministic.useEdgeCandidates`.
- The primary classifier shortlist now mixes high-ranked edge and base strip candidates so valid full-strip rotations are not starved behind edge-only passes.
- Opposite-orientation retry is disabled by default (`roiDeterministic.tryOppositeOrientation=false`).
- The default ROI checkpoint remains `backend/models/roi-rotaug-e30-640.pt` until a challenger wins on end-to-end OCR metrics.

## Debug Overlay Semantics
- `6a. OCR input candidate (initial preview)` is the first valid ROI candidate before classifier ranking.
- `6. OCR input candidate` is the final winning decode input.
- `7. classifier cell crops` shows the four cell crops passed to the current primary per-cell digit classifier.
- `8. strip reader input` shows the best whole-strip shadow-reader input and compact prediction/confidence summary.
- Diff artifacts and benchmark reports should use the last `6. OCR input candidate` frame as the selected OCR input snapshot.

## Active Benchmark Baseline
- Current local benchmark set: `26` images.
- Current promotion target:
  - UI test set: `MAE 64.36`, `Exact Match 11/26`, `No-read 1/26`
  - `npm run test:e2e`: passes (`7/7`)
- Use `MAE` as the primary promotion signal, with `Exact Match` and `No-read` as guardrails.

## OCR Workflow and Guardrails
- Before committing OCR changes, run both `npm run test:e2e` and the UI `Run test set`.
- Prefer running the test set with the debug overlay enabled.
- Test-set review should inspect `Detected`, `Absolute Error`, `Failure Reason`, stages `5/6/7/8`, and `selectionLog.stripReader`.
- `npm run benchmark:roi-diff` remains the standard checkpoint comparison workflow.
- The benchmark requires `backend/models/roi-rotaug-e30-640.pt`, `backend/models/roi.pt`, `backend/models/digit_classifier.pt`, and `backend/models/digit_strip_reader.pt` to exist locally before it will start.
- Keep `roi-rotaug-e30-640.pt` as default until a challenger improves end-to-end OCR, not just detection presence.

## Current Focus
1. Keep the `MAE 64.36` / `Exact Match 11/26` / `No-read 1/26` baseline as the primary-path promotion target.
2. Use strip-reader shadow logs to compare whole-strip predictions against the current per-cell classifier, especially on `meter_20260327.JPEG` and April captures.
3. Fix the remaining neural ROI miss on `meter_20201111.JPEG`.
4. Keep runtime/exporter comparison tooling available, but avoid broad promotion work unless live browser shadow results beat the primary path.
5. Medium-term: evaluate YOLO OBB ROI detection to reduce rotation and edge ambiguity.

## OBB Notes
- OBB inference outputs rotated geometry (`xywhr`) and polygon corners.
- OBB training labels use corners format: `class x1 y1 x2 y2 x3 y3 x4 y4`.
- Ultralytics OBB angle handling is constrained to the `0-90` exclusive range, so re-verify label/export assumptions before implementation.
