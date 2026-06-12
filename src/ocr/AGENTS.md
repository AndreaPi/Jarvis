# OCR Guidelines

## Scope
- This file covers the OCR pipeline in `src/ocr/`, including current behavior, benchmark baselines, and promotion policy.

## Current Working State
- Neural ROI is mandatory in the frontend OCR flow; heuristic ROI fallback has been removed.
- On neural ROI failure, the UI shows an explicit reason and asks for manual measurement input.
- Digit-classifier inference is mandatory in the frontend OCR flow (`OCR_CONFIG.digitClassifier.enabled` defaults to `true`).
- The whole-strip digit reader (`OCR_CONFIG.digitStripReader`) runs in shadow mode only. Its predictions are logged for comparison and debug, but they must not change user-visible OCR selection until a benchmark explicitly promotes it.
- The constrained house-specific `23xx` strip reader (`OCR_CONFIG.digitStripReader23xx`) also runs in shadow mode only. It logs accepted/abstained diagnostics under `selectionLog.stripReader23xx` and must not change user-visible OCR selection.
- Frontend OCR evaluation is strip-only, classifier-first candidate decoding. The old Tesseract word-pass and sparse-scan stages are not part of the active path.
- Edge-derived candidate generation is enabled by default and can be toggled with `OCR_CONFIG.roiDeterministic.useEdgeCandidates`.
- The primary classifier shortlist now mixes high-ranked edge and base strip candidates so valid full-strip rotations are not starved behind edge-only passes.
- Opposite-orientation retry is disabled by default (`roiDeterministic.tryOppositeOrientation=false`).
- The default ROI checkpoint is `backend/models/roi-rotaug-e30-640.pt`, refreshed on June 9, 2026 after the retrained ROI detector won the end-to-end OCR gate.

## Debug Overlay Semantics
- `6a. OCR input candidate (initial preview)` is the first valid ROI candidate before classifier ranking.
- `6. OCR input candidate` is the final winning decode input.
- `7. classifier cell crops` shows the four cell crops passed to the current primary per-cell digit classifier.
- `8. strip reader input` shows the best whole-strip shadow-reader input and compact prediction/confidence summary.
- Diff artifacts and benchmark reports should use the last `6. OCR input candidate` frame as the selected OCR input snapshot.

## Active Benchmark Baseline
- Current UI test-set surface is `assets/meter_readings.csv`, currently `33` images after the June 2026 ROI ingestion through `meter_20260612.JPEG`.
- Current promoted ROI + restored promoted per-cell classifier baseline, rechecked June 12, 2026:
  - UI production test-set run: `MAE 111.75`, `Exact Match 11/33`, `No-read 1/33`
  - New `meter_20260612.JPEG` row: expected `2339`, selected `2304` from `roi-90-base-roi` (absolute error `35`)
  - Previous 31-image ROI checkpoint diff surface before the June 10/12 ingestions: `MAE 106.83`, `Exact Match 11/31`, `No-read 1/31`
  - Previous ROI checkpoint on that 31-image run: `MAE 388.00`, `Exact Match 11/31`, `No-read 1/31`
  - `npm run test:e2e`: passes (`7/7`)
- Historical restored per-cell classifier baseline with the conservative geometry ranker, measured May 29, 2026:
  - UI test set: `MAE 166.07`, `Exact Match 10/29`, `No-read 1/29`
  - Same-corpus ranker-off control: `MAE 166.57`, `Exact Match 10/29`, `No-read 1/29`
  - `npm run test:e2e`: passes (`7/7`)
- Re-run the UI `Run test set` before treating any metric as the current promotion target.
- Use `MAE` as the primary promotion signal, with `Exact Match` and `No-read` as guardrails.

## OCR Workflow and Guardrails
- Before committing OCR changes, run both `npm run test:e2e` and the UI `Run test set`.
- Prefer running the test set with the debug overlay enabled.
- Test-set review should inspect `Detected`, `Absolute Error`, `Failure Reason`, stages `5/6/7/8`, `selectionLog.stripReader`, and `selectionLog.stripReader23xx`.
- `npm run benchmark:roi-diff` remains the standard checkpoint comparison workflow.
- The benchmark requires `backend/models/roi-rotaug-e30-640.pt`, `backend/models/roi.pt`, `backend/models/digit_classifier.pt`, and `backend/models/digit_strip_reader.pt` to exist locally before it will start.
- Keep a challenger out of the default path until it improves end-to-end OCR, not just detection presence.

## Current Focus
1. Keep the restored promoted `backend/models/digit_classifier.pt` as the primary safety baseline.
2. Treat the May 4, 2026 canonical-strip QA pass as accepted for the 23-image digit-training corpus; all retained strips are readable, with realistic crop-tightness variation.
3. Keep the retrained four-head strip reader shadow-only. Its May 4, 2026 focused runtime QA on `meter_20260327.JPEG` plus April captures reached only `1/7` exact, so it is not a promotion candidate.
4. Use strip-reader shadow logs to compare whole-strip predictions by source against the current per-cell classifier, especially when deciding whether a future constrained reader should use selected-source or confidence-best candidates.
5. Do not promote digit-classifier scratch retrains or clean-section-only fine-tunes unless they beat the restored checkpoint on the UI test set. A May 24, 2026 clean + curated-runtime-failure challenger improved grouped CV but failed the then-28-image UI gate (`MAE 139.11`, `Exact Match 3/28`, `No-read 1/28`).
6. Keep the constrained `23xx` reader shadow-only. The first May 4, 2026 checkpoint is diagnostic-only: cross-validation looked conservative (`0` guard false positives, `19` guard false negatives), but runtime QA still found accepted wrong predictions. Lowering the guard from `0.98` to `0.80` accepted more wrong values, so threshold tuning is not enough.
7. Use `npm run qa:cell-crops` to inspect candidate-family coverage before changing selection. The June 8, 2026 register-localization probe found expected readings only on already-covered rows, so it was rolled back; the retained report now includes non-readable candidates and expected-hit family counts.
8. Use `npm run qa:roi-geometry-audit` when expected readings remain absent from expanded candidates. The June 8, 2026 focused audit of the current `10` candidate-coverage rows split them into `7` crop-family boundary-clipped rows and `3` edge-window-present normalization-insufficient rows.
9. Do not reintroduce the June 9, 2026 `regwin` register-window crop family without a stronger guard. Its focused run produced near-miss values but no exact expected candidate recovery, and selectable experiments regressed to `MAE 1345.10` (`maxPrimaryCandidates=4`) and `MAE 1378.67` (`maxPrimaryCandidates=20`). The implementation was removed to avoid dead diagnostic code.
10. Keep `roiDeterministic.cellSplitProbe` shadow-only. The June 11, 2026 QA run found `splitProbeOnlyHitRowCount: 1` (`meter_20200701.JPEG`) and production UI metrics stayed `MAE 106.83`, `Exact Match 11/31`, `No-read 1/31`; this is diagnostic evidence for split placement, not a production promotion.
11. Medium-term: evaluate YOLO OBB ROI detection only if axis-aligned ROI retrains still leave rotation or edge ambiguity.

## Digit Classifier Training Guardrail
- Restore promoted checkpoints from DVC before digit experiments when local model outputs drift.
- Use `backend/export_runtime_digit_failure_set.py` to reconstruct train-only runtime failure crops, then `npm run qa:runtime-failure-dataset` for visual QA.
- Use `npm run qa:runtime-failure-dataset:selected` when the review should focus only on UI-selected failure candidates.
- Use `npm run qa:digit-classifier-cv` for grouped source-image CV on the train pool, but treat it as diagnostic only.
- May 27, 2026 split policy: the digit dataset has no validation split. `meter_20260323.JPEG` is train, while `meter_20260327.JPEG` remains a fixed hard test holdout. Grouped CV by source image is the default experiment evaluator, and sibling cells/crops from the same meter photo must never leak across folds.
- Write challengers under `backend/runs/...`; do not overwrite `backend/models/digit_classifier.pt` until the challenger improves UI `MAE` without worsening exact match or no-read count.
- May 24, 2026 runtime-failure review found most selected failures were upstream strip/cell geometry problems, not standalone digit-classifier misses. Prioritize candidate geometry/ranking fixes before more digit fine-tuning.
- May 29, 2026 conservative geometry ranker is a tiny tie-breaker, not a hard reject: it penalizes suspicious full-strip cell splits only when coherent same-prefix edge evidence already exists. Stronger geometry penalties regressed earlier UI benchmarks. On the then-current 29-image corpus, ranker-on measured `MAE 166.07`, `Exact Match 10/29`, `No-read 1/29`; ranker-off measured `MAE 166.57`, `Exact Match 10/29`, `No-read 1/29`, so the ranker stays enabled as a small same-corpus MAE improvement.
- May 24, 2026 curated-runtime fine-tune from the promoted checkpoint fixed some reviewed classifier/local-split misses (`meter_20260413.JPEG`) but broadly damaged previously-correct rows. Grouped CV improved, but the UI gate rejected the challenger, so keep runtime-failure fine-tuning as an experiment only until a full UI run improves `MAE` without guardrail regressions.

## House-Specific `23xx` Assumption
- The current meter is expected to stay in the `2300`-`2399` range for the useful life of this local project.
- The constrained reader may therefore emit `23` + two predicted suffix digits, reducing the learned task from four positions to two. It is gated by a dedicated second-digit-is-`3` guard and abstains unless the guard confidence is at least `0.98`.
- This assumption must be reviewed at least yearly, and immediately if readings approach `2390` or the system is reused for a different meter/home.
- The prefix is documented in config/model metadata; keep the unconstrained benchmark path available for comparison.
- Do not present the constrained reader as generally valid beyond this house-specific water-meter workflow.
