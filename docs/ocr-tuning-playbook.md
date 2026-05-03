# OCR Tuning Playbook

This playbook documents the practical loop used to improve OCR quality in Jarvis.

Current baseline policy:

- Use the latest UI **Run test set** histogram as source of truth (`window.__jarvisLastTestSetHistogram`).
- Treat fixed numeric snapshots as historical only; they go stale quickly as thresholds/ranking change.
- Evaluation uses `MAE` as the primary promotion signal; `Exact Match` and `No-read` are guardrails.
- The active local test-set CSV has `25` images.
- Current primary-path baseline: `MAE 65.79`, `Exact Match 11/25`, `No-read 1/25`.
- `meter_20260219.JPEG` is intentionally removed from the active raw/test/training corpus because the visible reading is ambiguous and the canonical strip cuts through the final digit.

Digit dataset status (current workflow):

- Dataset generation now uses `extract_digit_windows.py` -> `split_digit_windows.py` -> `label_digit_sections.py`.
- `split_digit_windows.py` canonicalizes orientation (major axis + optional reading-direction `flip180` overrides) before equispaced 4-way split.
- Classifier training consumes `data/digit_dataset/sections_labeled/{train,val,test}`.
- Whole-strip shadow-reader training consumes `data/digit_dataset/windows_canonical/{train,val,test}` and `canonical_windows.csv` readings.
- Synthetic generation remains train-only (`sections_synthetic/train`) and is mixed into training with `--synthetic-target-ratio`.

## Immediate Next Steps (May 1, 2026)

1. Keep the whole-strip reader shadow-only until its exact-match rate and `MAE` beat the current per-cell primary path.
2. Inspect strip-reader shadow predictions for `meter_20260327.JPEG` and the April captures; these are the current high-value mismatch probes.
3. Fix the remaining neural ROI miss on `meter_20201111.JPEG`.
4. Validate the canonical strip-window dataset visually before retraining; with only 25 strip samples, one bad crop or orientation can dominate the model.
5. Evaluate a house-specific `23xx` constrained strip-reader variant that hard-codes prefix `23` and predicts only the final two digits.
6. Verify each OCR tuning change on the full test set with `MAE` + guardrails (`Exact Match`, `No-read`) before keeping it.

## Goals

1. Reduce `mismatch` (wrong 4-digit value returned).
2. Reduce no-read outcomes (`ocr-no-digits`, `classifier-edge-gate-final-drop`, and related final drops).
3. Preserve neural-ROI-only policy and strip-only OCR path.

## Standard Iteration Loop

1. Run baseline checks

```bash
npm run test:e2e
```

- In UI (`http://localhost:8000`): run **Run test set** with debug overlay enabled.
- Record failure histogram and reject histogram.

2. Inspect hard failures

- Prioritize the current top `Absolute Error` rows and dominant `Failure Reason` buckets from the latest run.
- Inspect debug stages:
  - `0. neural roi detection`
  - `0b. neural roi crop`
  - `5. detected strip crop`
  - `6a. OCR input candidate (initial preview)`
  - `6. OCR input candidate` (winning decode input)
  - `7. classifier cell crops`
  - `8. strip reader input`
- Inspect selection logs in `window.__jarvisOcrSelectionLogs`.
- Compare `selectionLog.selected` against `selectionLog.stripReader` before considering any strip-reader promotion.

3. Apply one narrow change

- Make a single hypothesis-driven change (config or scoring/candidate logic).
- Avoid bundled edits that make regression attribution unclear.

4. Re-run and compare

- Re-run UI test set.
- Compare movement in:
  - `mismatch`
  - `ocr-no-digits`
  - ROI no-detection
- Re-run:

```bash
npm run test:e2e
```

5. Keep or revert

- Keep only changes with clear net improvement.
- Revert changes that shift failures without improving `MAE`.

## Automated Checkpoint Diff

Use the scripted checkpoint comparison to produce a per-image report between the pinned baseline and a challenger model:

```bash
npm run benchmark:roi-diff
```

The benchmark always runs neural-digit-only decode with the per-cell classifier enabled; the strip reader may run in shadow when its checkpoint/backend endpoint are available.
It requires these local checkpoints before starting:

- `backend/models/roi-rotaug-e30-640.pt`
- `backend/models/roi.pt`
- `backend/models/digit_classifier.pt`
- `backend/models/digit_strip_reader.pt`

Artifacts are saved to:

- `output/roi-checkpoint-diff/<timestamp>/roi-diff-report.md`
- `output/roi-checkpoint-diff/<timestamp>/roi-diff-report.json`
- `output/roi-checkpoint-diff/<timestamp>/{baseline,challenger}/stages/*` (stage `5` and `6` snapshots)

The report includes:

- Per-image `Detected`, `Failure Reason`, and top reject reason.
- Per-image selected metadata (`sourceLabel`, `method`, `preprocessMode`) from `window.__jarvisOcrSelectionLogs`.
- Side-by-side stage `5. detected strip crop` and `6. OCR input candidate`.
- Stage `6` now shows the exact strip variant used by the winning decode (after normalization/orientation selection).
- Stage `6` export uses the last `6. OCR input candidate` frame from each debug session.
- Stage `7` shows the four cell crops used by the current primary classifier.
- Stage `8` shows the best whole-strip shadow-reader input and prediction/confidence summary.
- Summary deltas for `MAE`, guardrail rates (`Exact Match`, `No-read`), and dominant failure buckets (`mismatch`, `classifier-edge-gate-final-drop`, `ocr-no-digits`, `no-detection`).

## Checkpoint Promotion Gates

Promote a challenger checkpoint only if all gates pass on the same test-set run:

1. **No-detection gate**: challenger `no-detection` count must be less than or equal to baseline.
2. **MAE gate**: challenger `MAE` must be less than or equal to baseline.
3. **Exact-match guardrail**: challenger `Exact Match` rate must be greater than or equal to baseline.
4. **No-read guardrail**: challenger `No-read` rate must be less than or equal to baseline.
5. **Failure-bucket gate**: challenger must not regress dominant no-read bucket counts (for example `classifier-edge-gate-final-drop` or `ocr-no-digits`) versus baseline.

If any gate fails, keep `roi-rotaug-e30-640.pt` as default and continue tuning extraction/selection.

Classifier-default rule:

- Keep `digitClassifier.enabled=true` by default and tune ranking/acceptance using `MAE` + guardrails.

Strip-reader shadow rule:

- Keep `digitStripReader.shadowOnly=true` until the same UI run shows whole-strip shadow exact match and `MAE` outperform the primary selected values.
- Do not promote based on canonical-window train/val/test metrics alone; the browser candidate crops are the promotion surface.

## High-Impact Tuning Areas

### 1) Candidate Generation (`ocr-no-digits`)

File: `src/ocr/alignment.js`

Focus:

- Rotation variant quality
- Edge-window extraction stability
- Normalization width and crop quality for strip readability

Signal to watch:

- Empty `topCandidates` in selection logs
- Debug stage `6` visually clear but still no accepted candidate

### 2) Classifier Candidate Ranking (`mismatch` vs `ocr-no-digits`)

Files:

- `src/ocr/pipeline.js` (candidate ranking + early stop)
- `src/ocr/config.js` (`digitClassifier.maxPrimaryCandidates`, edge safeguards)

Use temporary ranking/threshold experiments first, then codify only if net-positive.

### 3) Whole-Strip Shadow Reader

Files:

- `backend/train_strip_digit_reader.py`
- `backend/strip_digit_reader.py`
- `src/ocr/digit-classifier.js`
- `src/ocr/pipeline.js`

Focus:

- Compare `selectionLog.stripReader.value` to expected readings and selected classifier readings.
- Watch whether stage `8` receives a visually plausible full strip before blaming the model.
- Retrain after canonical windows change, then judge promotion only with the UI test set.

House-specific `23xx` shortcut:

- A constrained strip-reader experiment may hard-code the first two digits as `23` and train/predict only the final two digit positions.
- This is a deliberate local shortcut for the current home water meter, not a general OCR assumption.
- Review the assumption at least yearly, immediately if readings approach `2390`, and before reusing Jarvis for another meter.
- If implemented, persist the fixed prefix in config/checkpoint metadata and keep the unconstrained four-head reader benchmark available.

### 4) Acceptance/Support Guardrails

Files:

- `src/ocr/pipeline.js` (`finalizeSelection`, evidence ranking)
- `src/ocr/recognition.js` (candidate scoring)

Focus:

- Balance strictness (avoid false positives) vs recall (avoid no-read).
- Validate with histogram movement, not single-image anecdotes.
- Active guardrails in current pipeline: evidence ranking, mixed primary evaluation of top edge and base strip candidates, narrow `scan-roi` / base fallback only when base candidates were not already evaluated and edge support is still weak or edge-only, and final edge-confidence checks.

### 5) ROI Sanity Gates (usually not primary blocker)

Files:

- `src/ocr/neural-roi.js`
- `src/ocr/config.js` (`neuralRoi.sanity`)

Use only if evidence shows valid ROI boxes are being rejected.
Recent test-set verification showed no `invalid-geometry` failures.

## Useful Runtime Artifacts

- Browser selection logs:
  - `window.__jarvisOcrSelectionLogs`
- Last run histogram:
  - `window.__jarvisLastTestSetHistogram`

## Commit Checklist (OCR Changes)

1. `npm run test:e2e` passes.
2. UI test-set rerun completed.
3. `MAE` and guardrail deltas documented (`Exact Match`, `No-read`, and improved/regressed image counts).
4. Any tuning knob changes are explained in PR notes.
