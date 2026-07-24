# OCR Tuning Playbook

This playbook documents the practical loop used to improve OCR quality in Jarvis.

Current baseline policy:

- Use the latest UI **Run test set** histogram as source of truth (`window.__jarvisLastTestSetHistogram`).
- Treat fixed numeric snapshots as historical only; they go stale quickly as thresholds/ranking change.
- Evaluation uses `MAE` as the primary promotion signal; `Exact Match` and `No-read` are guardrails.
- The active local test-set surface is always the live `assets/meter_readings.csv`; derive its changing row count from the file rather than hard-coding it in this playbook.
- The latest verified promoted ROI + restored promoted per-cell classifier benchmark is the 36-image run from July 23, 2026: `MAE 104.71`, `Exact Match 11/36`, `No-read 1/36`. A standardized same-run ROI diff measured the same promoted stack at `MAE 103.83`, so use paired runs when small runtime variance matters.
- Historical May 29, 2026 ranker-on/ranker-off control on the then-29-image corpus: ranker-on `MAE 166.07`, `Exact Match 10/29`, `No-read 1/29`; ranker-off `MAE 166.57`, `Exact Match 10/29`, `No-read 1/29`. The ranker remains enabled because it slightly improved `MAE` without worsening guardrails on that corpus.
- Re-run the UI `Run test set` before treating any metric as the current promotion target.
- `meter_20260112.JPEG`, `meter_20260113.jpg`, and `meter_20260219.JPEG` are intentionally removed from the active raw/test/training corpus because their visible readings are ambiguous.

Digit dataset status (current workflow):

- The new full-image object-detection dataset is built and reviewed separately;
  see `docs/full-image-digit-dataset.md`. Its persistent five-fold manifest
  splits train sources at image level; the historical one-image sanity holdout
  remains isolated from checkpoint selection but is not a promotion gate. Its
  derived YOLO labels are not a training source until all canonical
  annotations are reviewed. A frozen recipe requires a newly collected locked
  external full-image test set before promotion. Run
  `evaluate_full_image_digit_detector.py` on every CV checkpoint so
  complete-reading exact match and no-read remain gates alongside mAP.
  `backend/data/full_image_digit_dataset/manifests/source_exclusions.csv`
  defines retained legacy-stress sources outside this detector's active scope;
  `meter_20201111.JPEG` is excluded there for severe defocus but remains
  retained in the source and canonical annotation history.
- Dataset generation now uses `extract_digit_windows.py` -> `split_digit_windows.py` -> `label_digit_sections.py`.
- `split_digit_windows.py` canonicalizes orientation (major axis + optional reading-direction `flip180` overrides) before equispaced 4-way split.
- Small reviewed canonical-strip fixes live in `data/digit_dataset/manifests/canonical_overrides.csv`; regenerate sections, labels, synthetic train sections, and `qa:strip-dataset` after changing it.
- Classifier training consumes `data/digit_dataset/sections_labeled/{train,val,test}`.
- Whole-strip shadow-reader training consumes `data/digit_dataset/windows_canonical/{train,val,test}` and `canonical_windows.csv` readings. The constrained `23xx` reader uses the same canonical windows with a guard label derived from `reading[1] === "3"` and suffix labels from the final two digits.
- Synthetic generation remains train-only (`sections_synthetic/train`) and is mixed into training with `--synthetic-target-ratio`.
- As of May 27, 2026, the digit dataset intentionally has no validation split: `meter_20260323.JPEG` moved into train, while `meter_20260327.JPEG` remains a fixed hard test holdout. Use grouped source-image CV on the train pool for model experiments; the fixed test image is a historical diagnostic, not a promotion gate.

## Immediate Next Steps (July 23, 2026)

1. Keep the whole-strip reader shadow-only until its exact-match rate and `MAE` beat the current per-cell primary path.
2. Treat the July 23 canonical strip-window QA pass as accepted for the 36-image digit corpus. The seven new June/July sources are readable and label-consistent, including their transition-wheel states.
3. Do not promote the July four-head strip-reader fine-tune. Focused runtime QA remained `1/7` exact and the fixed hard holdout remained `0/1` exact.
4. Keep the restored promoted per-cell digit classifier as the safety baseline. The July clean + balanced-synthetic fine-tune improved grouped CV from `48.6%` to `67.9%`, then failed the 36-image UI gate at `MAE 3063.31`, `Exact Match 0/36`, `No-read 1/36`.
5. Keep the existing `23xx` checkpoint shadow-only and reject the July retrain. Its CV produced `0` guard false positives, `33` false negatives, and no accepted predictions; focused runtime QA also accepted none.
6. Keep the June 9 ROI checkpoint promoted. The July retrain reduced readable-row `MAE` from `103.83` to `75.50`, but exact match fell from `11/36` to `8/36` and no-read rose from `1/36` to `12/36`.
7. Verify each OCR tuning change on the full test set with `MAE` + guardrails (`Exact Match`, `No-read`) before keeping it.

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
- Compare `selectionLog.selected` against `selectionLog.stripReader` and `selectionLog.stripReader23xx` before considering any strip-reader promotion.

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
- `backend/models/digit_strip_reader_23xx.pt` for constrained-reader shadow diagnostics

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

## Focused QA Exporters

Use these scripts when the UI histogram points to candidate-selection, strip-reader, or cell-crop failures:

```bash
npm run qa:strip-dataset
npm run qa:ocr-oracle
npm run qa:strip-runtime
npm run qa:cell-crops
npm run qa:runtime-failure-dataset
npm run qa:runtime-failure-dataset:selected
npm run qa:digit-classifier-cv
```

They write timestamped reports under `output/strip-dataset-qa/`, `output/ocr-candidate-oracle/`, `output/strip-runtime-qa/`, `output/cell-crop-failure-qa/`, and `output/runtime-failure-dataset-qa/`. Run `qa:strip-dataset` after digit-window regeneration and visually accept the canonical strips before using them for retraining. Use `qa:runtime-failure-dataset` after exporting runtime failure cells so hard-example crops can be reviewed before any fine-tuning run. Use `qa:runtime-failure-dataset:selected` when the review should focus only on the UI-selected failure candidate for each image.

## Digit Classifier Recovery Notes

The promoted `backend/models/digit_classifier.pt` checkpoint currently contains behavior that was not recovered by training from the clean canonical section dataset alone. Treat it as the safety baseline and restore it from DVC before digit-classifier experiments:

```bash
env DVC_SITE_CACHE_DIR=/tmp/dvc-site-cache \
  backend/.venv/bin/python -m dvc checkout --force \
  backend/models/digit_classifier.pt.dvc \
  backend/models/digit_strip_reader.pt.dvc
```

Runtime-failure crops can be reconstructed with:

```bash
cd backend
source .venv/bin/activate
python export_runtime_digit_failure_set.py
cd ..
npm run qa:runtime-failure-dataset
npm run qa:digit-classifier-cv
```

May 24, 2026 evidence: grouped source-image CV suggested runtime-failure cells are a useful ingredient (`51.8%` restored baseline, `61.6%` clean fine-tune, `66.1%` clean + curated-runtime-failure fine-tune), but the then-28-image full UI benchmark rejected the resulting challenger (`MAE 139.11`, `Exact Match 3/28`, `No-read 1/28`). It fixed some targeted rows, including `meter_20260413.JPEG`, but damaged many rows the restored checkpoint already handled. Do not promote any digit classifier trained only from `sections_labeled` or from a CV-positive recipe unless it beats the restored checkpoint on the same UI **Run test set**.

May 27, 2026 split policy: grouped cross-validation by source image is now the default model-experiment workflow. Fold assignment must group by original meter filename, not individual section/cell crops, so cells, canonical strips, synthetic variants, and runtime-failure crops derived from one photo cannot leak between train and evaluation folds. The default CV command uses only the train pool, because the single remaining `test` image is kept as a fixed historical hard-case diagnostic.

May 24, 2026 UI-selected runtime-failure review: the dominant failure mode is upstream strip/cell geometry, not pure digit classification. The tracked taxonomy lives in `docs/ocr-runtime-failure-selected-taxonomy.csv` and is rendered by `qa:runtime-failure-dataset`. Among the 15 UI-selected failure candidates, 13 are structurally invalid before classification: six have over-wide strips where cells 1 and 4 contain no digits while cells 2 and 3 contain two digits each (`meter_20200701.JPEG`, `meter_20260216.JPEG`, `meter_20260420.JPEG`, `meter_20260423.JPEG`, `meter_20260427.JPEG`, `meter_20260507.JPEG`), five are truncated on the right with two digits missing (`meter_20260401.JPEG`, `meter_20260409.JPEG`, `meter_20260512.JPEG`, `meter_20260515.JPEG`, `meter_20260518.JPEG`), and two are degenerate/rotated captures with one or zero usable digits (`meter_20260214.JPEG`, `meter_20260416.JPEG`). Only two reviewed failures looked like classifier or local split misses on otherwise usable strips (`meter_20260413.JPEG`, `meter_20260521.JPEG`). Prioritize candidate geometry/ranking fixes before spending more effort on digit-classifier fine-tuning.

May 29, 2026 selector follow-up: the conservative candidate geometry ranker is deliberately a ranking tie-breaker, not a hard gate. It measures per-cell texture to detect sparse edge cells plus crowded middle cells, then applies only a tiny score penalty when compatible same-prefix edge evidence exists. Stronger penalties and low-texture gates regressed earlier UI benchmarks, so keep the ranker conservative unless a future UI Run test set proves otherwise. On the then-current 29-image corpus, ranker-on measured `MAE 166.07`, `Exact Match 10/29`, `No-read 1/29`; ranker-off measured `MAE 166.57`, `Exact Match 10/29`, `No-read 1/29`.

## Checkpoint Promotion Gates

Promote a challenger checkpoint only if all gates pass on the same test-set run:

1. **No-detection gate**: challenger `no-detection` count must be less than or equal to baseline.
2. **MAE gate**: challenger `MAE` must be less than or equal to baseline.
3. **Exact-match guardrail**: challenger `Exact Match` rate must be greater than or equal to baseline.
4. **No-read guardrail**: challenger `No-read` rate must be less than or equal to baseline.
5. **Failure-bucket gate**: challenger must not regress dominant no-read bucket counts (for example `classifier-edge-gate-final-drop` or `ocr-no-digits`) versus baseline.

If any gate fails, keep `roi-rotaug-e30-640.pt` as default and continue tuning extraction/selection.

June 9, 2026 ROI promotion: the retrained `roi.pt` challenger was copied into the promoted `backend/models/roi-rotaug-e30-640.pt` default after `npm run benchmark:roi-diff` measured `MAE 106.83` versus the previous checkpoint's `388.00`, with exact match unchanged at `11/31` and no-read unchanged at `1/31`. A visual audit of changed rows showed the new detector generally produces readable register crops; remaining high-impact misses are primarily classifier/selection failures.

Classifier-default rule:

- Keep `digitClassifier.enabled=true` by default and tune ranking/acceptance using `MAE` + guardrails.

Strip-reader shadow rule:

- Keep `digitStripReader.shadowOnly=true` until the same UI run shows whole-strip shadow exact match and `MAE` outperform the primary selected values.
- Do not promote based on canonical-window train/val/test metrics alone; the browser candidate crops are the promotion surface.

Digit-classifier rule:

- Keep `backend/models/digit_classifier.pt` restored from DVC as the primary safety baseline.
- Train challengers into `backend/runs/...`, not `backend/models/`.
- A challenger must improve `MAE` and must not regress exact match or no-read count on the UI **Run test set** before promotion.
- Clean canonical section metrics and grouped CV are diagnostic only; browser runtime candidate crops are the promotion surface.
- The digit dataset no longer uses a one-image validation split. If `train_digit_classifier.py` sees no `val` samples, checkpoint selection falls back to train loss; use grouped CV plus the UI **Run test set** to judge the recipe.

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
- `backend/train_strip_digit_reader_23xx.py`
- `backend/strip_digit_reader.py`
- `backend/strip_digit_reader_23xx.py`
- `src/ocr/digit-classifier.js`
- `src/ocr/pipeline.js`

Focus:

- Compare `selectionLog.stripReader.value` and `selectionLog.stripReader23xx` to expected readings and selected classifier readings.
- Watch whether stage `8` receives a visually plausible full strip before blaming the model.
- Retrain after canonical windows change, then judge promotion only with the UI test set.
- The July 23, 2026 four-head fine-tune is also a rejected shadow challenger: focused runtime QA remained `1/7` exact. Keep the promoted shadow checkpoint as the comparison baseline.

House-specific `23xx` shortcut:

- The constrained strip-reader experiment hard-codes the first two digits as `23`, trains/predicts only the final two digit positions, and uses a dedicated second-digit-is-`3` guard before accepting any forced `23xx` value.
- This is a deliberate local shortcut for the current home water meter, not a general OCR assumption.
- Review the assumption at least yearly, immediately if readings approach `2390`, and before reusing Jarvis for another meter.
- The default guard threshold is `0.98`; false positives are the dangerous failure mode, so tune for near-zero false positives even if recall is poor.
- The promoted May 4 checkpoint persists the fixed prefix in config/checkpoint metadata and keeps the unconstrained four-head reader benchmark available. The July 23 retrain was worse (`33` CV false negatives and no accepted predictions), while lowering the guard threshold previously accepted more wrong values.

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

## July 23, 2026 Retraining Decision

- ROI report: `output/roi-checkpoint-diff/20260723-000715-neural-digit/roi-diff-report.md`; challenger rejected on exact-match and no-read guardrails.
- Digit grouped CV: `backend/runs/digit-classifier-cv-jul36/digit_classifier_finetune_cv_summary.json`; the CV-winning recipe failed the UI promotion gate.
- Strip QA: `output/strip-runtime-qa/20260723-001010/summary.json`; the unconstrained challenger stayed at `1/7` exact and the constrained challenger accepted none.
- Production checkpoints under `backend/models/` were left unchanged. The accepted 36-source digit manifests and DVC pointers were refreshed locally.

## Commit Checklist (OCR Changes)

1. `npm run test:e2e` passes.
2. UI test-set rerun completed.
3. `MAE` and guardrail deltas documented (`Exact Match`, `No-read`, and improved/regressed image counts).
4. Any tuning knob changes are explained in PR notes.
