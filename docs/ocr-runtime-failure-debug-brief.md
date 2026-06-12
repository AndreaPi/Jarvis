# OCR Runtime Failure Debug Brief

## Purpose

Create a focused starting point for the next OCR runtime-failure debugging pass. The current evidence points to strip candidate geometry and cell splitting as the highest-leverage area, with classifier training treated as secondary unless a full UI benchmark proves otherwise.

## Current State

| Area | Evidence | Working read |
| --- | --- | --- |
| Active OCR path | Neural ROI is mandatory; digit-classifier inference is mandatory; whole-strip readers remain shadow-only. | Debugging should stay on ROI candidate selection, strip normalization, cell splits, and primary classifier inputs. |
| Benchmark baseline | Current promoted ROI + restored per-cell classifier surface, rechecked June 11, 2026: `MAE 106.83`, `Exact Match 11/31`, `No-read 1/31`. The previous ROI checkpoint on the same 31-image surface was `MAE 388.00`, `Exact Match 11/31`, `No-read 1/31`. | The June 9 ROI checkpoint promotion remains valid; remaining high-impact failures are mostly crop/split/selection problems, not ROI detection misses. |
| Selected failure taxonomy | `right_truncated`: 5, `overwide_split`: 4, `classifier_or_local_split`: 2, plus one each of `overwide_rotated_split`, `overwide_blurry_split`, `degenerate_rotated`, and `degenerate_no_digits`. | The dominant failures are upstream geometry/cropping issues, not isolated digit-classifier misses. |
| Debug summaries | `output/debug-stage-inspection/summary.json` covers 7 rows; selected sources are `roi-90-base-roi` for 6 rows and `roi-90-edge-roi` for 1 row. Reject summaries include `classifier-edge-candidate-selected`: 14 and `classifier-missing-cell-digit`: 1. | Candidate selection is producing rejected edge alternatives and mostly selecting base ROI variants in the inspected subset. |
| Guardrails | `src/ocr/AGENTS.md` says to run both `npm run test:e2e` and the UI `Run test set` before treating OCR changes as promotable. | Any code change should be judged by UI-set `MAE`, with exact match and no-read as guardrails. |

## Likely Root Causes

1. Right-truncated runtime strips are the biggest visible bucket.
   Files in the selected taxonomy include `meter_20260401.JPEG`, `meter_20260409.JPEG`, `meter_20260512.JPEG`, `meter_20260515.JPEG`, and `meter_20260518.JPEG`.

2. Overwide splits are nearly as frequent.
   Several notes say cells 1 and 4 contain no digits while cells 2 and 3 contain two digits each, which suggests the chosen strip/cell geometry is too wide or badly aligned before four-way splitting.

3. Rotation and degenerate crops still need hard guardrails.
   `degenerate_rotated` and `degenerate_no_digits` rows show that some candidate crops are structurally invalid enough that classifier confidence should not be the primary arbiter.

4. Classifier-only work is a narrower path.
   The taxonomy has only two `classifier_or_local_split` rows. Prior guidance says the May 24 curated-runtime fine-tune helped selected misses but damaged broader UI performance, so retraining should wait until geometry/ranker fixes are exhausted or benchmark evidence changes.

## Immediate Next Moves

1. Regenerate focused QA artifacts.

```bash
npm run qa:strip-runtime
npm run qa:cell-crops
```

2. Inspect the dominant buckets first.

Use `docs/ocr-runtime-failure-selected-taxonomy.csv` to group examples by `right_truncated` and `overwide_split`, then compare the selected `6. OCR input candidate` and `7. classifier cell crops` frames for those files.

3. Trace the selection path in code.

Start with:

- `src/ocr/recognition.js` for candidate scoring, classifier probing, missing-cell handling, and selected-reading metadata.
- `src/ocr/alignment.js` for neural ROI candidate construction and debug stage generation.
- `src/ocr/config.js` for edge-candidate, overlap, geometry-ranker, and shadow-reader configuration.

4. Prototype geometry checks before model changes.

Candidate checks worth testing:

- reject or heavily penalize strips whose four cell crops leave cells 1 and 4 empty while middle cells contain doubled digits
- penalize right-truncated strips when the rightmost cell has weak foreground evidence and competing variants keep all four digits
- detect degenerate rotated/no-digit crops before classifier probing is allowed to select them

5. Validate with the established gates.

```bash
npm run test:e2e
```

Then run the UI `Run test set` with the debug overlay enabled. Promotion requires improved `MAE` without exact-match or no-read regressions against the active 31-image baseline.

## Open Questions

| Question | Why it matters |
| --- | --- |
| Are the latest QA outputs newer than the May 2026 guidance? | The brief uses existing checked-in summaries; rerunning QA may reveal drift. |
| Which files are currently failing on the 29-image UI set? | The selected taxonomy has 15 rows and may not cover every current failure. |
| Do edge candidates consistently preserve full digit coverage on right-truncated rows? | If yes, the next fix may be ranker arbitration rather than ROI crop generation. |

## Decision

Prioritize runtime strip geometry and candidate-ranking diagnostics. Keep the promoted `backend/models/digit_classifier.pt` as the baseline and keep both strip readers shadow-only until a full UI benchmark beats the current primary path.

## Execution Notes - 2026-06-06

The immediate next moves above were run against the local app and backend:

- `npm run qa:strip-runtime` produced `output/strip-runtime-qa/20260606-001708/` before the prototype and `output/strip-runtime-qa/20260606-002237/` after it.
- `npm run qa:cell-crops` produced `output/cell-crop-failure-qa/20260606-001942/` before the prototype and `output/cell-crop-failure-qa/20260606-002253/` after it.
- The prototype added severe edge-underfill and low-texture settings to the existing geometry ranker, plus a new `right-edge-severe-underfilled` diagnostic reason.
- Focused selected readings in the strip-runtime report did not change, and the cell-crop failure row count stayed at `13`. The prototype is therefore diagnostic/conservative, not a promoted accuracy improvement.
- `npm run test:e2e` passed (`7/7`).
- UI `Run test set` on the current `30` rows reported `MAE 161.48`, `Exact Match 10/30`, and `No-read 1/30`. The one row error was `meter_20201111.JPEG`, where neural ROI returned `no-detection`.

Next debugging should look beyond only increasing geometry penalties. Several dominant-bucket rows still lack a correct candidate even in the expanded cell-crop oracle, so the next useful implementation path is better candidate generation or ROI/crop normalization for full digit coverage.

## Execution Notes - 2026-06-07

A candidate-coverage prototype was attempted with shifted tight-register crops and guarded tight-register primary selection. The production behavior change was rejected:

- Pre-change `npm run qa:cell-crops` produced `output/cell-crop-failure-qa/20260607-222551/` with `13` expected-absent rows under the old report shape.
- The permissive shifted-register selection run produced `output/cell-crop-failure-qa/20260607-223214/` with `rowCount: 23`, `expectedAbsentRowCount: 13`, and visibly worse selected readings.
- The stricter suspicious-full-strip-only selection run produced `output/cell-crop-failure-qa/20260607-223335/` with `rowCount: 20`, `expectedAbsentRowCount: 13`, and no coverage improvement.
- The production shifted-crop/selection code was therefore rolled back. The retained improvement is QA reporting: `npm run qa:cell-crops` now separates expected-absent rows from expected-present-but-not-selected rows.

Next debugging should not spend more time on tight-register selection alone. The absent-candidate count staying at `13` points back to upstream candidate construction, ROI crop normalization, or edge/base arbitration.

## Execution Notes - 2026-06-08

The next pass started upstream in ROI candidate construction rather than retraining:

- Pre-change `npm run qa:cell-crops` on the current local `31`-row corpus produced `output/cell-crop-failure-qa/20260608-000736/` with `rowCount: 20`, `expectedAbsentRowCount: 14`, `expectedPresentRowCount: 6`, and coverage counts of `14` absent, `5` internal-variant present, and `1` direct-candidate present.
- A first padded edge-context prototype used horizontal shifts through `+0.14`. It improved coverage to `expectedAbsentRowCount: 11` in `output/cell-crop-failure-qa/20260608-000853/`, but the far-right shift also produced wrong primary selections on some rows.
- The retained upstream experiment narrows edge-context shifts to `[-0.08, 0, 0.08]` with `edgeContextMaxVariantsPerAngle: 3`. It preserves exact edge and base candidates while adding bounded, clamped, deduped padded edge-context crops in `src/ocr/alignment.js`.
- The retained run `output/cell-crop-failure-qa/20260608-001015/` produced `rowCount: 19`, `expectedAbsentRowCount: 13`, `expectedPresentRowCount: 6`, and coverage counts of `13` absent, `3` internal-variant present, and `3` direct-candidate present.
- UI `Run test set` on the same current `31` rows reported `MAE 388.00`, `Exact Match 11/31`, and `No-read 1/31`. From the pre-change cell-crop selected values, the same local surface was effectively `MAE 388.30`, `Exact Match 10/31`, and `No-read 1/31`, so this is a narrow promotable improvement on the local corpus.
- `npm run qa:strip-runtime` produced `output/strip-runtime-qa/20260608-001339/`; focused primary values remained wrong on the inspected hard rows, so strip readers remain shadow-only.
- `npm run test:e2e` passed (`7/7`).

Next debugging should continue upstream, but now with arbitration in mind: the new edge-context candidates improved direct candidate coverage for some rows, yet several correct candidates still lose to nearby wrong base or edge-context reads.

## Execution Notes - 2026-06-08 Arbitration Follow-up

The immediate arbitration follow-up improved diagnostics but did not produce a safe production behavior change:

- The cell-crop QA report now records selected-vs-expected comparison fields: selected source, best expected source, and score delta when the expected reading is present in expanded candidates.
- A guarded edge-context primary-slot and edge-over-base arbitration experiment was tested and rejected. `npm run qa:cell-crops` produced `output/cell-crop-failure-qa/20260608-075710/` with unchanged aggregate coverage (`rowCount: 19`, `expectedAbsentRowCount: 13`, `expectedPresentRowCount: 6`) but introduced a new selected-reading regression on `meter_20260310.JPEG` and worsened `meter_20260327.JPEG`.
- The production arbitration and primary-slot changes were rolled back. The retained diagnostic-only run is `output/cell-crop-failure-qa/20260608-080023/`, again with `rowCount: 19`, `expectedAbsentRowCount: 13`, `expectedPresentRowCount: 6`, and coverage counts of `13` absent, `3` internal-variant present, and `3` direct-candidate present.

Next debugging should use the new score deltas to separate two cases before changing selection: rows where the expected reading is absent from all candidates, and rows where the expected reading is present but loses by a small, explainable margin.

## Execution Notes - 2026-06-08 Shadow Normalization Probe

The upstream follow-up implemented a shadow-only crop-normalization probe:

- `src/ocr/alignment.js` now emits bounded target-aspect normalization candidates from edge crops, tagged as diagnostic by default through `roiDeterministic.normalizationProbe.shadowOnly`.
- `src/ocr/pipeline.js` filters diagnostic candidates out of production selection. They are decoded only when `digitClassifier.decodeDiagnosticCandidates` is enabled, and that pass cannot update `bestResult` or production evidence.
- `npm run qa:cell-crops` produced `output/cell-crop-failure-qa/20260608-115615/`. With diagnostic decoding enabled in the oracle pass, `expectedAbsentRowCount` fell from `13` to `10`; `shadowProbeHitRowCount` was `4`.
- The QA split now reports `candidate-coverage`, `shadow-coverage-gain`, and `selection-arbitration`. The retained run split was `10` candidate-coverage rows, `4` shadow-coverage-gain rows, and `5` selection-arbitration rows.
- A production-path UI test-set run on the current `31` rows stayed at `MAE 388.00`, `Exact Match 11/31`, and `No-read 1/31`, because diagnostic candidates remain shadow-only by default.

This made one narrow normalization family, the repeated winning probe shape `a24-h116-center`, the next guarded promotion experiment to try.

## Execution Notes - 2026-06-08 Guarded Promotion Experiment

The guarded `a24-h116-center` normalization promotion experiment was implemented and rejected:

- `npm run qa:cell-crops` produced `output/cell-crop-failure-qa/20260608-121226/` with `rowCount: 19`, `expectedAbsentRowCount: 10`, `expectedPresentRowCount: 9`, and split counts of `10` candidate-coverage rows, `2` shadow-coverage-gain rows, and `7` selection-arbitration rows.
- The experiment made normalized probe crops selectable only under a `23xx`/confidence/score-gap guard, but it still introduced selected-reading regressions. The clearest new regression was `meter_20260130.JPEG`, expected `2307` but selected `2309` from `roi-90-normprobe-a24-h116-center-roi`.
- Other hard rows were pulled toward wrong normprobe selections, so the production promotion hook was rolled back.
- The retained code keeps normalization probes shadow-only and keeps the QA/reporting improvements that distinguish candidate absence, shadow-probe coverage gain, and selection arbitration.

Next debugging should treat `a24-h116-center` as useful diagnostic evidence, not a production candidate family. Focus on why it helps coverage on a few rows while producing unsafe high-confidence wrong selections when promoted.

## Execution Notes - 2026-06-08 Register-Localization Probe

A diagnostic register-localization crop-construction experiment was implemented and rejected as a candidate-generation change:

- The experiment added measured ink-span `regloc` crops as diagnostic-only candidates, then `npm run qa:cell-crops` produced `output/cell-crop-failure-qa/20260608-124444/`.
- Aggregate coverage did not improve: `rowCount: 19`, `expectedAbsentRowCount: 10`, `expectedPresentRowCount: 9`, and split counts remained `10` candidate-coverage rows, `4` shadow-coverage-gain rows, and `5` selection-arbitration rows.
- `regloc` did find expected readings on `5` rows, but only on rows that were already covered by existing edge, edge-context, base, or normprobe families. No current candidate-coverage row gained the expected reading.
- The production candidate generator was rolled back to avoid extra diagnostic cost. The retained improvement is QA reporting: the cell-crop report now includes non-readable candidate trace rows, source-family counts, expected-hit family counts, and `registerLocalizationHitRowCount` for future experiments.

Next debugging should use the improved QA family counts before adding another crop family. The remaining `10` candidate-coverage rows still require a different upstream approach, likely earlier ROI geometry or a crop construction method that can recover expected readings on absent rows rather than adding redundant hits on already-covered rows.

## Execution Notes - 2026-06-08 ROI Geometry Audit

The focused upstream audit was implemented as a diagnostic-only workflow:

- `src/ocr/alignment.js` now tags emitted ROI candidates with trace-only geometry metadata: candidate family, angle, rotated ROI size, crop rect, edge rect, crop aspect, area ratio, and crop-frame ratios.
- `src/ocr/pipeline.js` now records neural ROI detector geometry and candidate geometry in `selectionLog`; this does not change candidate ranking, selection, classifier scoring, or strip-reader behavior.
- `npm run qa:roi-geometry-audit` writes `output/roi-geometry-audit/<timestamp>/summary.json` and `roi-geometry-audit.html`, filtering to rows where production selected the wrong reading and the expected reading is absent from expanded readable candidates.
- Latest run: `output/roi-geometry-audit/20260608-155031/` with `rowCount: 10`. The focused split is `7` `crop-family-boundary-clipped` rows and `3` `edge-window-present-normalization-insufficient` rows.

Next debugging should target the `crop-family-boundary-clipped` group first. These rows suggest the edge-derived crop families are frequently pushed against a rotated ROI boundary before they ever expose the correct four-digit candidate, so a better next experiment is bounded ROI/edge expansion or earlier ROI padding/orientation handling, not another selection promotion.

## Execution Notes - 2026-06-09 Rejected Regwin Diagnostic

A focused upstream register-window diagnostic was temporarily implemented for the two new iPhone captures:

- The temporary `regwin` candidate family emitted sliding and ink-anchored windows around the detected edge search area.
- The temporary focused audit wrote original and rotated ROI overlay images plus a focused summary under `output/focused-roi-register-audit/<timestamp>/`.
- Latest run: `output/focused-roi-register-audit/20260609-003251/` with `rowCount: 2`, `expectedPresentRowCount: 0`, and `regwinHitRowCount: 0`.
- The new windows reached the register region and produced near-miss readings (`2336`/`2305` around expected `2335`, and `2307`/`2339` around expected `2337`), but still did not recover exact expected values.

Treat this as diagnostic evidence, not a promotion candidate. The executable `regwin` implementation and focused audit command were removed after the failed promotion test to avoid carrying dead diagnostic code.

## Execution Notes - 2026-06-09 Guarded Regwin Promotion Experiment

The focused `regwin` near-misses were tested as selectable candidates through a temporary runtime override, without changing default config:

- Baseline production run on the current `31` rows stayed at `MAE 388.00`, `Exact Match 11/31`, and `No-read 1/31`.
- `regwin` selectable with the normal `maxPrimaryCandidates=4` failed the guardrails: `MAE 1345.10`, `Exact Match 9/31`, `No-read 1/31`, with `regwin` selected on `11` rows.
- `regwin` selectable with a wide `maxPrimaryCandidates=20` also failed: `MAE 1378.67`, `Exact Match 6/31`, `No-read 1/31`, with `regwin` selected on `16` rows.
- The two June iPhone rows improved only superficially in some cases but remained wrong: `meter_20260524.JPEG` selected `1231`/`1240` instead of `2335`, and `meter_20260606.JPEG` selected `1222`/`2007` instead of `2337`.

Conclusion: do not keep the `regwin` candidate family in production code. The useful signal was diagnostic crop proximity, not production selection. Next work should be a guard or geometry-quality discriminator that can reject high-confidence distractor crops before any similar future promotion attempt.

## Execution Notes - 2026-06-09 ROI Retrain Promotion

The ROI detector was retrained with the existing heavy-augmentation and rotation-expansion policy on the current 31-image ROI corpus, using a temporary training YAML that maps YOLO validation to the fixed test image because the canonical validation split is intentionally empty. The validation metric is diagnostic only; promotion was judged on browser OCR.

- `npm run benchmark:roi-diff` produced `output/roi-checkpoint-diff/20260609-113924-neural-digit/`.
- The challenger improved primary-path `MAE` from `388.00` to `106.83`, while exact match stayed `11/31` and no-read stayed `1/31`.
- The largest improvements were the two recent iPhone captures: `meter_20260524.JPEG` improved from `5332` to `2305`, and `meter_20260606.JPEG` improved from `9302` to `2007`.
- The main regression was `meter_20260603.JPEG`, from `2317` to `3332`, but visual audit showed the new detector crop still contains the expected register region; this remains a classifier/selection issue rather than a detector rejection.
- The retrained checkpoint was promoted by refreshing `backend/models/roi-rotaug-e30-640.pt`; `npm run test:e2e` passed (`7/7`).

## Execution Notes - 2026-06-11 Cell Split-Offset Probe

The reduced crop-inspection pass showed that some readable strips contain the expected digits visually, but the equal-width four-cell split cuts a digit boundary badly enough that no exact decoded candidate is produced. A diagnostic-only split-offset probe was added to measure that path without changing production selection:

- `src/ocr/recognition.js` now tries configured non-zero cell-split offsets only when `digitClassifier.decodeDiagnosticCandidates=true` and `enableCellSplitProbe=true`. Non-zero split offsets are limited to full-strip variants and remain ineligible for primary selection while `roiDeterministic.cellSplitProbe.shadowOnly` is true.
- `scripts/export-cell-crop-failure-qa.cjs` now reports split mode/offset metadata, supports `CELL_CROP_QA_FILES=...` for focused runs, and summarizes `splitProbeHitRowCount` plus `splitProbeOnlyHitRowCount`.
- Focused priority QA on the 10 reduced-review rows produced `output/cell-crop-failure-qa/20260611-235534/`: `expectedAbsentRowCount: 3`, `splitProbeHitRowCount: 3`, and `splitProbeOnlyHitRowCount: 1`. The one new exact decode was `meter_20200701.JPEG`, expected `1784`, recovered as an internal `scan-roi` full-strip split with `offset-left8`.
- Full QA produced `output/cell-crop-failure-qa/20260611-235607/`: `rowCount: 19`, `expectedAbsentRowCount: 9`, `expectedPresentRowCount: 10`, `splitProbeHitRowCount: 6`, and `splitProbeOnlyHitRowCount: 1`.
- Production behavior stayed unchanged. The UI production test-set run on the current 31 images stayed at `MAE 106.83`, `Exact Match 11/31`, and `No-read 1/31`; `npm run test:e2e` passed (`7/7`).

Conclusion: keep the split-offset probe as QA evidence, not as a production selector. It proves split placement can recover at least one absent expected reading, but it does not rescue the dominant remaining crop-coverage rows (`meter_20260327.JPEG`, `meter_20260603.JPEG`, `meter_20260606.JPEG`) and therefore should not displace upstream crop-normalization or selection-arbitration work.
