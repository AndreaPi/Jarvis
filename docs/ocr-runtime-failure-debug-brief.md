# OCR Runtime Failure Debug Brief

## Purpose

Create a focused starting point for the next OCR runtime-failure debugging pass. The current evidence points to strip candidate geometry and cell splitting as the highest-leverage area, with classifier training treated as secondary unless a full UI benchmark proves otherwise.

## Current State

| Area | Evidence | Working read |
| --- | --- | --- |
| Active OCR path | Neural ROI is mandatory; digit-classifier inference is mandatory; whole-strip readers remain shadow-only. | Debugging should stay on ROI candidate selection, strip normalization, cell splits, and primary classifier inputs. |
| Benchmark baseline | May 29, 2026 UI test set: `MAE 166.07`, `Exact Match 10/29`, `No-read 1/29`; ranker-off control: `MAE 166.57`, `Exact Match 10/29`, `No-read 1/29`. | The conservative geometry ranker is only a small tie-breaker, not a broad rejection mechanism. |
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

Then run the UI `Run test set` with the debug overlay enabled. Promotion requires improved `MAE` without exact-match or no-read regressions against the active 29-image baseline.

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
