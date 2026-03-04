# OCR Tuning Playbook

This playbook documents the practical loop used to improve OCR quality in Jarvis.

Current baseline notes (March 4, 2026, neural-digit-only):

- Test set: `0/14` exact-match (`Correct`)
- Pinned model (`roi-rotaug-e30-640.pt`): `mismatch` 6, `ocr-no-digits` 7, `no-detection` 1
- Challenger (`roi.pt`): `mismatch` 4, `ocr-no-digits` 10, `no-detection` 0
- Evaluation now uses `MAE` as the primary promotion signal; exact-match and no-read rates are guardrails.
- The active local test-set CSV now has `15` images; keep historical `0/14` snapshots only for trend context.

Critical blocker note (March 4, 2026):

- Digit-classifier training cells are currently unreliable due to strip orientation/cell-split issues in dataset export.
- `build_digit_dataset.py` splits cells assuming left-to-right horizontal strips; vertical strips become thin non-digit slices.
- Some horizontal strips are 180-deg inverted, so cell index to reading-digit assignment is reversed.
- Concrete examples from manual QA:
  - `meter_07012020_c3_4.png` appears as digit `1` in context.
  - `meter_01122026_c2_0.png` is a thin register slice, not a usable digit crop.

## Goals

1. Reduce `mismatch` (wrong 4-digit value returned).
2. Reduce `ocr-no-digits` (no accepted 4-digit candidate).
3. Preserve neural-ROI-only policy and strip-only OCR path.

## Standard Iteration Loop

1. Run baseline checks

```bash
npm run test:e2e
```

- In UI (`http://localhost:8000`): run **Run test set** with debug overlay enabled.
- Record failure histogram and reject histogram.

2. Inspect hard failures

- Prioritize:
  - `meter_02202026.JPEG`
  - `meter_02192026.JPEG`
  - `meter_07012020.JPEG`
  - `meter_02242026.JPEG`
- Inspect debug stages:
  - `0. neural roi detection`
  - `0b. neural roi crop`
  - `5. detected strip crop`
  - `6. OCR input candidate`
- Inspect selection logs in `window.__jarvisOcrSelectionLogs`.

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

The benchmark always runs neural-digit-only decode (classifier enabled).

Artifacts are saved to:

- `output/roi-checkpoint-diff/<timestamp>/roi-diff-report.md`
- `output/roi-checkpoint-diff/<timestamp>/roi-diff-report.json`
- `output/roi-checkpoint-diff/<timestamp>/{baseline,challenger}/stages/*` (stage `5` and `6` snapshots)

The report includes:

- Per-image `Detected`, `Failure Reason`, and top reject reason.
- Per-image selected metadata (`sourceLabel`, `method`, `preprocessMode`) from `window.__jarvisOcrSelectionLogs`.
- Side-by-side stage `5. detected strip crop` and `6. OCR input candidate`.
- Stage `6` export uses the last `6. OCR input candidate` frame from each debug session.
- Summary deltas for `MAE`, guardrail rates (`Exact Match`, `No-read`), `mismatch`, `ocr-no-digits`, and `no-detection`.

## Checkpoint Promotion Gates

Promote a challenger checkpoint only if all gates pass on the same test-set run:

1. **No-detection gate**: challenger `no-detection` count must be less than or equal to baseline.
2. **MAE gate**: challenger `MAE` must be less than or equal to baseline.
3. **Exact-match guardrail**: challenger `Exact Match` rate must be greater than or equal to baseline.
4. **No-read guardrail**: challenger `No-read` rate must be less than or equal to baseline.
5. **OCR no-digits gate**: challenger `ocr-no-digits` count must be less than or equal to baseline.

If any gate fails, keep `roi-rotaug-e30-640.pt` as default and continue tuning extraction/selection.

Classifier-default rule:

- Keep `digitClassifier.enabled=true` by default and tune ranking/acceptance using `MAE` + guardrails.

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

### 3) Acceptance/Support Guardrails

Files:

- `src/ocr/pipeline.js` (`finalizeSelection`, evidence ranking)
- `src/ocr/recognition.js` (candidate scoring)

Focus:

- Balance strictness (avoid false positives) vs recall (avoid no-read).
- Validate with histogram movement, not single-image anecdotes.
- Active guardrails in current pipeline: evidence ranking plus edge-candidate corroboration/cell-strength checks.

### 4) ROI Sanity Gates (usually not primary blocker)

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
