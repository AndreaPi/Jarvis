# OCR Tuning Playbook

This playbook documents the practical loop used to improve OCR quality in Jarvis.

Current baseline (March 2, 2026, fallback `OFF`):

- Test set: `0/14`
- Pinned model (`roi-rotaug-e30-640.pt`): `mismatch` 6, `ocr-no-digits` 7, `no-detection` 1
- Challenger (`roi.pt`): `mismatch` 4, `ocr-no-digits` 10, `no-detection` 0
- Gated fallback experiment (`JARVIS_DIGIT_FALLBACK=1`) reduced `ocr-no-digits` but increased `mismatch` with no accuracy gain, so fallback remains disabled.

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
- Revert changes that shift failures without improving accuracy.

## Automated Checkpoint Diff

Use the scripted checkpoint comparison to produce a per-image report between the pinned baseline and a challenger model:

```bash
npm run benchmark:roi-diff
```

For the gated classifier fallback experiment (fallback only after `ocr-no-digits`), run:

```bash
JARVIS_DIGIT_FALLBACK=1 npm run benchmark:roi-diff
```

Compare fallback `ON` vs `OFF` with the same promotion gates; do not accept `ocr-no-digits` reductions that simply convert into `mismatch`.

Artifacts are saved to:

- `output/roi-checkpoint-diff/<timestamp>/roi-diff-report.md`
- `output/roi-checkpoint-diff/<timestamp>/roi-diff-report.json`
- `output/roi-checkpoint-diff/<timestamp>/{baseline,challenger}/stages/*` (stage `5` and `6` snapshots)

The report includes:

- Per-image `Detected`, `Failure Reason`, and top reject reason.
- Side-by-side stage `5. detected strip crop` and `6. OCR input candidate`.
- Summary deltas for `mismatch`, `ocr-no-digits`, and `no-detection`.

## Checkpoint Promotion Gates

Promote a challenger checkpoint only if all gates pass on the same test-set run:

1. **No-detection gate**: challenger `no-detection` count must be less than or equal to baseline.
2. **Value Match gate**: challenger `Correct` must be at least baseline and at least `1` image on the current 14-image set.
3. **OCR no-digits gate**: challenger `ocr-no-digits` count must be less than or equal to baseline.

If any gate fails, keep `roi-rotaug-e30-640.pt` as default and continue tuning extraction/selection.

Fallback-specific rule:

- Keep `digitClassifier.enabled=false` by default until fallback `ON` beats fallback `OFF` on `Correct` and does not increase `mismatch`.

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

### 2) Word-pass Input Modes (`mismatch` vs `ocr-no-digits`)

File: `src/ocr/config.js` (`OCR_CONFIG.roiDeterministic.wordPassModes`)

Examples:

- `['raw']` (current default): conservative
- `['raw', 'soft']`: can reduce no-read but may increase mismatches
- `['raw', 'soft', 'binary']`: often increases wrong confident reads

Use temporary experiments first, then codify only if net-positive.

### 3) Acceptance/Support Guardrails

Files:

- `src/ocr/pipeline.js` (`finalizeSelection`, evidence ranking)
- `src/ocr/recognition.js` (candidate scoring)

Focus:

- Balance strictness (avoid false positives) vs recall (avoid no-read).
- Validate with histogram movement, not single-image anecdotes.
- Active guardrail in current pipeline: word-pass support (`hits` / `topHits` vs `minWordPassHits`).

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
3. Histogram delta documented (`mismatch` vs `ocr-no-digits`).
4. Any tuning knob changes are explained in PR notes.
