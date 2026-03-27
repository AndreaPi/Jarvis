# OCR Tuning Playbook

This playbook documents the practical loop used to improve OCR quality in Jarvis.

Current baseline policy:

- Use the latest UI **Run test set** histogram as source of truth (`window.__jarvisLastTestSetHistogram`).
- Treat fixed numeric snapshots as historical only; they go stale quickly as thresholds/ranking change.
- Evaluation uses `MAE` as the primary promotion signal; `Exact Match` and `No-read` are guardrails.
- The active local test-set CSV has `17` images.

Digit dataset status (current workflow):

- Dataset generation now uses `extract_digit_windows.py` -> `split_digit_windows.py` -> `label_digit_sections.py`.
- `split_digit_windows.py` canonicalizes orientation (major axis + optional reading-direction `flip180` overrides) before equispaced 4-way split.
- Classifier training consumes `data/digit_dataset/sections_labeled/{train,val,test}`.
- Synthetic generation remains train-only (`sections_synthetic/train`) and is mixed into training with `--synthetic-target-ratio`.

## Immediate Next Steps (March 4, 2026)

1. Keep `digitClassifier.forceInitialPreviewCandidate=false` by default (force mode increased no-read materially).
2. Reduce `classifier-edge-gate-final-drop` with targeted edge-gate tuning and explicit A/B runs.
3. Verify each edge-gate tweak on full test set with `MAE` + guardrails (`Exact Match`, `No-read`) before keeping it.
4. Prioritize fixes on current top-absolute-error images after each run, not a fixed historical list.

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
It requires these local checkpoints before starting:

- `backend/models/roi-rotaug-e30-640.pt`
- `backend/models/roi.pt`
- `backend/models/digit_classifier.pt`

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
- Active guardrails in current pipeline: evidence ranking, mixed primary evaluation of top edge and base strip candidates, narrow `scan-roi` / base fallback only when base candidates were not already evaluated and edge support is still weak or edge-only, and final edge-confidence checks.

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
