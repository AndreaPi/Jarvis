# Predicted ROI context experiment - September 27, 2026

Decision: reject the wider crop as a replacement for the existing shadow
configuration. It reduces numeric error but loses more exact readings than it
recovers. No runtime code, model, canonical label, or DVC pointer was changed.

## Reviewed truth and scope

The user confirmed all 49 canonical readings, with no corrections or doubts.
The [review export](../assets/jarvis-revisione-letture-20260927.json) and
[verification record](../backend/data/digit_dataset/reading_reviews/20260927.json)
retain source/detail hashes and three separate observations about crop quality.
Those observations are not new annotation geometry approvals.

This paired experiment covers all 44 sources from the September training CV,
including previously successful readings. It uses the reviewed correction
`meter_20260628.JPEG = 2342`. The three subsequent imports, the separate test
holdout, and the excluded legacy stress photo do not enter this comparison.

## Frozen comparison

- Each source uses its original held-out digit-model fold checkpoint. Checkpoint
  hashes, original provenance, exact original fold-manifest hash, and source
  hashes were verified before inference.
- Baseline: the existing shadow crop, expanding the accepted predicted ROI by
  0.26 of its width and 0.16 of its height on each corresponding side.
- Single candidate: expand that same raw predicted ROI on all sides by 0.75
  times its longest pixel side, clipping to image bounds. The coefficient comes
  from the training crop recipe; there was no parameter sweep.
- The promoted ROI detector and its confidence, IoU, input size, and sanity
  checks remain unchanged. A missing/rejected ROI remains a no-read.
- Digit inference uses CPU, confidence 0.30, IoU 0.70, input size 1280,
  class-agnostic NMS, and maximum 300 detections. Crops are inferred individually.
- Reading direction is the frozen angle selected by the primary UI pipeline.
  Neither human boxes nor human reading direction is used to construct or read
  candidate crops. Images use EXIF transpose, RGB decoding and runtime BGR input.
- The pre-run criterion required no lost exact readings, no increase in
  no-reads or readable MAE, and at least one improvement to continue.

All 44 baseline values and detection counts reproduced the frozen report.
Protected inputs remained unchanged after the run. This is a direct runtime
component replay with frozen UI orientation, not a fresh browser Run test set.

## Results

| Metric | Existing crop | Wider predicted ROI crop |
| --- | ---: | ---: |
| Exact readings | 25/44 | 24/44 |
| No-reads | 7/44 | 7/44 |
| Wrong accepted readings | 12/44 | 13/44 |
| MAE among readable photos | 580.51 | 266.43 |
| MAE on the same 36 readable photos | 429.97 | 273.00 |

Three exact readings are recovered: `meter_20260216.JPEG` (2312),
`meter_20260227.JPEG` (2315), and `meter_20260719.JPEG` (2346).
Four formerly exact readings become wrong: `meter_20260214.JPEG` (2311 -> 2374),
`meter_20260603.JPEG` (2336 -> 2339), `meter_20260612.JPEG` (2339 -> 2336),
and `meter_20260307.JPEG` (2317 -> 2311).

The no-read total hides a swap: `meter_20200701.JPEG` changes from wrong 7784
to no-read (truth 1784), while `meter_20260821.jpeg` changes from no-read to
wrong 2322 (truth 2352). Lower readable MAE partly reflects abstention on the
6000-unit error, although error also falls on the common readable subset.
None of the seven baseline no-reads becomes an exact reading.

| Fold | Existing exact | Candidate exact | Existing / candidate no-reads |
| --- | ---: | ---: | ---: |
| 0 | 6/9 | 6/9 | 2 / 2 |
| 1 | 4/9 | 1/9 | 3 / 3 |
| 2 | 3/9 | 4/9 | 1 / 1 |
| 3 | 6/9 | 5/9 | 1 / 1 |
| 4 | 6/8 | 8/8 | 0 / 0 |

## Interpretation and next step

A universal context increase is not supported: the perfect fold-4 result does
not hold across folds. Keep the existing configuration. Do not choose between
the two crops using the known correct reading or cherry-pick per-photo rules.

A subsequent experiment should isolate ROI localization failures or runtime
orientation selection, with its rule fixed before evaluating the whole corpus.
Crop quality concerns from the human review remain separate dataset diagnostics.
No additional training or model promotion follows from this result.

These are development measurements, not an untouched generalization estimate.
The digit model excludes each evaluated source from its training fold, but the
promoted ROI model may have source overlap. Historical weights still encode the
old incorrect 2345 training label in folds other than 3; corrected-truth scoring
does not retrain those weights. The prior fold-4 calibration and error inspection
also make this corpus tuning data. Promotion still needs the prescribed checks
and a locked external test set.

## Retained local evidence

- [Pre-run plan and input hashes](../output/retraining/20260927-predicted-roi-context/plan.json)
- [Per-photo predictions and metrics](../output/retraining/20260927-predicted-roi-context/results.json)
- [Evaluation script](../output/retraining/20260927-predicted-roi-context/evaluate.py)
- [Execution log](../output/retraining/20260927-predicted-roi-context/run.log)
- [Original retraining report](full-image-digit-retraining-20260914.md)

The output directory and historical checkpoints are local ignored artifacts;
Git alone does not restore them. The script refuses to overwrite a completed
results file. Reproduction requires a separate experiment directory, the same
verified checkpoints and source bytes, and the recorded frozen UI reports.
