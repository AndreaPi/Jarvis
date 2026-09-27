# Full-image digit retraining: September 14, 2026

## Decision

Label correction, September 16: the user confirmed `meter_20260628.JPEG` as
**2342**, not 2345. Truth-only rescoring of the retained predictions yields
25/44 exact, 7 no-reads, 12 wrong accepted, readable MAE 580.51. On 28 common
sources, historical/candidate exact readings are 15/18, no-reads 1/1, and MAE
440.30/484.22. The promotion decision is unchanged. Detailed original numbers
below remain a dated record of evaluation with the old truth.

The photo was held out in fold 3; folds 0,1,2,4 trained with its incorrect
label. Rescoring cannot repair that training. Canonical labels and derivatives
are now corrected; original checkpoints and provenance are preserved.
See the [correction record](../output/label-corrections/20260916-meter_20260628/README.md).

Update, September 16: all five folds and their runtime evaluations are complete.
The candidate produces 24/44 exact readings, 7 no-reads, and readable MAE 580.59.
On the 28 sources shared with retained historical folds 1-4, exact readings
improve from 14 to 17 and no-reads remain at 1, but readable MAE worsens from
440.41 to 484.33. The candidate does not satisfy all promotion criteria.
Keep the existing models/settings. The next useful step is a focused audit of
the largest out-of-fold errors and no-reads before choosing another experiment.

The following paragraphs record the earlier decision to extend fold 4:

The initial candidate at confidence 0.25 failed the runtime criteria: it
reduced readable MAE but lost one exact reading overall and introduced one
no-read on the seven common validation photos. Its unchanged configuration
was therefore not extended immediately.

The subsequently authorized calibration at confidence 0.30 recovered the
missing correct reading, retained every previously correct full-corpus
reading, and introduced no new no-reads. It meets the predeclared development
criteria for extension: lower common-source MAE, no reduction in exact matches,
and no increase in no-reads. Folds 0-3 were consequently queued for serial training and
evaluation, with the training recipe unchanged and runtime confidence frozen
at 0.30. This is not a promotion: one additional diagnostic abstention became
an accepted wrong reading, as recorded below.

This is a development decision on a small, previously tuned fold, not an
external generalization estimate. The existing default models and the disabled
full-image shadow configuration remain unchanged.

## Frozen experiment

- Training code: `ae373d7025df97eed5b4ef2b0f132fd0cc1cac01`; runtime evaluation:
  `0143cb43dd7431c228227b4c24a64585317607e1`. The intervening changes do not
  alter the training, backend, OCR, or shadow benchmark code.
- Recipe: YOLOv8n pretrained base, fold 4, seed 42, MPS, image size 1280,
  batch 4, up to 120 epochs, patience 25, register crops with context 0.75,
  and train-only balanced digit crops targeting 48 examples per class.
- Candidate: 36 training source photos, 8 validation source photos; 350
  materialized training images including derived crops. Training stopped after
  90 epochs; the selected checkpoint was from epoch 65.
- Historical baseline: 28 training source photos and 7 validation source
  photos. Every original source keeps its persistent fold assignment. All
  derivatives of a source photo stay in its training group.
- Baseline run: `backend/runs/full-image-digit-detector-balanced48-crops-fold4/`.
- Candidate run: `backend/runs/full-image-digit-detector-sep2026-balanced48-fold4/`.

Both runs use `weights/best.pt`. Their SHA-256 values are:

| Model | Checkpoint SHA-256 |
| --- | --- |
| Baseline | `82bb9ece770dcd08c56384eac833b57823ae967dbb63fa1a4761f2748e5bb6b8` |
| Candidate | `8da562b69d82ad9ef2dc57f54381a6f3b1b9a26200aa163bb563424314f0af09` |

The original baseline CV manifest has SHA-256
`fd7a1d720b3f36265f793ed46692abaf8cec0d8252c5c4b328c24bcdc08ede21`;
the candidate manifest has SHA-256
`0c069df1284f3d37994a597016ec2cc55f3e423c4b157a1ae7dba9dd8b37a555`.
Each checkpoint was validated against its own original manifest; neither
training provenance file was rewritten.

## Controlled runtime comparison

Both checkpoints ran through `npm run qa:full-image-digit-shadow`, using the
same canonical ROI and per-cell classifier models, CPU inference, confidence
0.25, NMS IoU 0.70, image size 1280, maximum 300 detections, and ROI expansion
0.26 horizontally / 0.16 vertically. Each photo was processed individually.
The shadow used the primary OCR angle and did not change the selected reading.

The frontend source hashes, runtime settings, common-photo bytes, primary
readings, and selected angles matched between runs. All seven comparison
photos belong to validation fold 4 in both original manifests.

| Path | Exact readings | No-read | Readable MAE |
| --- | ---: | ---: | ---: |
| Existing primary OCR, both runs | 3/7 | 0/7 | 40.00 |
| Baseline shadow | 5/7 | 0/7 | 14.57 |
| Candidate shadow | 4/7 | 1/7 | 1.00 |

MAE excludes no-reads. On the six photos read by both models, MAE also improves
from 17.00 to 1.00; the numerical improvement is real on that subset, but it
does not compensate for losing a previously correct reading to abstention.

| Source | Expected | Baseline shadow | Candidate shadow |
| --- | ---: | ---: | ---: |
| `meter_20260227.JPEG` | 2315 | 2315 | 2312 |
| `meter_20260310.JPEG` | 2318 | 2318 | 2318 |
| `meter_20260323.JPEG` | 2320 | 2320 | 2320 |
| `meter_20260413.JPEG` | 2324 | 2324 | NO-READ |
| `meter_20260423.JPEG` | 2327 | 2322 | 2327 |
| `meter_20260512.JPEG` | 2331 | 2331 | 2331 |
| `meter_20260719.JPEG` | 2346 | 2249 | 2349 |

The candidate's complete eight-photo validation slice is 5/8 exact, 1/8
no-read, MAE 0.8571. Its additional source, `meter_20260828.jpeg=2353`, is
correctly read by the candidate but absent from the baseline's original CV
manifest, so it is reported separately from the planned seven-photo comparison.

The complete 46-photo UI diagnostic is 30/46 exact, 6 no-reads, MAE 370.60 for
the baseline, versus 24/46, 11, and 436.89 for the candidate. This includes
training overlap, the historical holdout, and retained legacy-stress images;
it is not an unbiased validation result or a model-promotion gate.

## Verified regressions

Direct single-image inference at the unchanged runtime settings reproduced
both regressions, including the UI detection counts and readings. ROI crop
geometry was identical between checkpoints. Visual inspection showed:

- On `meter_20260227.JPEG`, the candidate labels the final `5` as `2` at
  confidence 0.450; the baseline correctly labels it `5` at 0.777.
- On `meter_20260413.JPEG`, the candidate detects the four real digits but
  adds a false `7` on the meter rim, outside the digit windows, at confidence
  0.282. Five detections correctly trigger the existing no-read safeguard.

These are classification and false-positive regressions, not a change in ROI
geometry or reading direction. No confidence/NMS sweep, geometry filter,
annotation change, or additional training was applied during this initial
diagnosis. The subsequent authorized confidence experiment is recorded below;
this fold cannot become a fresh test set again through parameter tuning.

## Repeatability check

Both complete UI benchmarks were repeated later on September 14 with fresh
backend processes, unchanged photo/checkpoint hashes, the same frontend code,
and the same CPU settings. All seven common validation records are identical
for each checkpoint, including confidences and rotation candidates. The main
table for confidence 0.25 and its failed guardrails are unchanged by the repeat.

Five additional consecutive direct inferences on `meter_20260413.JPEG` also
reproduce the candidate's complete previous response exactly: five boxes,
`digit-detection-count`, and a false `7` at confidence 0.2817733585834503.
This no-read is a repeatable model/acceptance-rule outcome in the tested
configuration, not an observed intermittent request failure.

The complete 46-photo app result is not perfectly identical: the candidate's
first report had a `network-error` on `meter_20260606.JPEG` (fold 3, outside
the common validation slice). The repeat reads its expected `2337` correctly.
The original exception details were not retained, so its precise technical
cause is unknown. Every other row matches, and all 46 baseline rows match.
Candidate full-corpus metrics therefore change from 24/46 exact, 11 no-reads,
MAE 436.89 to 25/46 exact, 10 no-reads, MAE 424.75. The original full-corpus
no-read count must not be interpreted as eleven failures of digit detection.

The repeat does not establish invariance across different devices, software
versions, input preprocessing, or retraining runs. It confirms the stability
of the specific validation failure with the saved checkpoint and fixed inputs.
See the local [repeat verification](../output/retraining/20260914-full-image-fold4-repeat-1/repeat-verification.json)
and [five direct repetitions](../output/retraining/20260914-full-image-fold4-repeat-1/no-read-repeat.json).

## Authorized calibration and CV extension

The confidence-only experiment retained the same checkpoint, UI source,
photos, primary readings/angles, and inference settings except for changing
confidence from 0.25 to 0.30. It used the repeat run, which had no shadow
network error, as its candidate-0.25 control.

| Common seven validation sources | Exact | No-read | Readable MAE |
| --- | ---: | ---: | ---: |
| Historical baseline, confidence 0.25 | 5/7 | 0/7 | 14.57 |
| Candidate, confidence 0.25 | 4/7 | 1/7 | 1.00 |
| Candidate, confidence 0.30 | 5/7 | 0/7 | 0.8571 |

Only two full-corpus readings changed relative to candidate 0.25:

- `meter_20260413.JPEG`: NO-READ becomes correct `2324` after excluding the
  false rim detection at confidence 0.282.
- `meter_20260416.JPEG`: NO-READ becomes wrong `2322` (expected `2325`). This
  additional accepted error is a tradeoff, not a second correct recovery.

No previously correct reading was lost, no new no-read appeared, and the run
reported no shadow request failure. Candidate full-corpus results change from
25/46 exact, 10 no-reads, MAE 424.75 to 26/46 exact, 8 no-reads, MAE 402.47.
Accepted wrong readings increase from 11 to 12. These training-overlap results
still trail the historical baseline's full-corpus diagnostic and are not
evidence for promotion.

The local [calibration summary](../output/retraining/20260914-full-image-fold4-confidence030/calibration-summary.json)
records the original development gates and all changed readings. The existing
model defaults remain at their original settings; 0.30 is an experiment-only
environment override.

The four additional folds each hold out 9 of the 44 active CV source photos,
training on the other 35. Preflight materialization passed for all four:
352 training images for folds 0-2 and 354 for fold 3, including train-only
derived crops. Fold 4 and its 8 validation sources reuse the completed run.
New runs use `backend/runs/full-image-digit-detector-sep2026-balanced48-foldN/`
for `N=0,1,2,3`, without overwriting existing checkpoints.

The [frozen CV plan](../output/retraining/20260914-full-image-cv-confidence030/plan.json)
protects source/code/checkpoint hashes. A serial driver trains each fold, runs
the full-image evaluator at 0.30, then runs the UI benchmark at 0.30. It
retains all checkpoints and raw reports locally and reports accepted wrong
readings separately. Validation request failures receive one bounded retry;
model no-reads are never retried away. The final aggregate must contain each
of the 44 validation source photos exactly once.

Matching historical balanced48 checkpoints were verified for folds 1-4.
No matching retained fold-0 checkpoint was found, so a complete historical
five-fold comparison cannot be claimed. New folds 1-3 get common-source
comparisons against historical 0.25 checkpoints; the existing fold-4 comparison
is recorded above. All resulting CV metrics remain development evidence:
fold 4 was used to calibrate confidence and is not a locked external test.

## Completed five-fold evaluation (September 15-16)

The final runtime aggregate contains each of the 44 active source photos exactly
once, assigned to its held-out fold. It is grouped cross-validation, not an
ensemble or a single checkpoint trained on all 44 photos. Fold 4 was used for
confidence calibration; these remain development results, not an untouched
external test. All selected validation requests succeeded technically.

| Validation fold | Sources | Exact | No-read | Wrong accepted | Readable MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 9 | 6 | 2 | 1 | 0.14 |
| 1 | 9 | 4 | 3 | 2 | 1166.33 |
| 2 | 9 | 3 | 1 | 5 | 1159.13 |
| 3 | 9 | 5 | 1 | 3 | 650.50 |
| 4 | 8 | 6 | 0 | 2 | 0.75 |
| Total | 44 | 24 | 7 | 13 | 580.59 |

The MAE denominator is the 37 readable photos, not all 44; the total is weighted
by readable counts, not an unweighted average of fold means. The primary OCR
path on the same rows gives 11 exact, 5 no-reads, 28 wrong accepted, and readable
MAE 359.69. This primary-path reference is not the historical detector comparison
and does not establish that primary OCR is out of training on these photos.

### Paired historical detector comparison

Only 28 validation sources in folds 1-4 can be compared against verified retained
historical checkpoints. Fold 0 has no matching retained historical checkpoint.
Baseline confidence is 0.25; candidate confidence is the frozen 0.30. This
compares the two configured workflows, not model weights at a common threshold.

| Workflow | Exact | No-read | Wrong accepted | Readable MAE |
| --- | ---: | ---: | ---: | ---: |
| Historical detector | 14/28 | 1/28 | 13/28 | 440.41 |
| Retrained detector | 17/28 | 1/28 | 10/28 | 484.33 |

Both workflows read the same 27 photos, so the MAE regression is not caused by
a different readable denominator. The candidate gains five correct readings
and loses two, for a net gain of three. Fewer wrong readings coexist with larger
errors on some remaining failures. For example, `meter_20260416.JPEG` (2325)
changes from 2322 to 3323, and `meter_20251009.JPEG` (2279) from 6222 to 6522.
The candidate's largest full-CV errors include 1784 -> 7784 and 2357 -> 7555;
these need inspection before attributing them to classification, ordering,
orientation, or ROI geometry. No new threshold or training sweep was launched.

### Completion and integrity

Fold 3 was paused at the user's request on September 14 and resumed September
15 from the verified `last.pt`/early-stopping pair (zero-based saved epoch 58).
It completed through displayed epoch 120, with the best checkpoint at epoch 95.
The original data, recipe, and optimizer/early-stopping resume gates were kept.
Its best checkpoint SHA-256 is
`7debefcc42408717c498dfd699531b04f76f503b5828c6b978da5a0f4fe5c23c`.

The subsequent UI evaluation initially failed before inference because system
`python3` required acceptance of an Xcode license. Recovery prepended the
project virtualenv to the evaluation process PATH, reused completed training
and full-image evaluation, and retained the failed log. Candidate and baseline
UI evaluations then completed on September 15. No license was accepted or
system configuration changed by this recovery.

The final audit verified unique source membership, recomputed all metrics,
checked frozen inputs, checkpoint/provenance/original-fold-manifest hashes,
and identical frontend source hashes across reports. Paired expected readings,
primary readings, selected angles, and orientation sources match. No model was
promoted or uploaded.

Local evidence:

- [Final CV report](../output/retraining/20260914-full-image-cv-confidence030/cv-summary.json).
- [Final verification and paired comparison](../output/retraining/20260914-full-image-cv-confidence030/final-verification-and-paired-comparison.json).
- [Fold 3 resumed training log](../output/retraining/20260914-full-image-cv-confidence030/fold3-training-resume-20260915.log).
- [Fold 3 candidate UI report](../output/retraining/20260914-full-image-cv-confidence030/fold3-candidate-ui.json).
- [Fold 3 historical UI report](../output/retraining/20260914-full-image-cv-confidence030/fold3-baseline-ui.json).

The proposed shared training/thermal launcher and optional automatic pause are
recorded separately in the [deferred thermal-control proposal](./training-thermal-control-proposal.md).

## Other measurements and retained evidence

The earlier full-image-only evaluator, without the runtime ROI crop and using
the recorded reading direction, returned 0/7 exact for both checkpoints.
Baseline/candidate no-reads were 1/0 and readable MAE was 339.83/7.29. Keep this
separate from the runtime table above. The two training validation mAP values
also cover different sets of seven/eight photos and are not the common-source
head-to-head metric.

Local evidence, intentionally outside ordinary Git commits:

- [Training plan, snapshots, and full-image evaluation](../output/retraining/20260913-full-image-fold4/experiment.json).
- [Runtime comparison, checksums, and decision](../output/retraining/20260914-full-image-fold4-comparison/comparison-summary.json).
- [Baseline UI report](../output/full-image-digit-shadow-qa/20260914-124304/README.md).
- [Candidate UI report](../output/full-image-digit-shadow-qa/20260914-124349/README.md).
- [2315 regression overlay](../output/retraining/20260914-full-image-fold4-comparison/meter_20260227-comparison.jpg).
- [2324 regression overlay](../output/retraining/20260914-full-image-fold4-comparison/meter_20260413-comparison.jpg).

Each overlay shows baseline on the left and candidate next to it. Raw
detections are in `regression-detections.json` beside the comparison summary.
These links require the retained local artifacts; cloning Git alone does not
restore them.

Two initial UI attempts used frontend port 8760, outside the backend CORS
allowlist. All rows failed before useful inference. Those attempts are marked
invalid and retained under `invalid-frontend-port/`; none of their metrics
contribute to this report. The valid reruns used frontend port 8000.
