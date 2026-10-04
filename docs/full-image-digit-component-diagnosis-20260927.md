# Component diagnosis - September 27, 2026

The evidence points to a recognition generalization gap in addition to ROI
failures. Correcting localization and reading order does not make recognition
reliable: two different human-guided crop constructions each reach 29/44 exact
readings. This is not a theoretical ceiling; the constructions change input
geometry and can lose previously correct readings.

No models, runtime settings, canonical annotations, or DVC pointers were
changed. This diagnosis does not authorize model promotion or claim that a
particular new training recipe will solve the remaining errors.

## Scope and method

Use all 44 sources and all five checkpoints from the original September grouped
CV, scoring against the readings confirmed in the
[human review](../backend/data/digit_dataset/reading_reviews/20260927.json).
Exclude the three later imports, separate test holdout and legacy stress source.
Verify original checkpoint hashes, training provenance, exact original fold
manifest, source bytes, and unchanged human digit geometry/direction before
inference. The historical 2345 label is scored as the confirmed 2342; the
historical weights and provenance are preserved.

Reuse the [verified baseline replay](full-image-digit-roi-context-20260927.md),
checking its protected input hashes and binding its results file to this plan.
New predictions are single-image CPU inference, confidence 0.30, IoU 0.70,
imgsz 1280, max detections 300 and class-agnostic NMS, with EXIF transpose and the
same RGB-to-BGR conversion as runtime. No parameter sweep is performed.

The diagnostic arms are fixed before inference:

1. Existing runtime predictions and the frozen UI-selected reading order.
2. The same detections, sorted in the human-reviewed reading direction. No new
   inference or pixel rotation: this isolates order selection.
3. Human ROI annotations, expanded by the same 0.26 width / 0.16 height runtime
   margins, read in the frozen UI-selected direction.
4. The same human ROI crop, read in the human-reviewed direction.
5. The union of reviewed digit boxes plus 0.75 times its longest side of context,
   read in the human-reviewed direction. This also changes context and scale.
6. The same union crop physically rotated upright before a new inference pass.
   Physical rotation is separate from sorting existing detections.

The five ROI-blocked sources have no frozen UI angle. Arm 3 is therefore
reported only on the other 39 photos; assigning those five a human angle would
mix localization and orientation interventions.

For the training-source comparison, infer every source with each checkpoint
using arm 5: 44 held-out source/checkpoint pairs and 176 pairs from training
sources. These are the same 44 photos, each seen by four checkpoints, not 176
independent training photos. Source membership follows the verified original
fold manifest and the trainer's split logic. This is inference on unaugmented
source-derived crops, not the training loss on augmented batches.

## Component results on all 44 held-out sources

| Diagnostic input | Exact | No-read | Wrong accepted | Readable MAE |
| --- | ---: | ---: | ---: | ---: |
| Existing runtime | 25 | 7 | 12 | 580.51 |
| Existing detections, human order | 26 | 7 | 11 | 363.32 |
| Human ROI and human order | 29 | 2 | 13 | 147.55 |
| Digit-box union crop and human order | 29 | 2 | 13 | 145.24 |
| Union crop physically upright | 26 | 1 | 17 | 67.23 |

Three runtime angles disagree with the reviewed direction. Correct sorting
alone recovers `meter_20260224.JPEG` (4132 -> 2314). It reduces the numeric error
but does not recover the whole reading for `meter_20251009.JPEG` or
`meter_20260911.jpeg`, where digit classes are also wrong.

On the 39 sources with an available runtime angle:

| Crop / order | Exact | No-read |
| --- | ---: | ---: |
| Runtime / runtime | 25/39 | 2 |
| Runtime / human | 26/39 | 2 |
| Human ROI / runtime | 25/39 | 2 |
| Human ROI / human | 27/39 | 2 |

Replacing ROI geometry alone has no net exact-match improvement on this subset.
With human ROI and human order, the five blocked sources become two correct
readings and three wrong readings. Removing the ROI block does not imply five
successful readings, and accepting more outputs is not sufficient improvement.

Human ROI plus human order recovers six baseline failures but loses two baseline
successes. The wider union construction recovers seven and loses three. Equal
29/44 totals do not mean these constructions fail on identical photos. Upright
inference lowers MAE but increases wrong accepted readings; it is not a general
solution either.

## Digit localization versus classification

Map predictions back to the full image using the actual integer crop bounds.
Match human boxes to detections with class-agnostic greedy one-to-one IoU >= 0.5.
The matching is diagnostic and does not choose the emitted reading.

| Input | Human digit boxes | Matched | Missing | Extra | Correct class among matched |
| --- | ---: | ---: | ---: | ---: | ---: |
| Runtime, 39 accepted ROI sources | 156 | 154 | 2 | 0 | 137/154 |
| Human ROI, all 44 | 176 | 174 | 2 | 1 | 159/174 |
| Union crop, all 44 | 176 | 174 | 2 | 0 | 158/174 |
| Union crop, 176 training-source pairs | 704 | 704 | 0 | 2 | 691/704 |

With the union crop, the 13 wrong accepted sequences each have all four digit
boxes matched to human annotations, without extra detections. Their remaining
errors are digit-class errors, rather than gross digit-box localization errors
under this IoU criterion. There are also two no-reads, each missing one digit.
The 16 matched digit-class errors include 1/7, 6/9, 7/1 and other confusions.
Correct human geometry does not restore detail absent in a blurred source.

## Seen versus held-out sources

Both groups below use identical union-crop construction and reviewed order:

| Source membership | Exact | No-read | Wrong accepted |
| --- | ---: | ---: | ---: |
| Seen during checkpoint training | 161/176 (91.5%) | 2 | 13 |
| Excluded from checkpoint training | 29/44 (65.9%) | 2 | 13 |

Every fold has a higher exact-match rate on its training sources than on its
held-out sources. The gap persists when excluding `meter_20260628.JPEG`, whose
historical training label was wrong: 159/172 (92.4%) versus 28/43 (65.1%).
Removing that source from scoring does not remove its influence on the weights.
Of the 15 failures with the held-out union crop, 11 are read correctly by all
four checkpoints that had seen that source during training.

This supports a generalization problem; it does not isolate insufficient
variety, augmentation, optimization, and model inductive bias from one another.
The high training-source score argues against a total inability to fit the
examples, but does not prove that training or architecture is optimal.

## Data coverage and training hypotheses

The 44 sources contain 176 reviewed digits:

| Digit | Instances | Distinct source photos |
| --- | ---: | ---: |
| 0 | 6 | 6 |
| 1 | 13 | 12 |
| 2 | 57 | 43 |
| 3 | 56 | 42 |
| 4 | 10 | 10 |
| 5 | 12 | 11 |
| 6 | 5 | 5 |
| 7 | 8 | 8 |
| 8 | 6 | 6 |
| 9 | 3 | 3 |

Digits 2 and 3 account for 113/176 instances. In a given fold, training coverage
for rare digits is smaller still. Existing balancing adds crops of available
sources; it does not create new independent wheel appearances. All 176
`transition_state` fields are `unknown`, so a quantitative claim that transitions
cause most errors cannot be made from that metadata.

Original provenance records horizontal and vertical reflections at probability
0.5 each, rotation up to 180 degrees, mosaic, mixup, and other transforms. These
are a concrete training hypothesis: reflections create mirrored glyphs while
retaining their classes. Whether that hurts this task requires a controlled
ablation. This experiment has not measured that effect, and does not justify
removing every augmentation or assuming mirrored glyphs are the sole cause.

## Recommended decision sequence

1. Test the narrow augmentation hypothesis first: a paired training experiment
   with horizontal and vertical reflections disabled, holding the corrected
   dataset, recipe, seed and evaluation fixed. Both control and challenger need
   the same corrected data; comparing a newly corrected challenger only with
   these historical weights would confound the result. The current measurements
   motivate this experiment but do not run or validate it.
2. If the generalization gap persists, pilot controlled synthetic examples or
   targeted real captures for sparse digits and wheel positions. Compare
   real-only versus real-plus-synthetic with otherwise identical training.
   Keep source families within one fold and validation exclusively real.
   Require exact-match gains without no-read or accepted-error regressions
   before scaling generation by an order of magnitude.
3. If focused data and recipe experiments do not help, compare an alternative
   recognizer (such as a rectified sequence reader) under the same real-image
   evaluation. Do not infer architectural superiority from these oracles.

ROI recovery remains a separate useful engineering task, but its five baseline
blocks explain only part of the performance plateau. Prioritize reliable digit
classification/generalization alongside ROI availability rather than assuming
ROI repair alone will solve the task.

This corpus has been repeatedly inspected and tuned. It is development evidence,
not a fresh generalization or promotion test. The ROI model may have training
source overlap. Synthetic examples and diagnostic human geometry must not enter
the final real external test. No model is promoted from this report.

## Retained local evidence and checks

- [Frozen diagnostic plan and input hashes](../output/retraining/20260927-component-diagnosis/plan.json)
- [Per-source predictions, matching, and aggregate results](../output/retraining/20260927-component-diagnosis/results.json)
- [Coverage and residual-error summary](../output/retraining/20260927-component-diagnosis/diagnostic-summary.json)
- [Evaluation script](../output/retraining/20260927-component-diagnosis/evaluate.py)
- [Independent aggregation checks](../output/retraining/20260927-component-diagnosis/summarize.py)
- [Execution log](../output/retraining/20260927-component-diagnosis/run.log)

All protected inputs remained unchanged. Independent aggregation verified 44
unique held-out sources, 220 unique source/checkpoint pairs, reported counts and
agreement between the held-out tables. The scripts refuse to overwrite completed
evaluation results. The ignored output directory and historical checkpoints are
local artifacts; cloning Git does not restore them. This is a component benchmark,
not a fresh browser Run test set or an automated code-test suite.
