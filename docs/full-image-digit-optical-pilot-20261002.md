# Controlled optical-variation pilot - October 2, 2026

Status: completed on October 2, 2026; paired-source review verified on October 4.
The pilot did not pass its predeclared improvement criterion; do not extend this
optical recipe to other folds. No checkpoint was promoted.
This is the first small synthetic-data experiment authorized after the
[reflection ablation](full-image-digit-reflection-ablation-20260927.md).
The remaining oracle seen/held-out gap is 22.16 percentage points. That motivates
an experiment, not a claim that generated data will resolve the gap.

## Frozen design

Use fold 0, selected by ordinal before this pilot's inference. Reuse its completed
no-reflections checkpoint as control. Train one challenger from the same YOLOv8n
base, with the same seed 42, MPS, batch 4, image size 1280, 120 maximum epochs,
patience 25, register context 0.75 and balanced digit target 48. Both arms disable
horizontal and vertical flips. The development baseline has 6/9 exact runtime
readings, two no-reads, one wrong accepted reading and readable MAE 0.286.

Replace exactly 32 existing balanced single-digit training crops with optical
variants at the same filenames: four each for digits 0, 1, 4, 5, 6, 7, 8 and 9.
The frozen source counts establish these as less represented than 2 and 3.
Selection is deterministic: crop variant index first, filename second. All donors
belong to folds other than 0; no validation or test source is used to generate
variants. Lineage includes the source filename, fold and byte hash.

The four variants per class are mild directional illumination, a soft bright
reflection band, Gaussian blur at sigma 0.45 native pixels, and illumination plus
the reflection band. No digit shape, orientation, image dimensions, or box
coordinates are changed. The source labels were reviewed by the user; generated
crop labels are inherited byte-for-byte. Agent visual QA inspected all 32
original/variant pairs and the retained boxes. This is ordinary photometric
training augmentation, not newly predicted boxes or a canonical annotation
import. Changed geometry or ambiguous digit identity would require renewed human
review rather than silently inheriting a label.

Train counts remain 354 images, with nine real validation sources and the same
one separate test source. Per-class exposure, batches per epoch and maximum
training budget are unchanged. Early stopping can still select different run
lengths. This isolates optical variation from simply adding more optimizer
steps. All other training crops and all validation/test images and labels remain
byte-identical to the control materialization.

## Evaluation and limits

Use the existing frozen component evaluator on the nine held-out real photos,
with the same ROI model and frozen reading order. Report runtime exact readings,
no-read, wrong accepted readings and MAE on accepted readings; report the human
geometry/order oracle separately. Pass the pilot only with more runtime exact
readings and no increase in no-read, wrong accepted or readable MAE. Inspect
paired source changes before deciding whether to extend to other folds.

This is a single-seed, small development fold. It is not a new external test,
not an automatic promotion gate, and not a tenfold data expansion. These optical
variants reuse existing glyphs: failure would test this limited augmentation
hypothesis, not rule out independently rendered digits, compositing, or targeted
real captures. Existing online augmentation remains the same in both arms.

## Implementation and evidence

The trainer's optional `--train-optical-variants` validates retained QA, donor
membership, source/crop/variant hashes, unchanged dimensions and labels before
replacing any temporary training image. Canonical datasets are never modified.
The manifest and QA hashes become part of checkpoint provenance; removing or
changing this recipe blocks resume. Without the flag, historical provenance and
materialization remain compatible.

- [Plan and protected hashes](../output/retraining/20261002-rare-digit-pilot/plan.json)
- [Original and synthetic preview](../output/retraining/20261002-rare-digit-pilot/index.html)
- [Contact sheet](../output/retraining/20261002-rare-digit-pilot/preview/contact-sheet.jpg)
- [Accepted optical manifest](../output/retraining/20261002-rare-digit-pilot/optical-manifest.json)
- [Visual QA evidence](../output/retraining/20261002-rare-digit-pilot/qa.json)
- [Preflight results](../output/retraining/20261002-rare-digit-pilot/preflight.json)

The baseline materialization hash matches the historical control exactly.
Preflight exercised the public training CLI with `--validate-only`: precisely
32 training images differ, with zero label or validation/test changes. All 64
fast backend tests passed, including rejection of pending QA, wrong folds,
changed artifacts/dimensions, and incompatible resume provenance.

Historical initial launch command (already executed; do not rerun this completed run):

```bash
bash /Users/andrea/GitHubRepositories/Jarvis/output/retraining/20261002-rare-digit-pilot/start-with-auto-pause.sh
```

The driver guards against existing processes and run directories, applies
`scripts/train-with-thermal.py --auto-pause` to the individual training process,
then evaluates and writes the paired comparison. It uses the integrated scripts
in this checkout. No further fold starts automatically. Output and checkpoints
remain local ignored experiment artifacts; no upload, commit or model promotion
is included.

## Completed result and paired-source review

The challenger completed epoch 120 and the thermal launcher exited successfully.
The final comparison completed on October 2 at 18:29 Europe/Rome. The run was
interrupted during epoch 65 when its launcher disappeared; recovery preserved
checkpoint/provenance and resumed after 64 completed epochs. Optimizer and
recorded early-stopping state were verified before resume. This does not establish
bitwise equivalence to uninterrupted training and is a limitation of the pilot.

| Held-out runtime metric | No-reflections control | Optical challenger |
| --- | ---: | ---: |
| Exact readings | 6/9 | 6/9 |
| No-read | 2/9 | 2/9 |
| Wrong accepted readings | 1/9 | 1/9 |
| MAE on accepted readings | 0.285714 | 0.285714 |
| Human geometry/order oracle exact | 7/9 | 7/9 |
| Oracle MAE | 0.444444 | 0.555556 |

The paired review confirms identical runtime readings on all nine sources: zero
new correct readings and zero lost correct readings. The six correct sources
remain correct. The three unresolved cases are:

| Source | Reviewed truth | Runtime, both arms | Control oracle | Challenger oracle |
| --- | --- | --- | --- | --- |
| `meter_20260724.JPEG` | 2348 | 2346 | 2349 | 2346 |
| `meter_20260803.JPEG` | 2349 | No-read: ROI not detected | 2346 | 2346 |
| `meter_20260805.jpeg` | 2350 | No-read: ROI not detected | 2350 | 2350 |

The two runtime no-reads occur before digit inference, so modifying only the
digit detector cannot recover them in this cascade. The July 24 digit error also
persists with human geometry/order; this points to a remaining recognition issue
rather than solely ROI localization. Oracle MAE worsens slightly because its
prediction changes from 2349 to 2346 for truth 2348.

This is a frozen component replay on a development fold, not a fresh browser
Run test set or a locked external test. Held-out refers to digit training;
ROI-training overlap remains possible. The pilot only tests these 32 mild
photometric replacements. It does not establish that independently rendered
glyphs, compositing, or new real captures would fail.

## Decision and next experiment boundary

Close this optical pilot without additional fold training or model promotion.
Retain the control, candidate, original frozen plan, recovery snapshot and QA.
A future experiment should explicitly separate ROI recovery from digit-shape
coverage: the former targets the two no-reads, while the latter targets residual
recognition errors. Any new synthetic design needs its own frozen comparison
before training; these results do not justify a tenfold expansion by themselves.

Verified on October 4: checkpoint, provenance and fold-manifest hashes match both
evaluations; all nine paired expected readings agree; runtime aggregates were
recomputed from saved per-source predictions. No inference or training was rerun.

- [Completed comparison](../output/retraining/20261002-rare-digit-pilot/comparison.json)
- [Candidate predictions](../output/retraining/20261002-rare-digit-pilot/candidate-evaluation.json)
- [Paired-source review](../output/retraining/20261002-rare-digit-pilot/paired-source-review.json)
- [Completed driver status](../output/retraining/20261002-rare-digit-pilot/status.json)
