# Controlled reflection ablation - September 27, 2026

Completed October 2, 2026: all ten runs and their evaluations finished. The
no-reflections arm passes the predefined development criteria. No model is
promoted; the original launch and recovery notes below are historical. The user authorized the paired
comparison and, if the generalization gap persists, a controlled synthetic-data
pilot for rare digits.

The [component diagnosis](full-image-digit-component-diagnosis-20260927.md)
motivates this experiment. Both arms are trained anew on identical corrected
data. Historical checkpoints with the incorrect 2345 label are not the control.

## Frozen design

- Original 44-source, five-fold CV membership, corrected `2342` label, reviewed
  boxes, and the separate test source retained in materialization. Test is not
  evaluated or used for selection. The three later imports are excluded from
  both arms to preserve the diagnostic cohort.
- Full source bytes, YOLO labels and manifests copied into the experiment's
  snapshot directory. Both arms use that snapshot. Source and code hashes are
  checked between stages; canonical data is not rewritten.
- Ten serial runs: control then no-reflections for fold 0, then the same pair
  for folds 1 through 4. Unique names preserve all historical checkpoints.
- Same YOLOv8n base weights, seed 42, MPS, image size 1280, batch 4, up to 120
  epochs, patience 25, register context 0.75 and balanced digit target 48.
- Control: `fliplr=0.5`, `flipud=0.5`. Challenger:
  `--disable-reflections`, setting only those two probabilities to zero.
  All other augmentation remains unchanged. Both recipes retain ordinary
  rotation augmentation. The trainer records the effective augmentation and
  rejects a resume whose recipe differs from original provenance.
- Training uses Ultralytics 8.4.45. A shared seed and deterministic option do
  not establish MPS bitwise repeatability or quantify training variance.

## Evaluation and decisions

After each run the driver evaluates held-out runtime complete readings with the
same ROI cascade, CPU inference, confidence 0.30, IoU 0.70, image size 1280 and
frozen primary UI angles. These angles depend on the unchanged primary pipeline,
not on either challenger. Five historical ROI-blocked sources have no angle;
this is a component replay, not a fresh browser Run test set.

Separately, both arms use human digit-box union crops and reviewed direction to
measure the seen/held-out recognition gap. Geometry oracles are diagnostic only.
This produces 44 held-out predictions and 176 repeated training-source predictions
per arm across the five folds; the latter are not 176 independent photos.

The no-reflections arm passes development criteria only if it increases exact
runtime readings without increasing no-reads, wrong accepted readings, or
readable MAE. Inspect paired sources and consistency across folds before drawing
conclusions. A promising single-seed result still needs confirmation and the
normal promotion gates.

A remaining seen/held-out oracle gap of at least 10 percentage points in the
no-reflections arm is a predeclared practical trigger for the authorized small
synthetic pilot. This threshold is not a significance test or proof that more
data will solve the problem. The driver records the trigger after comparison;
it does not generate unreviewed images or silently start a synthetic training.
If results are inconclusive, inspect the errors before changing the experiment.

## Execution and observability at initial launch

Local experiment directory:
`output/retraining/20260927-reflection-ablation/`.

- [Plan and hashes](../output/retraining/20260927-reflection-ablation/plan.json)
- [Live stage and child PID](../output/retraining/20260927-reflection-ablation/status.json)
- [Driver log](../output/retraining/20260927-reflection-ablation/driver.log)
- [First control training log](../output/retraining/20260927-reflection-ablation/fold0-control-training.log)
- [Serial driver](../output/retraining/20260927-reflection-ablation/driver.py)
- [Per-run evaluator](../output/retraining/20260927-reflection-ablation/evaluate-run.py)

The driver runs detached with `caffeinate -i`; this prevents idle sleep, not all
forms of sleep. It stops on a stage failure or protected-input mismatch. A
`STOP` file in the experiment directory stops before the next stage. Interrupting
the recorded driver PID forwards SIGINT to its active child; retained training
checkpoints can be resumed using the trainer's existing resume workflow. The
serial driver itself refuses to overwrite existing run directories; restarting
it blindly is not a resume mechanism.

The thermal monitor was not started: `sudo -n` reported that a password is
required. The user can start the existing logger in a separate terminal:

```bash
/Users/andrea/GitHubRepositories/Jarvis/scripts/monitor-thermal.sh /Users/andrea/GitHubRepositories/Jarvis/output/retraining/20260927-reflection-ablation/thermal-events.log
```

Automatic thermal pause and shared logger lifecycle remain the
[deferred proposal](training-thermal-control-proposal.md), not active protection.

## Validation before launch

- Both fold-0 preflights produced the identical materialized-dataset hash; only
  the two reflection probabilities differed in their provenance.
- The training regression test checks effective no-reflection kwargs and
  recorded provenance, unchanged defaults, and refusal to resume with the flag
  omitted. The full fast backend suite passed: 60 tests.
- Driver/evaluator syntax checks and `git diff --check` passed.
- No existing training process was active at launch. The new process reported
  MPS on Apple M5 with the intended control parameters.

The experiment and checkpoints are local ignored artifacts. No DVC upload,
Git commit/push, or model promotion is included in this execution.

## Recovery on September 28

The fold-0 control completed 114 epochs in 2.245 hours; early stopping selected
its epoch-89 checkpoint. The driver then stopped at its post-training input
check because the original review export no longer existed in `assets/`.
Evaluation and the remaining nine trainings had not begun.

All other protected files matched their pre-run hashes. The retained human-review
QA record references the exact missing export hash; its frozen readings and
source hashes match every training/test source in the experiment snapshot.
Recovery preserves the original plan and failure log, protects that QA record
instead of the absent export, and reuses the verified completed checkpoint.
Neither training data nor checkpoint provenance was rewritten. No deleted user
file was recreated.

- [Recovery evidence](../output/retraining/20260927-reflection-ablation/recovery-20260928.json)
- [Recovery plan](../output/retraining/20260927-reflection-ablation/plan-recovery-20260928.json)
- [Recovery driver log](../output/retraining/20260927-reflection-ablation/driver-recovery-20260928.log)

The recovery driver was started at 00:29 Europe/Rome, September 28, beginning
with evaluation of the completed control before the no-reflections run.

At 00:48 Europe/Rome the original review export was present again and its SHA-256
matched the initial plan and retained QA record exactly. All frozen training
readings and source-image hashes were checked against it. An identical copy is
now retained inside the experiment's `review-evidence/` directory, with
[verification evidence](../output/retraining/20260927-reflection-ablation/review-restored-20260928.json).
The active recovery driver was not changed or restarted. Its fold-0
no-reflections run was verified active at epoch 20/120.

## Earlier resume through the logging-only thermal launcher

On September 28, commit `beba73b` was cherry-picked into the retraining branch
as `3b7c8ee`. The documentation conflict was resolved using the implemented
launcher description; the restored local backend guidance retains both the
launcher instruction and `--disable-reflections` instruction. Local training
code/test changes were restored byte-for-byte. All 155 previously inventoried
experiment/run files retained their hashes across integration.

Before resumption, no active training was found. The interrupted fold-0
no-reflections `weights/last.pt` has epoch index 32 (33 completed epochs),
optimizer state and matching early-stopping hash. The real trainer's resume
validation passed using the original run directory and frozen snapshot, with
`YOLO.train` intercepted before training. A new resume begins epoch 34.
The scripts suite and all 60 fast backend tests passed.

The user starts the prepared command in their own foreground terminal so any
sudo password stays in that terminal:

```bash
bash /Users/andrea/GitHubRepositories/Jarvis/output/retraining/20260927-reflection-ablation/resume-with-thermal.sh
```

This invokes `scripts/train-with-thermal.py` without bypassing monitoring, waits
for its first thermal-pressure sample, then runs the
`driver-thermal-resume-20260928.py` coordinator. The coordinator reuses the
completed fold-0 control and its verified evaluation, resumes the original
no-reflections run with `--resume-from .../weights/last.pt` and
`--disable-reflections`, and queues the remaining paired folds. The overnight
STOP marker is archived only when the guarded coordinator actually starts.
The recovered coordinator raises from signal handlers and reaps children outside
the handler, avoiding the earlier nested-wait deadlock.

Thermal launcher logs/status are in timestamped subdirectories of
`output/retraining/20260927-reflection-ablation/thermal-launches/`.
The resumed training output is
`fold0-no-reflections-training-resume-thermal-20260928.log` in the experiment
directory. Existing failure logs, plans, paused checkpoint backup and original
provenance remain intact. The new protected-input plan includes the integrated
launcher and monitor. It does not rewrite historical experiment provenance.

The prepared command must be run by the user; preparation/preflight is not proof
of active thermal monitoring or resumed training. At that earlier integration, automatic pause based on high
thermal pressure was not implemented; the launcher supervised monitor liveness
and process cleanup. The later integration below supersedes that command.


## Historical per-training automatic thermal pause: September 28

Commit `beba73b` was already integrated as `3b7c8ee` and was skipped. Commit
`92cdc33` was cherry-picked as `4cc0362`. The merge retains the local
`--disable-reflections` option and the new cooperative thermal checkpoints.
All 311 inventoried experiment/run artifacts retained their hashes, including
original provenance and checkpoint backups. Historical plans remain unchanged;
a new resume plan records the authorized code changes and protects the new
launcher, controller, driver, and completed results.

The six runs for folds 0-2 are complete and evaluated. The next run is fold-3
control: its original `weights/last.pt` contains 13 completed epochs. The real
trainer validated the checkpoint, optimizer, original provenance, and matching
early-stopping state with `YOLO.train` intercepted; resumption starts epoch 14.
The remaining queue is fold-3 no-reflections, then both fold-4 arms.

Run this command in the user's foreground terminal:

```bash
bash /Users/andrea/GitHubRepositories/Jarvis/output/retraining/20260927-reflection-ablation/resume-with-auto-pause.sh
```

The coordinator starts directly. Each individual training subprocess is wrapped
in the integrated checkout's `scripts/train-with-thermal.py --auto-pause -- ...`.
Evaluations run separately. Each launcher authenticates sudo in the terminal,
waits for a live pressure sample, and coordinates cooperative pauses with that
trainer. A duplicate-process check and exclusive driver lock precede resumption.
The existing STOP marker is consumed only after protected inputs are verified.

Per-job output is under `thermal-jobs/foldN-arm-TIMESTAMP/`: `training.log`,
`thermal-events.log`, `thermal-control-events.jsonl`, and `status.json`.
The coordinator's `status.json` identifies the active job and log directory.
These paths replace the earlier fixed per-fold training-log paths for this resume.

The 62 fast backend tests and all 29 script tests passed. A launcher test assertion
was corrected to match a complete output line: Python's interrupted traceback
can contain the text of an unexecuted `print`, which is not workload completion.
Preparation also verified that the integrated launcher accepts the exact resume
command and that each training subprocess inherits the terminal.

No training was active at handoff. Non-interactive sudo verification reported
that authentication is required; live auto-pause monitoring and actual training
will begin only after the user runs the command above. No model was promoted.


## Completed comparison - October 2, 2026

Completion: 11:28 Europe/Rome. All ten checkpoint hashes match the retained
per-run evaluation and completed-state records. No training, evaluation, or
thermal-monitor process remained active at verification.

| Held-out component replay, 44 real sources | Reflections enabled | Reflections disabled |
| --- | ---: | ---: |
| Exact whole readings | 20/44 (45.5%) | 30/44 (68.2%) |
| No-read | 6 | 5 |
| Wrong accepted | 18 | 9 |
| MAE on accepted readings | 504.47 | 248.33 |

Each source is evaluated with the digit checkpoint that excluded it from
training. This is the frozen component replay defined above, not a new browser
benchmark or an independent external test. MAE denominators are 38 and 39 accepted
readings, respectively.

Exact readings improve by two in each of the five folds. Paired comparison finds
10 newly correct sources and zero losses of previously correct readings. The
last fold is 6/8 versus 8/8 exact. This supports keeping reflections disabled for
subsequent experiments, while a single seed and repeatedly inspected development
corpus do not establish statistical reliability or authorize promotion.

With human geometry and order, held-out exact readings improve from 24/44 to
33/44. That diagnostic MAE does not improve (133.36 to 138.59); better exact match
does not eliminate rare large errors. Without reflections, the seen-source
oracle score is 171/176 (97.2%) versus 33/44 (75.0%) held-out: a 22.16 percentage
point gap. The 176 seen-source predictions reuse the same photos across models.
The predeclared synthetic-pilot trigger is met, but it does not prove that optical
augmentation or additional generated images will close that gap.

- [Completed comparison](../output/retraining/20260927-reflection-ablation/comparison.json)
- [Controlled optical-variation pilot](full-image-digit-optical-pilot-20261002.md)
