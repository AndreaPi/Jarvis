# Training thermal monitoring and cooperative pauses

Recorded September 16, 2026. Both phases are implemented on the
`codex/training-thermal-monitor` branch as of September 28.

## Shared launcher

`scripts/train-with-thermal.py` authenticates for `powermetrics` in the user's
terminal, waits for a real pressure sample, and runs training with `caffeinate`.
Training, monitor output and non-nominal events have separate per-run logs.
Completion, interruption and logger failure trigger process-group cleanup.
`--without-monitor` is the explicit opt-out for this first phase.

## Opt-in thermal pause/resume

Use `--auto-pause` with a direct Python invocation of one of the five instrumented
`backend/train_*.py` scripts. The launcher rejects unsupported commands before
starting them. Multi-fold drivers must wrap each training subprocess separately.
No change is made to an already-running experiment or to its frozen provenance.

The controller polls the official `NSProcessInfo.thermalState` once per second
via Foundation, without sudo or an additional Python dependency. The separate
powermetrics event logger still runs. Its pressure labels are not mapped to
Foundation's `nominal`, `fair`, `serious`, and `critical` states.

Default policy:

- `fair`: continue and record the state change.
- `serious` continuously for 60 seconds: request a pause.
- `critical`: request a pause at the next poll.
- Missing/invalid native telemetry: request a pause.
- After any pause, require 120 continuous seconds of `nominal` to resume.
- Three pauses within 30 minutes: latch a hold requiring human attention.
  `touch <run-log-directory>/thermal-resume.request` authorizes release, but
  never bypasses the normal-state cooling period.

These timings are Jarvis policy, not Apple-prescribed hardware safety limits.
Pauses take effect at a cooperative batch boundary, after synchronizing queued
MPS/CUDA work. Current batch/preprocessing work and bounded data-loader prefetch
can finish first. Validation is covered too. Monitor and caffeinate remain active;
no process group is frozen. CPU/MPS single-process training is the supported path.

Weights, accumulated gradients, optimizer, scheduler, RNG and early-stopping
state remain in memory. Checkpoints are not rewritten by a thermal pause.
Ultralytics' wall-clock training budget excludes pause time; custom PyTorch
training reports retain their elapsed wall time including cooling. On process
termination or power loss, the normal saved-checkpoint resume rules still apply.

## Evidence and failures

`thermal-control.json` publishes an atomic decision and heartbeat. The shared
OS monotonic clock works across macOS system Python 3.9 and newer training
interpreters. A trainer pauses if the heartbeat is older than 10 seconds or the
file is unreadable/invalid. Unknown native states and polling gaps reset the
cooling interval. A stalled/dead launcher therefore cannot silently authorize
further batches; it must recover and publish a fresh decision or the user must
stop the remaining trainer. Failure of the powermetrics logger retains the
first-phase behavior: stop the training process group.

`status.json` reports a pause request, while `thermal-clients/<pid>.json` records
whether the trainer has reached a paused boundary. State/decision changes are
retained in `thermal-control-events.jsonl`; pause/resume messages appear in
`training.log`. No registered client within 120 seconds, or successful command
completion without registration, is reported as failure.

Tests replay severe states without heating the Mac, exercise repeated cycles
and stale telemetry, verify interruption while paused, and compare deterministic
CPU optimizer steps with and without a pause. They check unchanged weights,
accumulated gradients, optimizer, scheduler, RNG, early-stopping metadata and
checkpoint bytes across the pause, plus identical continuation. The native
nominal-state reader is also checked on macOS; simulated critical states are
not evidence of a live overheating event or hardware protection certification.

See [launcher usage](../README.md#thermal-monitoring-for-long-macos-training).
Reference: [Apple ProcessInfo thermal states](https://developer.apple.com/documentation/foundation/processinfo/thermalstate-swift.enum).
