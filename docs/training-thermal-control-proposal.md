# Training thermal monitoring and cooperative pauses

Recorded September 16, 2026. Both phases are implemented on the
`codex/training-thermal-monitor` branch as of September 28.

## Shared launcher

`scripts/train-with-thermal.py` authenticates for `powermetrics` in the user's
terminal, waits for a real pressure sample, and runs training with `caffeinate`.
Training, monitor output and non-nominal events have separate per-run logs.
Completion, interruption (including terminal hangup, `SIGHUP`) and logger
failure trigger process-group cleanup. Repeated stop signals are ignored during
cleanup so they cannot orphan the remaining processes.
`--without-monitor` is the explicit opt-out for this first phase.

## Authentication across a training queue

`scripts/run-training-queue.py -- <driver.py> [arguments]` runs a Python
coordinator in the same interpreter/process after one initial `sudo -v`.
Use the Python interpreter needed by that driver and keep its usual working
directory. `runpy` preserves the coordinator's main entry point, PID, arguments,
locks and signal/child-cleanup ownership; it does not add a parent process that
might be mistaken for a second coordinator. The driver must handle child cleanup
on failure/signals and wait for all jobs before returning. The wrapper converts
`SIGHUP` into an interrupt so the driver can stop its children before sudo renewal
is shut down; repeated hangups are ignored while those cleanups finish.

`scripts/training_auth.py` maintains a queue-scoped session. Its background
worker runs `sudo -n -v` every 60 seconds through training and evaluation gaps,
with a 10-second subprocess timeout and no terminal input/output. The interval
is configurable; sudo policies that forbid reusable credentials cannot be made
unattended by this mechanism. Shutdown stops renewal and waits at most 12 seconds
for an in-flight check. It neither stores passwords nor changes sudoers nor runs
`sudo -k` (which could invalidate the user's other work).

The inherited `JARVIS_TRAINING_AUTH_SESSION` points to atomic
`authentication.json` state. Updated per-fold launchers reject failed, closed,
missing, stale or dead-owner sessions, then validate with `sudo -n -v` before
starting a monitor. They never fall back to an interactive prompt in a queue.
Heartbeat freshness tolerates one renewal interval plus the check timeout and
five seconds; it is a liveness check, not a credential or a security boundary.
If renewal fails, the failure is sticky: an existing monitored training can
finish, but another fold cannot start, even if the sudo cache later recovers.
The wrapper also exits unsuccessfully if the driver returns after renewal failed.
State is available in the queue's authentication log directory independently of
terminal display. Drivers must inherit the session environment when launching
children. Both wrapper and per-fold launcher must be updated before use.

This fixes the between-fold password wait; it does not alter the frozen driver
or a currently running queue. Start/resume with the wrapper after the normal
safe stop. See [queue invocation](../README.md#thermal-monitoring-for-long-macos-training).

## Live terminal output

Training stdout/stderr go directly to `training.log`. A separate reader tails
that file and feeds bounded display queues (128 chunks of up to 4096 characters
and 32 notices of up to 8192 characters). Producers never wait for queue capacity;
overflow replaces older visual updates, while complete log files remain intact.
A dedicated daemon writer owns terminal output. Raw fd writes avoid holding
Python stdout's buffered lock if the terminal blocks during interpreter shutdown.
Display shutdown uses bounded joins. Neither the controller, monitor collector,
training subprocess, nor process cleanup writes directly to the terminal.
Standalone authentication happens interactively before training starts; managed
queues authenticate once before their coordinator starts.

`--quiet` hides the training stream but retains lifecycle/thermal notices. The
launcher enables unbuffered Python output. Progress carriage returns are supported;
notices start on a separate line and cursor-up sequences are suppressed so the
next progress update cannot overwrite an alert. Full-screen terminal apps and
multi-row dashboards are not supported.

Powermetrics pressure notices show the first sample and each changed pressure,
including recovery to Nominal. The monitor is read in verbose mode internally;
`thermal-events.log` still retains only non-nominal events. With `--auto-pause`,
Foundation state/decision changes and observed client pause/resume changes are
also displayed, each with its source and timestamp. Repeated unchanged samples
are not echoed. Client changes are retained in `thermal-control-events.jsonl`.

Tests deliberately block the terminal while checking continuing heartbeats,
a training client's pause acknowledgment, complete log capture, and bounded
shutdown. A separate unconsumed OS-pipe test verifies interpreter exit as well.
Filesystem stalls are outside this terminal-isolation guarantee; the existing
stale-heartbeat behavior still applies.

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
