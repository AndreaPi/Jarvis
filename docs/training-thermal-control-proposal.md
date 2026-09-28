# Deferred proposal: training and thermal monitoring

Recorded September 16, 2026 at the user's request. The shared launcher was
implemented on the training-thermal-monitor branch on September 28. Automatic
thermal pause/resume below remains a proposal; its thresholds are not active
settings or Apple-prescribed timings.

## Shared launcher

Implemented as `scripts/train-with-thermal.py`. It combines one training command
with the existing
`scripts/monitor-thermal.sh`, including a complete multi-fold experiment.
It authenticates for `powermetrics` and waits for a reported thermal-pressure
level before training starts. It keeps thermal evidence and training output in a
per-run log directory, runs training under `caffeinate`, and stops its children
on completion, interruption, or monitor failure. `--without-monitor` is the
explicit opt-out. The monitor's event-only log is preserved; monitor terminal
output has its own log. See [README](../README.md#thermal-monitoring-for-long-macos-training)
for usage and limits.

## Optional thermal pause/resume

Use the official macOS thermal state for control. Do not assume that
`powermetrics` labels map directly to `ProcessInfo.ThermalState` without checking.
Initial policy to validate on this Mac:

- Mild elevation: log and continue.
- Sustained `serious` for about one minute: request a controlled training pause.
- `critical`: request a pause immediately.
- `nominal` continuously for two minutes: permit automatic resume.
- Repeated short pause/resume cycles: remain paused and request user attention.

Pause cooperatively between batches while retaining training state and keeping
monitoring active. An immediate request does not mean zero-latency GPU stopping.
Avoid freezing the entire process group, which would also stop the monitor.
Specify behavior for missing/stale telemetry and process failures before enabling
unattended control. Verify optimizer/scheduler/early-stopping state, checkpoints,
shutdown, and resume behavior. Log pause/resume reasons and timestamps.

Implement and validate thermal control separately from the shared launcher.
This supplements macOS thermal management and is not a hardware-safety guarantee.

Reference: [Apple guidance on thermal-state changes](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/RespondToThermalStateChanges.html).
