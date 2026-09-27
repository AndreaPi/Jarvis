# Deferred proposal: training and thermal monitoring

Recorded September 16, 2026 at the user's request. Not implemented; thresholds
below are proposals to validate, not active settings or Apple-prescribed timings.

## Shared launcher

Use one launcher for training/resume and the existing
`scripts/monitor-thermal.sh`, including a complete multi-fold experiment.
Authenticate for `powermetrics` and verify monitoring before training starts.
Keep thermal evidence with the experiment logs. Manage `caffeinate` and child
processes together; clean them up on completion, interruption, and failure.
Do not silently start without monitoring: require an explicit opt-out if the
logger cannot start, and visibly report monitoring failures during the run.
Preserve the logger's event-only disk output and optional verbose display.

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

Implement the shared launcher first; add and validate thermal control separately.
This supplements macOS thermal management and is not a hardware-safety guarantee.

Reference: [Apple guidance on thermal-state changes](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/power_efficiency_guidelines_osx/RespondToThermalStateChanges.html).
