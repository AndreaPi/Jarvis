"""Cooperative batch-boundary pauses; no model/optimizer/checkpoint mutation."""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

CONTROL_ENV = "JARVIS_THERMAL_CONTROL"
HEARTBEAT_TIMEOUT = 10


class ThermalGate:
  def __init__(self, path: Path):
    self.path = path
    self.ack = path.parent / "thermal-clients" / f"{os.getpid()}.json"
    self.ack.parent.mkdir(exist_ok=True)
    self.last_ack = 0.0
    self.last_phase = None

  def acknowledge(self, phase: str, reason: str) -> None:
    now = time.clock_gettime(time.CLOCK_MONOTONIC)
    if phase == self.last_phase and now - self.last_ack < 1:
      return
    temporary = self.ack.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": os.getpid(), "phase": phase,
                                     "reason": reason, "updated_monotonic": now}) + "\n")
    temporary.replace(self.ack)
    self.last_ack, self.last_phase = now, phase

  def decision(self) -> tuple[bool, str]:
    try:
      data = json.loads(self.path.read_text())
      # Use the OS clock across different Python versions/processes on macOS.
      age = time.clock_gettime(time.CLOCK_MONOTONIC) - float(data["updated_monotonic"])
      if not math.isfinite(age) or not 0 <= age <= HEARTBEAT_TIMEOUT:
        return False, "stale_telemetry"
      return data["action"] == "run", str(data.get("reason", "unknown"))
    except (OSError, ValueError, TypeError, KeyError):
      return False, "missing_or_invalid_telemetry"

  def wait(self, device=None) -> float:
    started = None
    while True:
      can_run, reason = self.decision()
      if can_run:
        self.acknowledge("running", reason)
        if started is not None:
          elapsed = time.monotonic() - started
          print(f"Thermal training resumed after {elapsed:.1f}s", flush=True)
          return elapsed
        return 0.0
      if started is None:
        started = time.monotonic()
        # Finish already queued accelerator work before reporting a paused batch.
        torch = sys.modules.get("torch")
        device_type = getattr(device, "type", str(device).split(":")[0])
        if torch is not None and device_type == "mps":
          torch.mps.synchronize()
        elif torch is not None and device_type == "cuda":
          torch.cuda.synchronize(device)
        print(f"Thermal training paused: {reason}", flush=True)
      self.acknowledge("paused", reason)
      time.sleep(.2)


_gate = None


def thermal_checkpoint(device=None) -> float:
  global _gate
  path = os.environ.get(CONTROL_ENV)
  if not path:
    return 0.0
  if _gate is None or _gate.path != Path(path) or _gate.ack.stem != str(os.getpid()):
    _gate = ThermalGate(Path(path))
  return _gate.wait(device)


def install_yolo_thermal_callbacks(model) -> None:
  if not os.environ.get(CONTROL_ENV):
    return
  if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    raise RuntimeError("Thermal pauses currently support single-process training only.")
  trainer_ref = []

  def checkpoint(context):
    elapsed = thermal_checkpoint(getattr(context, "device", None))
    trainer = trainer_ref[0] if trainer_ref else context
    # Exclude cooling from Ultralytics' optional wall-clock training budget.
    for name in ("train_time_start", "epoch_time_start"):
      if elapsed and hasattr(trainer, name):
        setattr(trainer, name, getattr(trainer, name) + elapsed)

  def train_start(trainer):
    trainer_ref[:] = [trainer]
    checkpoint(trainer)

  model.add_callback("on_train_start", train_start)
  for event in ("on_pretrain_routine_start", "on_train_batch_start", "on_val_start", "on_val_batch_start"):
    model.add_callback(event, checkpoint)
