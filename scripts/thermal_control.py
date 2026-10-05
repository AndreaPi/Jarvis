"""macOS thermal policy and per-run cooperative training control (stdlib only)."""
from __future__ import annotations

import ctypes
import json
import sys
import time
from collections import deque
from pathlib import Path

STATES = ("nominal", "fair", "serious", "critical")


def atomic_json(path: Path, value: dict) -> None:
  temporary = path.with_suffix(".tmp")
  temporary.write_text(json.dumps(value) + "\n")
  temporary.replace(path)


class MacThermalState:
  """Read NSProcessInfo.thermalState; powermetrics pressure labels are separate."""
  def __init__(self):
    if sys.platform != "darwin":
      raise RuntimeError("Automatic thermal control requires macOS.")
    self.foundation = ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
    self.objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    self.objc.objc_getClass.argtypes = [ctypes.c_char_p]
    self.objc.objc_getClass.restype = ctypes.c_void_p
    self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
    self.objc.sel_registerName.restype = ctypes.c_void_p
    send_object = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
      ("objc_msgSend", self.objc))
    self.send_integer = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(
      ("objc_msgSend", self.objc))
    self.process = send_object(self.objc.objc_getClass(b"NSProcessInfo"),
                               self.objc.sel_registerName(b"processInfo"))
    self.selector = self.objc.sel_registerName(b"thermalState")
    if not self.process:
      raise RuntimeError("NSProcessInfo is unavailable.")

  def __call__(self) -> str:
    value = self.send_integer(self.process, self.selector)
    if value not in range(len(STATES)):
      raise RuntimeError(f"Unknown NSProcessInfo thermal state: {value}")
    return STATES[value]


class ThermalPolicy:
  def __init__(self, serious_seconds=60, cool_seconds=120, cycle_window=1800, max_cycles=3):
    self.serious_seconds = serious_seconds
    self.cool_seconds = cool_seconds
    self.cycle_window = cycle_window
    self.max_cycles = max_cycles
    self.hot_since = None
    self.cool_since = None
    self.last_sample = None
    self.paused = False
    self.latched = False
    self.release_requested = False
    self.pauses = deque()

  def update(self, state: str, now: float) -> dict:
    # A missing heartbeat must not count as continuous cooling/heating evidence.
    if self.last_sample is not None and now - self.last_sample > 10:
      self.cool_since = self.hot_since = None
      self.paused = True
    self.last_sample = now
    if state == "nominal":
      if self.cool_since is None:
        self.cool_since = now
    else:
      self.cool_since = None
    if state == "serious":
      if self.hot_since is None:
        self.hot_since = now
    else:
      self.hot_since = None
    unsafe = (state not in STATES or state == "critical" or
              (self.hot_since is not None and now - self.hot_since >= self.serious_seconds))
    if unsafe and not self.paused:
      self.paused = True
      self.pauses.append(now)
      while self.pauses and now - self.pauses[0] > self.cycle_window:
        self.pauses.popleft()
      self.latched = len(self.pauses) >= self.max_cycles
    cooled = self.cool_since is not None and now - self.cool_since >= self.cool_seconds
    if self.paused and cooled and (not self.latched or self.release_requested):
      self.paused = False
      if self.latched:
        self.pauses.clear()
      self.latched = self.release_requested = False
    reason = ("manual_release_required" if self.latched else
              "telemetry_unavailable" if state not in STATES else
              "cooling" if self.paused and not unsafe else
              state if self.paused else "within_policy")
    return {"action": "pause" if self.paused else "run", "state": state,
            "reason": reason, "manual_release_required": self.latched}


class ThermalController:
  def __init__(self, directory: Path, source=None, policy=None, notify=None):
    self.directory = directory
    self.notify = notify
    self.path = directory / "thermal-control.json"
    self.source = source if source is not None else MacThermalState()
    self.policy = policy or ThermalPolicy()
    self.previous = None
    self.last_tick = None
    # Python 3.9 on macOS uses a process-relative origin for time.monotonic().
    # CLOCK_MONOTONIC is shared with trainer processes using newer Python.
    self.started = time.clock_gettime(time.CLOCK_MONOTONIC)
    self.client_phases = {}
    self.participant_seen = False
    self.snapshot = {}
    self.tick(force=True)

  def tick(self, force=False) -> dict:
    now = time.clock_gettime(time.CLOCK_MONOTONIC)
    if not force and self.last_tick is not None and now - self.last_tick < 1:
      return self.snapshot
    self.last_tick = now
    request = self.directory / "thermal-resume.request"
    if request.exists():
      request.unlink()
      if self.policy.latched:
        self.policy.release_requested = True
    error = None
    try:
      state = self.source()
    except Exception as exc:
      state, error = "unavailable", str(exc)
    self.snapshot = {**self.policy.update(state, now), "updated_monotonic": now,
                     "error": error}
    atomic_json(self.path, self.snapshot)
    change = tuple(self.snapshot[key] for key in ("state", "action", "reason", "error"))
    if change != self.previous:
      with (self.directory / "thermal-control-events.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": time.time(), **self.snapshot}) + "\n")
      if self.notify is not None:
        action = "pause requested" if self.snapshot["action"] == "pause" else "run permitted"
        self.notify(f"Thermal state (macOS): {state}; {action} ({self.snapshot['reason']})")
      self.previous = change
    for path in (self.directory / "thermal-clients").glob("*.json"):
      self.participant_seen = True
      try:
        client = json.loads(path.read_text())
        if not 0 <= now - float(client["updated_monotonic"]) <= 10:
          continue
        phase = client["phase"]
        if phase not in ("running", "paused"):
          continue
      except (OSError, ValueError, TypeError, KeyError):
        continue
      previous = self.client_phases.get(path.name)
      if phase != previous and (phase == "paused" or previous == "paused"):
        activity = "resumed" if phase == "running" else "paused"
        message = f"Training {activity} (PID {path.stem}): {client.get('reason', 'unknown')}"
        with (self.directory / "thermal-control-events.jsonl").open("a") as stream:
          stream.write(json.dumps({"time": time.time(), "event": "training_state", **client}) + "\n")
        if self.notify is not None:
          self.notify(message)
      self.client_phases[path.name] = phase
    return self.snapshot

  def check_participant(self, finished=False) -> None:
    self.participant_seen |= any((self.directory / "thermal-clients").glob("*.json"))
    if not self.participant_seen and (finished or time.clock_gettime(time.CLOCK_MONOTONIC) - self.started > 120):
      raise RuntimeError("No cooperative thermal client registered; use an instrumented trainer.")

  def close(self) -> None:
    atomic_json(self.path, {"action": "pause", "state": "unavailable",
                           "reason": "controller_stopped", "updated_monotonic": time.clock_gettime(time.CLOCK_MONOTONIC)})
