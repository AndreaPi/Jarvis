#!/usr/bin/env python3
"""Run a training command with the macOS thermal logger and caffeinate."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from thermal_control import ThermalController

ROOT = Path(__file__).resolve().parents[1]
MONITOR = ROOT / "scripts/monitor-thermal.sh"
CAFFEINATE = "/usr/bin/caffeinate"
READY_LINE = "Current pressure level:"
DEFAULT_READY_TIMEOUT = 60
STOP_TIMEOUT = 8


class LauncherError(RuntimeError):
  pass


def timestamp() -> str:
  return datetime.now(timezone.utc).isoformat()


def write_status(directory: Path, **fields: object) -> None:
  target = directory / "status.json"
  temporary = directory / "status.json.tmp"
  temporary.write_text(json.dumps({"updated_at": timestamp(), **fields}, indent=2) + "\n")
  temporary.replace(target)


def monitor_command(events: Path) -> list[str]:
  return ["/bin/bash", str(MONITOR), "--no-prompt", str(events)]


def authenticate() -> None:
  try:
    subprocess.run(["sudo", "-v"], check=True)
    subprocess.run(["sudo", "-n", "-v"], check=True)
  except (OSError, subprocess.CalledProcessError) as error:
    raise LauncherError("Thermal monitoring authentication failed; training was not started.") from error


def group_alive(process: subprocess.Popen[str]) -> bool:
  # Reap the leader, but do not mistake its exit for all descendants exiting.
  process.poll()
  try:
    os.killpg(process.pid, 0)
  except ProcessLookupError:
    return False
  except PermissionError:
    return True
  return True


def stop_process(process: subprocess.Popen[str] | None, label: str) -> None:
  if process is None:
    return
  for action, timeout in ((signal.SIGINT, STOP_TIMEOUT), (signal.SIGTERM, 3), (signal.SIGKILL, 3)):
    if not group_alive(process):
      return
    try:
      os.killpg(process.pid, action)
    except ProcessLookupError:
      return
    except PermissionError:
      # A surviving privileged sampler must be reported, never hidden by
      # terminating only the unprivileged shell that launched it.
      pass
    deadline = time.monotonic() + timeout
    while group_alive(process):
      if time.monotonic() >= deadline:
        print(f"Waiting for {label} to stop after {action.name}...", file=sys.stderr)
        break
      time.sleep(0.05)
    else:
      return
  if group_alive(process):
    raise LauncherError(f"Could not stop {label}; check process group {process.pid}.")


def collect_monitor(stream, destination, ready: threading.Event, failed: threading.Event) -> None:
  try:
    for line in stream:
      destination.write(line)
      destination.flush()
      if READY_LINE in line:
        ready.set()
  except (OSError, ValueError):
    failed.set()
  finally:
    # EOF also means monitoring is gone, even if its wrapper is still alive.
    failed.set()
    stream.close()


def validate_platform() -> None:
  if sys.platform != "darwin" or not Path(CAFFEINATE).is_file() or not MONITOR.is_file():
    raise LauncherError("This launcher requires macOS, caffeinate, and scripts/monitor-thermal.sh.")


def wait_for_sample(
  monitor: subprocess.Popen[str], ready: threading.Event,
  failed: threading.Event, timeout: float,
) -> None:
  deadline = time.monotonic() + timeout
  while not ready.is_set():
    if failed.is_set() or monitor.poll() is not None:
      raise LauncherError("Thermal monitor exited before the first sample; training was not started.")
    if time.monotonic() >= deadline:
      raise LauncherError("Timed out waiting for the first thermal sample; training was not started.")
    ready.wait(min(0.2, max(0, deadline - time.monotonic())))
  if failed.is_set() or monitor.poll() is not None:
    raise LauncherError("Thermal monitor failed during startup; training was not started.")


def validate_controlled_command(command: list[str]) -> None:
  supported = {ROOT / "backend" / name for name in (
    "train_full_image_digit_detector.py", "train_roi.py", "train_digit_classifier.py",
    "train_strip_digit_reader.py", "train_strip_digit_reader_23xx.py",
  )}
  arguments = command[1:]
  while arguments and arguments[0] in ("-u", "-B"):
    arguments = arguments[1:]
  if (not command or not Path(command[0]).name.startswith("python") or not arguments or
      (ROOT / arguments[0]).resolve() not in supported):
    raise LauncherError("--auto-pause requires a direct Python command for a supported backend/train_*.py trainer; "
                        "wrap each training command separately in multi-fold drivers.")


def run_training(directory: Path, command: list[str], without_monitor: bool, timeout: float,
                 auto_pause: bool = False) -> int:
  monitor: subprocess.Popen[str] | None = None
  controller = None
  children: dict[str, subprocess.Popen[str]] = {}
  monitor_log = None
  collector: threading.Thread | None = None
  ready = threading.Event()
  failed = threading.Event()
  outcome = "failed"
  exit_code: int | None = None
  details = ""
  print(f"Run logs: {directory}", flush=True)
  old_sigterm = signal.getsignal(signal.SIGTERM)
  def interrupted(_signum, _frame):
    raise KeyboardInterrupt
  signal.signal(signal.SIGTERM, interrupted)
  try:
    validate_platform()
    if auto_pause:
      validate_controlled_command(command)
    if auto_pause and without_monitor:
      raise LauncherError("--auto-pause requires thermal monitoring.")
    if not without_monitor:
      print("Authenticating thermal monitor before training...", flush=True)
      authenticate()
      monitor_log = (directory / "thermal-monitor.log").open("w", buffering=1)
      monitor = subprocess.Popen(
        monitor_command(directory / "thermal-events.log"), cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, preexec_fn=os.setpgrp,
      )
      assert monitor.stdout is not None
      collector = threading.Thread(
        target=collect_monitor,
        args=(monitor.stdout, monitor_log, ready, failed), daemon=True,
      )
      collector.start()
      write_status(directory, phase="waiting_for_thermal_sample", monitor_pid=monitor.pid,
                   training_pid=None, thermal_monitor="required")
      wait_for_sample(monitor, ready, failed, timeout)
      print("Thermal sample received; starting training.", flush=True)
      if auto_pause:
        controller = ThermalController(directory)
      exit_code = _run_command(directory, command, monitor, failed, children, controller)
    else:
      print("Thermal monitoring explicitly disabled for this run.", flush=True)
      exit_code = _run_command(directory, command, None, failed, children)
    outcome = "completed" if exit_code == 0 else "training_failed"
    return exit_code
  except LauncherError as error:
    details = str(error)
    print(f"Training launcher: {error}", file=sys.stderr)
    return 1
  except (OSError, ValueError, RuntimeError) as error:
    details = str(error)
    print(f"Training launcher: {error}", file=sys.stderr)
    return 1
  except KeyboardInterrupt:
    outcome = "interrupted"
    details = "Interrupted by user"
    print("Training launcher interrupted; stopping its processes.", file=sys.stderr)
    return 130
  finally:
    # A second Ctrl-C must not interrupt cleanup and orphan the remaining group.
    old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    cleanup_errors = []
    if controller is not None:
      try:
        controller.close()
      except OSError as error:
        cleanup_errors.append(str(error))
    remaining = {}
    for label, process in (("training", children.get("training")), ("thermal monitor", monitor)):
      try:
        stop_process(process, label)
      except (LauncherError, OSError) as error:
        cleanup_errors.append(str(error))
        remaining[label] = process.pid if process is not None else None
    if collector is not None:
      collector.join(timeout=2)
      if collector.is_alive():
        cleanup_errors.append("Thermal output collector did not stop.")
    if monitor_log is not None:
      try:
        monitor_log.close()
      except OSError as error:
        cleanup_errors.append(str(error))
    if cleanup_errors:
      outcome = "cleanup_failed"
      details = "; ".join(filter(None, [details, *cleanup_errors]))
      print(f"Training launcher cleanup: {details}", file=sys.stderr)
    try:
      write_status(directory, phase=outcome, exit_code=exit_code, error=details or None,
                   training_pid=remaining.get("training"), monitor_pid=remaining.get("thermal monitor"),
                   thermal_monitor="disabled" if without_monitor else "required")
    except OSError as error:
      cleanup_errors.append(str(error))
      print(f"Could not save final status: {error}", file=sys.stderr)
    finally:
      signal.signal(signal.SIGINT, old_sigint)
      signal.signal(signal.SIGTERM, old_sigterm)
    if cleanup_errors:
      return 1


def _run_command(directory: Path, command: list[str], monitor: subprocess.Popen[str] | None,
                 failed: threading.Event, children: dict[str, subprocess.Popen[str]],
                 controller: ThermalController | None = None) -> int:
  environment = os.environ.copy()
  # Do not accidentally inherit another run's cooperative control channel.
  environment.pop("JARVIS_THERMAL_CONTROL", None)
  if controller is not None:
    environment["JARVIS_THERMAL_CONTROL"] = str(controller.path)
  last_control_update = None
  with (directory / "training.log").open("w") as training_log:
    training = subprocess.Popen(
      [CAFFEINATE, "-i", *command], cwd=ROOT,
      stdout=training_log, stderr=subprocess.STDOUT, start_new_session=True, env=environment,
    )
    children["training"] = training
    write_status(directory, phase="training", training_pid=training.pid,
                 monitor_pid=monitor.pid if monitor else None,
                 thermal_monitor="disabled" if monitor is None else "required")
    print(f"Training started (PID {training.pid}).", flush=True)
    while training.poll() is None:
      if controller is not None:
        snapshot = controller.tick()
        controller.check_participant()
        if snapshot["updated_monotonic"] != last_control_update:
          write_status(directory, phase="pause_requested" if snapshot["action"] == "pause" else "training",
                       training_pid=training.pid, monitor_pid=monitor.pid if monitor else None,
                       thermal_monitor="required", thermal_control=snapshot)
          last_control_update = snapshot["updated_monotonic"]
      if monitor is not None and (failed.is_set() or monitor.poll() is not None):
        print("Thermal monitor stopped; stopping training.", file=sys.stderr)
        stop_process(training, "training")
        raise LauncherError("Thermal monitor failed during training.")
      time.sleep(0.2)
    if monitor is not None and (failed.is_set() or monitor.poll() is not None):
      raise LauncherError("Thermal monitor failed while training finished.")
    if controller is not None and training.returncode == 0:
      controller.check_participant(finished=True)
    return training.returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--log-dir", type=Path,
                      help="New per-run log directory (default: backend/runs/training-launcher/<timestamp>).")
  parser.add_argument("--without-monitor", action="store_true",
                      help="Explicitly run without powermetrics when monitoring is unavailable.")
  parser.add_argument("--auto-pause", action="store_true",
                      help="Cooperatively pause instrumented trainers using macOS thermal state.")
  parser.add_argument("--monitor-timeout", type=float, default=DEFAULT_READY_TIMEOUT,
                      help="Seconds to wait for a complete first thermal sample (default: 60).")
  parser.add_argument("command", nargs=argparse.REMAINDER,
                      help="Training command after --, for example: -- backend/.venv/bin/python backend/train_roi.py ...")
  args = parser.parse_args(argv)
  if args.command and args.command[0] == "--":
    args.command.pop(0)
  if not args.command:
    parser.error("Provide a training command after --.")
  if not math.isfinite(args.monitor_timeout) or args.monitor_timeout <= 0:
    parser.error("--monitor-timeout must be finite and positive.")
  if args.auto_pause and args.without_monitor:
    parser.error("--auto-pause cannot be combined with --without-monitor.")
  return args


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  directory = args.log_dir or ROOT / "backend/runs/training-launcher" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
  directory = directory.resolve()
  if directory.exists():
    print(f"Training launcher: Log directory already exists: {directory}", file=sys.stderr)
    return 2
  try:
    directory.mkdir(parents=True)
  except OSError as error:
    print(f"Training launcher: Cannot create log directory: {error}", file=sys.stderr)
    return 2
  return run_training(directory, args.command, args.without_monitor, args.monitor_timeout, args.auto_pause)


if __name__ == "__main__":
  raise SystemExit(main())
