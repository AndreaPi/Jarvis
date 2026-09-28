from __future__ import annotations

import importlib.util
import json
import os
import signal
import threading
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().with_name("train-with-thermal.py")
spec = importlib.util.spec_from_file_location("train_with_thermal", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(launcher)


def monitor_program(kind: str) -> list[str]:
  if kind == "ready":
    body = "import time; print('*** Sampled system activity', flush=True); print('Current pressure level: Nominal', flush=True); time.sleep(30)"
  elif kind == "exits":
    body = "print('monitor failed', flush=True)"
  elif kind == "header_only":
    body = "import time; print('*** Sampled system activity', flush=True); time.sleep(30)"
  elif kind == "closed_stream":
    body = "import os,time; print('Current pressure level: Nominal', flush=True); time.sleep(.4); os.close(1); os.close(2); time.sleep(30)"
  else:
    body = "import time; print('*** Sampled system activity', flush=True); print('Current pressure level: Nominal', flush=True); time.sleep(.4)"
  return [sys.executable, "-u", "-c", body]


class TrainingLauncherTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    wrapper = self.root / "fake-caffeinate"
    wrapper.write_text("#!/usr/bin/env python3\nimport os,sys\nos.execvp(sys.argv[2],sys.argv[2:])\n")
    wrapper.chmod(0o755)
    self.patches = [
      patch.object(launcher, "CAFFEINATE", str(wrapper)),
      patch.object(launcher, "validate_platform"),
      patch.object(launcher, "authenticate"),
    ]
    for item in self.patches:
      item.start()
      self.addCleanup(item.stop)

  def run_case(self, kind: str, command: list[str], without_monitor=False):
    directory = self.root / kind
    directory.mkdir()
    with patch.object(launcher, "monitor_command", return_value=monitor_program(kind)):
      result = launcher.run_training(directory, command, without_monitor, 2)
    return result, directory, json.loads((directory / "status.json").read_text())

  def test_ready_sample_precedes_training_and_logs_are_retained(self):
    result, directory, state = self.run_case(
      "ready", [sys.executable, "-c", "print('trained')"])
    self.assertEqual(result, 0)
    self.assertEqual(state["phase"], "completed")
    self.assertEqual(state["thermal_monitor"], "required")
    self.assertIn("trained", (directory / "training.log").read_text())
    self.assertIn("*** Sampled system activity", (directory / "thermal-monitor.log").read_text())

  def test_sample_header_without_pressure_does_not_start_training(self):
    directory = self.root / "header_only"
    directory.mkdir()
    with patch.object(launcher, "monitor_command", return_value=monitor_program("header_only")):
      result = launcher.run_training(directory, [sys.executable, "-c", "print('must not run')"], False, .3)
    self.assertEqual(result, 1)
    self.assertFalse((directory / "training.log").exists())
    self.assertEqual(json.loads((directory / "status.json").read_text())["phase"], "failed")

  def test_authentication_failure_prevents_training(self):
    directory = self.root / "auth-failed"
    directory.mkdir()
    with patch.object(launcher, "authenticate", side_effect=launcher.LauncherError("authentication failed")):
      result = launcher.run_training(directory, [sys.executable, "-c", "print('must not run')"], False, 2)
    self.assertEqual(result, 1)
    self.assertFalse((directory / "training.log").exists())
    self.assertEqual(json.loads((directory / "status.json").read_text())["phase"], "failed")

  def test_monitor_failure_blocks_or_stops_training(self):
    for kind in ("exits", "fails_later", "closed_stream"):
      with self.subTest(kind=kind):
        result, directory, state = self.run_case(
          kind, [sys.executable, "-c", "import time; time.sleep(30)"])
        self.assertEqual(result, 1)
        self.assertEqual(state["phase"], "failed")
        self.assertIn("monitor", state["error"].lower())
        if kind == "exits":
          self.assertFalse((directory / "training.log").exists())
        else:
          self.assertIsNone(state["training_pid"])

  def test_explicit_opt_out_propagates_training_exit_code(self):
    result, directory, state = self.run_case(
      "without", [sys.executable, "-c", "raise SystemExit(7)"], without_monitor=True)
    self.assertEqual(result, 7)
    self.assertEqual(state["phase"], "training_failed")
    self.assertEqual(state["thermal_monitor"], "disabled")
    self.assertFalse((directory / "thermal-monitor.log").exists())

  def test_cleanup_attempts_monitor_even_if_training_cleanup_fails(self):
    directory = self.root / "cleanup-failed"
    directory.mkdir()
    labels = []
    def fail_first(_process, label):
      labels.append(label)
      if label == "training":
        raise launcher.LauncherError("training cleanup failed")
    with patch.object(launcher, "stop_process", side_effect=fail_first):
      result = launcher.run_training(directory, [sys.executable, "-c", "pass"], True, 2)
    self.assertEqual(result, 1)
    self.assertEqual(labels, ["training", "thermal monitor"])
    state = json.loads((directory / "status.json").read_text())
    self.assertEqual(state["phase"], "cleanup_failed")
    self.assertIsInstance(state["training_pid"], int)

  def test_interrupt_stops_training_group(self):
    directory = self.root / "interrupted"
    directory.mkdir()
    pid_file = directory / "training.pid"
    command = "import os,time; from pathlib import Path; Path(" + repr(str(pid_file)) + ").write_text(str(os.getpid())); time.sleep(30)"
    timer = threading.Timer(.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    original_stop = launcher.stop_process
    def repeated_interrupt(process, label):
      os.kill(os.getpid(), signal.SIGTERM)
      os.kill(os.getpid(), signal.SIGINT)
      original_stop(process, label)
    try:
      with patch.object(launcher, "stop_process", side_effect=repeated_interrupt):
        result = launcher.run_training(
          directory, [sys.executable, "-c", command], True, 2)
    finally:
      timer.cancel()
    self.assertEqual(result, 130)
    with self.assertRaises(ProcessLookupError):
      os.kill(int(pid_file.read_text()), 0)
    self.assertEqual(json.loads((directory / "status.json").read_text())["phase"], "interrupted")

  def test_cleanup_drains_group_even_after_leader_exits(self):
    for leader_exited in (False, True):
      with self.subTest(leader_exited=leader_exited):
        leader = subprocess.Popen(
          [sys.executable, "-c", "import time; time.sleep(30)"],
          preexec_fn=os.setpgrp, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        member = None
        reaper = None
        try:
          # Keep the fixture in our session so the test can reap both members.
          # The launcher must stop the entire group, regardless of parentage.
          member = subprocess.Popen(
            [sys.executable, "-u", "-c",
             "import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); print('ready'); time.sleep(30)"],
            preexec_fn=lambda: os.setpgid(0, leader.pid), stdout=subprocess.PIPE, text=True)
          self.assertEqual(member.stdout.readline().strip(), "ready")
          reaper = threading.Thread(target=member.wait)
          reaper.start()
          if leader_exited:
            leader.terminate()
            leader.wait(timeout=2)
          with patch.object(launcher, "STOP_TIMEOUT", .15):
            launcher.stop_process(leader, "fixture")
          reaper.join(timeout=2)
          self.assertFalse(reaper.is_alive())
          self.assertEqual(member.returncode, -signal.SIGTERM)
          self.assertFalse(launcher.group_alive(leader))
        finally:
          try:
            os.killpg(leader.pid, signal.SIGKILL)
          except ProcessLookupError:
            pass
          leader.wait(timeout=2)
          if member is not None:
            member.wait(timeout=2)
            member.stdout.close()
          if reaper is not None:
            reaper.join(timeout=2)

  def test_invalid_timeouts_are_rejected(self):
    for value in ("nan", "inf", "0", "-1"):
      with self.subTest(value=value), self.assertRaises(SystemExit) as error:
        launcher.parse_args(["--monitor-timeout", value, "--", "true"])
      self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
  unittest.main()
