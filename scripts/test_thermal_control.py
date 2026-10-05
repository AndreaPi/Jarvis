import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from thermal_control import ThermalController, ThermalPolicy

SPEC = importlib.util.spec_from_file_location("controlled_launcher", Path(__file__).with_name("train-with-thermal.py"))
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)
ROOT = Path(__file__).resolve().parents[1]


class ThermalControlTests(unittest.TestCase):
  def test_policy_hysteresis_missing_samples_and_repeated_cycles(self):
    policy = ThermalPolicy(serious_seconds=2, cool_seconds=2, max_cycles=3)
    for now, state, action in (
      (0, "fair", "run"), (1, "serious", "run"), (2, "serious", "run"),
      (3, "serious", "pause"), (4, "nominal", "pause"), (5, "fair", "pause"),
      (6, "nominal", "pause"), (7, "nominal", "pause"), (8, "nominal", "run"),
      (9, "critical", "pause"), (10, "nominal", "pause"), (11, "nominal", "pause"),
      (12, "nominal", "run"), (13, "critical", "pause"),
      (14, "nominal", "pause"), (15, "nominal", "pause"), (16, "nominal", "pause"),
    ):
      with self.subTest(now=now):
        self.assertEqual(policy.update(state, now)["action"], action)
    self.assertTrue(policy.latched)
    policy.release_requested = True
    self.assertEqual(policy.update("critical", 17)["action"], "pause")
    for now in (18, 19):
      self.assertEqual(policy.update("nominal", now)["action"], "pause")
    self.assertEqual(policy.update("nominal", 20)["action"], "run")
    self.assertFalse(policy.latched)
    self.assertEqual(policy.update("unavailable", 21)["action"], "pause")
    self.assertEqual(policy.update("nominal", 22)["action"], "pause")
    self.assertEqual(policy.update("nominal", 40)["action"], "pause")
    self.assertEqual(policy.update("nominal", 41)["action"], "pause")
    self.assertEqual(policy.update("nominal", 42)["action"], "run")

  def test_launcher_cooperates_and_rejects_uninstrumented_commands(self):
    # Reject unsupported commands before authentication or subprocess creation.
    with self.assertRaises(launcher.LauncherError):
      launcher.validate_controlled_command([sys.executable, "-c", "pass"])
    launcher.validate_controlled_command([sys.executable, "-u", "backend/train_full_image_digit_detector.py"])
    for mode in ("resume", "monitor_failure", "unsupported"):
      instrumented = mode != "unsupported"
      with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        fake = directory / "caffeinate"
        fake.write_text("#!/usr/bin/env python3\nimport os,sys\nos.execvp(sys.argv[2],sys.argv[2:])\n")
        fake.chmod(0o755)
        monitor = [sys.executable, "-u", "-c",
                   "import time; print('Current pressure level: Nominal',flush=True); time.sleep(" +
                   (".5" if mode == "monitor_failure" else "30") + ")"]
        def controller(run, **kwargs):
          start = time.monotonic()
          return ThermalController(run, source=lambda: "critical" if mode == "monitor_failure" or time.monotonic()-start < .5 else "nominal",
                                   policy=ThermalPolicy(cool_seconds=0), **kwargs)
        command = [sys.executable, "-u", "-c",
                   "from backend.thermal_pause import thermal_checkpoint; thermal_checkpoint(); print('completed workload')"
                   if instrumented else "print('unsupported workload')"]
        with patch.object(launcher, "CAFFEINATE", str(fake)), \
             patch.object(launcher, "authenticate"), patch.object(launcher, "validate_platform"), \
             patch.object(launcher, "monitor_command", return_value=monitor), \
             patch.object(launcher, "ThermalController", side_effect=controller), \
             patch.object(launcher, "validate_controlled_command"):
          result = launcher.run_training(directory, command, False, 2, auto_pause=True)
        status = json.loads((directory / "status.json").read_text())
        if mode == "monitor_failure":
          self.assertEqual(result, 1, status)
          self.assertIn("Thermal monitor", status["error"])
          output = (directory / "training.log").read_text()
          self.assertIn("Thermal training paused", output)
          self.assertNotIn("completed workload", output.splitlines())
        elif instrumented:
          self.assertEqual(result, 0, status)
          output = (directory / "training.log").read_text()
          self.assertIn("Thermal training paused", output)
          self.assertIn("Thermal training resumed", output)
          self.assertIn("completed workload", output.splitlines())
          self.assertEqual(json.loads((directory / "thermal-control.json").read_text())["action"], "pause")
        else:
          self.assertEqual(result, 1, status)
          self.assertIn("No cooperative thermal client", status["error"])


if __name__ == "__main__":
  unittest.main()
