import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from thermal_control import ThermalController
from training_console import TrainingConsole

spec = importlib.util.spec_from_file_location('live_launcher', Path(__file__).with_name('train-with-thermal.py'))
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class BlockingTerminal:
  def __init__(self):
    self.entered = threading.Event()
    self.release = threading.Event()

  def isatty(self):
    return True

  def write(self, text):
    self.entered.set()
    self.release.wait()

  def flush(self):
    pass


class TrainingConsoleTests(unittest.TestCase):
  def test_blocked_terminal_does_not_block_control_pause_logs_or_shutdown(self):
    with tempfile.TemporaryDirectory() as tmp:
      directory = Path(tmp)
      terminal = BlockingTerminal()
      display = TrainingConsole(stream=terminal, capacity=2)
      fake = directory / 'caffeinate'
      fake.write_text('#!/usr/bin/env python3\nimport os,sys\nos.execvp(sys.argv[2],sys.argv[2:])\n')
      fake.chmod(0o755)
      critical = directory / 'critical'
      verified = threading.Event()
      stop_observer = threading.Event()
      readings = set()
      monitor = [sys.executable, '-u', '-c',
                 "import time; print('Current pressure level: Nominal',flush=True); time.sleep(30)"]
      command = [sys.executable, '-c',
                 "import time; from pathlib import Path; from backend.thermal_pause import thermal_checkpoint; "
                 "thermal_checkpoint(); print('PAYLOAD-'*20000); Path(" + repr(str(critical)) + ").touch(); "
                 "exec('while True:\\n thermal_checkpoint()\\n time.sleep(.05)')"]
      def controller(run, **kwargs):
        return ThermalController(run, source=lambda: 'critical' if critical.exists() else 'nominal', **kwargs)
      def observe():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not stop_observer.is_set():
          try:
            state = json.loads((directory / 'thermal-control.json').read_text())
            if state['action'] == 'pause':
              readings.add(state['updated_monotonic'])
              clients = [json.loads(p.read_text()) for p in (directory/'thermal-clients').glob('*.json')]
              if len(readings) >= 2 and any(c['phase'] == 'paused' for c in clients) and terminal.entered.is_set():
                verified.set()
                os.kill(os.getpid(), signal.SIGTERM)
                return
          except (OSError, ValueError):
            pass
          time.sleep(.03)
        if not stop_observer.is_set():
          terminal.release.set()
          os.kill(os.getpid(), signal.SIGTERM)
      observer = threading.Thread(target=observe)
      failsafe = threading.Timer(8, terminal.release.set)
      with patch.object(sys, 'stdout', terminal), patch.object(sys, 'stderr', terminal), \
           patch.object(launcher, 'TrainingConsole', return_value=display), \
           patch.object(launcher, 'validate_platform'), patch.object(launcher, 'authenticate'), \
           patch.object(launcher, 'validate_controlled_command'), \
           patch.object(launcher, 'monitor_command', return_value=monitor), \
           patch.object(launcher, 'ThermalController', side_effect=controller), \
           patch.object(launcher, 'CAFFEINATE', str(fake)):
        observer.start(); failsafe.start()
        try:
          result = launcher.run_training(directory, command, False, 2, auto_pause=True)
          self.assertTrue(verified.is_set(), 'Thermal decisions/acknowledgment stalled with the terminal')
          self.assertFalse(terminal.release.is_set(), 'Shutdown waited for terminal recovery')
          self.assertEqual(result, 130)
          self.assertGreater(display.dropped, 0)
          self.assertLessEqual(display.progress.qsize(), 2)
          log = (directory/'training.log').read_text()
          self.assertIn('PAYLOAD-'*20000, log)
          self.assertIn('Thermal training paused', log)
          self.assertEqual(json.loads((directory/'status.json').read_text())['phase'], 'interrupted')
        finally:
          stop_observer.set(); terminal.release.set(); failsafe.cancel(); observer.join(timeout=6)
          if display.writer: display.writer.join(timeout=2)

  def test_progress_notices_quiet_mode_and_closed_terminal(self):
    for quiet, tty in ((False, False), (False, True), (True, True)):
      with self.subTest(quiet=quiet, tty=tty):
        stream = io.StringIO()
        stream.isatty = lambda: tty
        display = TrainingConsole(quiet=quiet, stream=stream)
        display.start()
        display.output('\rEpoch 1 10%')
        deadline = time.monotonic()+1
        while not quiet and not stream.getvalue() and time.monotonic() < deadline:
          time.sleep(.01)
        display.notice('Thermal pressure: Heavy')
        deadline = time.monotonic()+1
        while 'Heavy' not in stream.getvalue() and time.monotonic() < deadline:
          time.sleep(.01)
        display.output('\x1b[')
        display.output('A\rEpoch 1 20%\n')
        display.close('Finished')
        output = stream.getvalue()
        self.assertIn('Thermal pressure: Heavy\n', output)
        self.assertNotIn('\x1b[A', output)
        self.assertEqual('Epoch' in output, not quiet)
        self.assertTrue(output.endswith('Finished\n'))
        if not quiet:
          self.assertIn('10%\n[', output)
    class ClosedTerminal(io.StringIO):
      def write(self, value): raise BrokenPipeError('closed terminal')
    display = TrainingConsole(stream=ClosedTerminal())
    display.notice('message'); display.start(); display.close()
    self.assertTrue(display.unavailable.is_set())

  def test_full_output_pipe_does_not_delay_exit_or_truncate_training_log(self):
    with tempfile.TemporaryDirectory() as tmp:
      directory = Path(tmp)
      fake = directory / 'caffeinate'
      fake.write_text('#!/usr/bin/env python3\nimport os,sys\nos.execvp(sys.argv[2],sys.argv[2:])\n')
      fake.chmod(0o755)
      code = (
        "import importlib.util,sys; sys.path.insert(0,sys.argv[1]); "
        "spec=importlib.util.spec_from_file_location('launcher',sys.argv[1]+'/train-with-thermal.py'); "
        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        "m.validate_platform=lambda:None; m.CAFFEINATE=sys.argv[2]; "
        "raise SystemExit(m.main(['--without-monitor','--log-dir',sys.argv[3],'--',sys.executable,'-c',"
        "\"import os,time; os.write(1,b'x'*2000000); time.sleep(.4)\"]))"
      )
      process = subprocess.Popen([sys.executable, '-B', '-c', code, str(Path(__file__).parent),
                                  str(fake), str(directory/'run')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      try:
        # Deliberately do not drain stdout until the launcher has exited.
        self.assertEqual(process.wait(timeout=5), 0)
        output, errors = process.communicate(timeout=2)
        self.assertGreater(len(output), 4096)
        self.assertLess(len(output), 2000000)
        self.assertEqual((directory/'run/training.log').read_bytes(), b'x'*2000000)
      finally:
        if process.poll() is None: process.kill(); process.wait()
        process.stdout.close(); process.stderr.close()

  def test_monitor_reports_transitions_including_recovery_without_repetition(self):
    stream = io.StringIO(''.join('Current pressure level: '+state+'\n'
                               for state in ('Nominal','Heavy','Heavy','Nominal')))
    log, screen = io.StringIO(), io.StringIO()
    display = TrainingConsole(stream=screen)
    ready, failed = threading.Event(), threading.Event()
    launcher.collect_monitor(stream, log, ready, failed, display)
    display.close()
    self.assertEqual(screen.getvalue().count('Heavy'), 1)
    self.assertEqual(screen.getvalue().count('Nominal'), 2)
    self.assertEqual(log.getvalue().count('Heavy'), 2)
    self.assertTrue(ready.is_set())


if __name__ == '__main__':
  unittest.main()
