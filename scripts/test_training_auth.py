from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from training_auth import (
  AuthenticationError, SESSION_ENV, SudoSession, authenticate_training,
  check_session, validate_sudo,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-training-queue.py"


class TrainingAuthenticationTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.root = Path(self.temp.name)
    self.environment = patch.dict(os.environ)
    self.environment.start()
    self.addCleanup(self.environment.stop)
    os.environ.pop(SESSION_ENV, None)

  def test_single_initial_prompt_renewal_across_gaps_and_cleanup(self):
    with patch("training_auth.validate_sudo") as sudo:
      with SudoSession(self.root, interval=.02) as session:
        path = os.environ[SESSION_ENV]
        authenticate_training()  # first fold
        deadline = time.monotonic() + 2
        while sudo.call_count < 6 and time.monotonic() < deadline:
          time.sleep(.01)  # evaluation gap without a thermal launcher
        self.assertGreaterEqual(sudo.call_count, 6)
        authenticate_training()  # second fold
        check_session(path)
      self.assertNotIn(SESSION_ENV, os.environ)
      calls = sudo.call_count
      time.sleep(.06)
      self.assertEqual(sudo.call_count, calls)
      self.assertFalse(session.worker.is_alive())
      self.assertEqual(sum(c.kwargs.get("interactive", False) for c in sudo.call_args_list), 1)
      self.assertEqual(json.loads(Path(path).read_text())["phase"], "closed")

  def test_lost_renewal_blocks_next_fold_even_if_cache_recovers(self):
    with patch("training_auth.validate_sudo") as sudo:
      with self.assertRaisesRegex(AuthenticationError, "revoked"):
        with SudoSession(self.root, interval=.02) as session:
          sudo.side_effect = AuthenticationError("revoked")
          session.worker.join(timeout=2)
          self.assertFalse(session.worker.is_alive())
          sudo.side_effect = None
          calls = sudo.call_count
          with self.assertRaises(AuthenticationError):
            authenticate_training()
          self.assertEqual(sudo.call_count, calls)
      self.assertNotIn(SESSION_ENV, os.environ)
      self.assertEqual(json.loads(session.path.read_text())["phase"], "failed")

  def test_missing_stale_dead_closed_and_invalid_sessions_never_prompt(self):
    path = self.root / "auth.json"
    state = {"phase": "active", "pid": os.getpid(), "updated_at": time.time(), "max_age": 30}
    cases = [None, {**state, "updated_at": 0}, {**state, "phase": "closed"},
             {**state, "pid": -1}, {}, {**state, "pid": 99999999}]
    with patch.dict(os.environ, {SESSION_ENV: str(path)}), patch("training_auth.validate_sudo") as sudo:
      for value in cases:
        with self.subTest(value=value):
          if value is not None:
            path.write_text(json.dumps(value))
          with self.assertRaises(AuthenticationError):
            authenticate_training()
      sudo.assert_not_called()

  def test_noninteractive_checks_have_timeout_and_no_terminal_io(self):
    for error in (None, subprocess.TimeoutExpired("sudo", 10), subprocess.CalledProcessError(1, "sudo")):
      with self.subTest(error=error), patch("training_auth.subprocess.run", side_effect=error) as run:
        if error is None:
          validate_sudo()
        else:
          with self.assertRaises(AuthenticationError):
            validate_sudo()
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["sudo", "-n", "-v"])
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
        self.assertGreater(kwargs["timeout"], 0)

  def test_initial_failure_and_driver_exception_stop_renewal(self):
    for failure in (True, False):
      with self.subTest(initial_failure=failure):
        directory = self.root / str(failure)
        directory.mkdir()
        with patch("training_auth.validate_sudo", side_effect=AuthenticationError("denied") if failure else None):
          session = SudoSession(directory, interval=.02)
          with self.assertRaisesRegex(RuntimeError, "denied|driver failed"):
            with session:
              raise RuntimeError("driver failed")
          self.assertNotIn(SESSION_ENV, os.environ)
          if session.worker:
            self.assertFalse(session.worker.is_alive())
          with self.assertRaises(AuthenticationError):
            check_session(session.path)

  def test_cli_runs_coordinator_in_same_process_and_cleans_up_on_signal(self):
    # Fake sudo expires between folds without renewal. No privileged commands.
    fake = self.root / "sudo"
    fake.write_text("#!" + sys.executable + "\n" + '''
import json,os,sys,time
from pathlib import Path
root=Path(os.environ['AUTH_FIXTURE'])
with (root/'calls').open('a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
cache=root/'cache'
if '-n' in sys.argv and (not cache.exists() or time.time()-float(cache.read_text()) > .5):
  sys.exit(1)
cache.write_text(str(time.time()))
''')
    fake.chmod(0o755)
    driver = self.root / "driver.py"
    driver.write_text('''import os,signal,subprocess,sys,time
from pathlib import Path
from training_auth import authenticate_training
root=Path(os.environ['AUTH_FIXTURE'])
(root/'pid').write_text(str(os.getpid()))
assert sys.argv[1:]==['fixture-arg']
authenticate_training()
time.sleep(.9)
authenticate_training()
if os.environ.get('WAIT_FOR_SIGNAL'):
  child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
  (root/'child-pid').write_text(str(child.pid))
  try:
    (root/'second-fold').touch()
    time.sleep(30)
  finally:
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    if int(os.environ['WAIT_FOR_SIGNAL'])==signal.SIGHUP:
      os.kill(os.getpid(),signal.SIGHUP)
    child.terminate()
    child.wait(timeout=3)
    (root/'driver-cleaned').touch()
else:
  (root/'second-fold').touch()
raise SystemExit(0)
''')
    for signum in (None, signal.SIGTERM, signal.SIGHUP):
      with self.subTest(signum=signum):
        directory = self.root / str(signum)
        marker = self.root / "second-fold"
        marker.unlink(missing_ok=True)
        for name in ("driver-cleaned", "child-pid"):
          (self.root / name).unlink(missing_ok=True)
        env = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ["PATH"],
                   AUTH_FIXTURE=str(self.root), WAIT_FOR_SIGNAL=str(int(signum)) if signum else "")
        process = subprocess.Popen([sys.executable, str(RUNNER), "--log-dir", str(directory),
                                    "--renew-interval", ".08", "--", str(driver), "fixture-arg"],
                                   env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
          deadline = time.monotonic() + 6
          while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
          self.assertTrue(marker.exists())
          self.assertEqual(int((self.root / "pid").read_text()), process.pid)
          if signum:
            process.send_signal(signum)
          _, stderr = process.communicate(timeout=5)
          self.assertEqual(process.returncode, 130 if signum else 0, stderr)
          if signum:
            self.assertTrue((self.root / "driver-cleaned").exists())
            with self.assertRaises(ProcessLookupError):
              os.kill(int((self.root / "child-pid").read_text()), 0)
          self.assertEqual(json.loads((directory / "authentication.json").read_text())["phase"], "closed")
        finally:
          if process.poll() is None:
            process.kill()
            process.wait()
          if (self.root / "child-pid").exists():
            try:
              os.kill(int((self.root / "child-pid").read_text()), signal.SIGKILL)
            except ProcessLookupError:
              pass
          if process.stderr:
            process.stderr.close()
    calls = [json.loads(line) for line in (self.root / "calls").read_text().splitlines()]
    self.assertEqual(calls.count(["-v"]), 3)  # exactly once per queue invocation
    self.assertGreater(calls.count(["-n", "-v"]), 10)


if __name__ == "__main__":
  unittest.main()
