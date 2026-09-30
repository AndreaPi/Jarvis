"""Queue-scoped sudo renewal; no password handling or terminal I/O in the worker."""
from __future__ import annotations

import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path

SESSION_ENV = "JARVIS_TRAINING_AUTH_SESSION"
CHECK_TIMEOUT = 10


class AuthenticationError(RuntimeError):
  pass


def validate_sudo(interactive=False):
  command = ["sudo", "-v"] if interactive else ["sudo", "-n", "-v"]
  options = {} if interactive else {
    "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
    "stderr": subprocess.DEVNULL, "timeout": CHECK_TIMEOUT,
  }
  try:
    subprocess.run(command, check=True, **options)
  except (OSError, subprocess.SubprocessError) as error:
    raise AuthenticationError(
      "Sudo authentication unavailable; restart the queue from your terminal. "
      "No training was started by this authentication check."
    ) from error


def check_session(path):
  try:
    state = json.loads(Path(path).read_text())
    if state["phase"] != "active":
      raise ValueError(state.get("error") or state["phase"])
    age = time.time() - state["updated_at"]
    if not 0 <= age <= state["max_age"]:
      raise ValueError("queue authentication heartbeat expired")
    pid = state["pid"]
    if not isinstance(pid, int) or pid <= 0:
      raise ValueError("invalid queue owner")
    os.kill(pid, 0)
  except (OSError, ValueError, KeyError, TypeError) as error:
    raise AuthenticationError(
      f"Queue authentication is unavailable ({path}): {error}. "
      "Restart the queue from your terminal; no password will be requested here."
    ) from error


def authenticate_training():
  session = os.environ.get(SESSION_ENV)
  if session is not None:
    check_session(session)
  else:
    validate_sudo(interactive=True)
  validate_sudo()
  if session is not None:
    check_session(session)


class SudoSession:
  """Wrap a whole queue, including evaluations; children inherit its session file.

  Use in the main thread. The coordinator remains responsible for waiting for
  children and cleaning them up on exceptions/signals before leaving this scope.
  """
  def __init__(self, directory: Path, interval=60):
    if not math.isfinite(interval) or interval <= 0:
      raise ValueError("Sudo renewal interval must be finite and positive")
    self.path = directory.resolve() / "authentication.json"
    self.interval = interval
    self.stopping = threading.Event()
    self.worker = None
    self.failure = None

  def _publish(self, phase):
    state = {"phase": phase, "pid": os.getpid(), "updated_at": time.time(),
             "max_age": self.interval + CHECK_TIMEOUT + 5, "error": self.failure}
    temporary = self.path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(self.path)

  def __enter__(self):
    if SESSION_ENV in os.environ:
      raise AuthenticationError("Nested training authentication sessions are not supported")
    self._publish("authenticating")
    try:
      validate_sudo(interactive=True)
      validate_sudo()
      self._publish("active")
      self.worker = threading.Thread(target=self._renew, daemon=True)
      self.worker.start()
      os.environ[SESSION_ENV] = str(self.path)
    except BaseException as error:
      self.stopping.set()
      self.failure = str(error) or type(error).__name__
      self._publish("failed")
      raise
    return self

  def _renew(self):
    while not self.stopping.wait(self.interval):
      try:
        validate_sudo()
        self._publish("active")
      except (AuthenticationError, OSError) as error:
        self.failure = str(error)
        try:
          self._publish("failed")
        except OSError:
          pass  # Readers also reject the stale heartbeat if storage fails.
        return

  def __exit__(self, kind, value, traceback):
    self.stopping.set()
    if self.worker is not None:
      self.worker.join(CHECK_TIMEOUT + 2)
      if self.worker.is_alive():
        self.failure = self.failure or "Sudo renewal worker did not stop"
    os.environ.pop(SESSION_ENV, None)
    try:
      self._publish("failed" if self.failure else "closed")
    except OSError:
      if kind is None:
        raise
    if kind is None and self.failure:
      raise AuthenticationError(self.failure)
