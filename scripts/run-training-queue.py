#!/usr/bin/env python3
"""Run a Python training coordinator with one queue-wide sudo session."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import runpy
import signal
import sys

from training_auth import AuthenticationError, SudoSession

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--log-dir", type=Path,
                      help="New queue authentication log directory.")
  parser.add_argument("--renew-interval", type=float, default=60,
                      help="Seconds between noninteractive renewals (default: 60).")
  parser.add_argument("driver", nargs=argparse.REMAINDER,
                      help="Python coordinator file and arguments after --.")
  args = parser.parse_args(argv)
  command = args.driver
  if command and command[0] == "--":
    command = command[1:]
  if not command:
    parser.error("Provide a Python coordinator file after -- (not a Python executable).")
  driver = Path(command[0]).resolve()
  if not driver.is_file():
    parser.error(f"Coordinator not found: {driver}")
  directory = (args.log_dir or ROOT / "backend/runs/training-queue" /
               datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
  try:
    session = SudoSession(directory, args.renew_interval)
    directory.mkdir(parents=True, exist_ok=False)
  except (OSError, ValueError) as error:
    parser.error(str(error))
  # Run in this process so the coordinator keeps its PID, signal handlers,
  # child cleanup and lock semantics. Authenticate before it copies os.environ.
  old_argv, old_path = sys.argv, sys.path[:]
  old_sigterm = signal.getsignal(signal.SIGTERM)
  old_sigint = signal.getsignal(signal.SIGINT)
  def interrupted(_signum, _frame):
    raise KeyboardInterrupt
  signal.signal(signal.SIGTERM, interrupted)
  print(f"Queue authentication: {session.path}\nAuthenticate once before the queue starts.", flush=True)
  try:
    with session:
      sys.argv = [str(driver), *command[1:]]
      sys.path.insert(0, str(driver.parent))
      try:
        runpy.run_path(str(driver), run_name="__main__")
      except SystemExit as error:
        if error.code not in (None, 0):
          raise
    return 0
  except AuthenticationError as error:
    print(f"Training queue: {error}", file=sys.stderr)
    return 1
  except KeyboardInterrupt:
    return 130
  finally:
    sys.argv, sys.path[:] = old_argv, old_path
    signal.signal(signal.SIGTERM, old_sigterm)
    signal.signal(signal.SIGINT, old_sigint)


if __name__ == "__main__":
  raise SystemExit(main())
