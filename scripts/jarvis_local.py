#!/usr/bin/env python3
"""Start and stop the local Jarvis frontend and OCR backend safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable
import urllib.error
import urllib.request


REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_URL = "http://127.0.0.1:8000"
BACKEND_URL = "http://127.0.0.1:8001"
HEALTH_URL = f"{BACKEND_URL}/health"
STATE_VERSION = 1


class LauncherError(RuntimeError):
  """A user-actionable launcher failure."""


def service_specs(repo_root: Path) -> dict[str, dict[str, Any]]:
  backend_executable = repo_root / "backend" / ".venv" / "bin" / "uvicorn"
  return {
    "backend": {
      "command": [
        str(backend_executable),
        "backend.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8001"
      ],
      "markers": [
        str(repo_root),
        "uvicorn",
        "backend.app:app",
        "--port 8001"
      ],
      "log_name": "backend.log"
    },
    "frontend": {
      "command": [
        "/usr/bin/python3",
        "-m",
        "http.server",
        "8000",
        "--bind",
        "127.0.0.1",
        "--directory",
        str(repo_root)
      ],
      "markers": [
        str(repo_root),
        "http.server",
        "8000",
        "--directory"
      ],
      "log_name": "frontend.log"
    }
  }


def runtime_dir(repo_root: Path = REPO_ROOT) -> Path:
  override = os.environ.get("JARVIS_RUNTIME_DIR", "").strip()
  if override:
    return Path(override).expanduser().resolve()
  digest = hashlib.sha256(str(repo_root).encode("utf-8")).hexdigest()[:12]
  return Path(tempfile.gettempdir()) / f"jarvis-local-{os.getuid()}-{digest}"


def state_path(runtime_root: Path) -> Path:
  return runtime_root / "state.json"


def blank_state(repo_root: Path) -> dict[str, Any]:
  return {
    "version": STATE_VERSION,
    "repo_root": str(repo_root),
    "services": {}
  }


def load_state(runtime_root: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
  path = state_path(runtime_root)
  if not path.exists():
    return blank_state(repo_root)
  try:
    state = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return blank_state(repo_root)
  if (
    state.get("version") != STATE_VERSION
    or state.get("repo_root") != str(repo_root)
    or not isinstance(state.get("services"), dict)
  ):
    return blank_state(repo_root)
  return state


def save_state(runtime_root: Path, state: dict[str, Any]) -> None:
  runtime_root.mkdir(parents=True, exist_ok=True)
  try:
    runtime_root.chmod(0o700)
  except OSError:
    pass
  destination = state_path(runtime_root)
  temporary = destination.with_suffix(".tmp")
  temporary.write_text(
    json.dumps(state, indent=2, sort_keys=True) + "\n",
    encoding="utf-8"
  )
  temporary.replace(destination)


def fetch_url(url: str, timeout: float = 1.0) -> tuple[int, bytes] | None:
  request = urllib.request.Request(url, headers={"User-Agent": "Jarvis local launcher"})
  try:
    with urllib.request.urlopen(request, timeout=timeout) as response:
      return int(response.status), response.read()
  except urllib.error.HTTPError as error:
    return int(error.code), error.read()
  except (urllib.error.URLError, TimeoutError, OSError):
    return None


def backend_health_issues(health: Any, repo_root: Path = REPO_ROOT) -> list[str]:
  if not isinstance(health, dict):
    return ["health response is not an object"]

  issues = []
  for flag in (
    "ok",
    "ready",
    "roi_ready",
    "digit_ready",
    "strip_digit_ready",
    "strip_digit_23xx_ready"
  ):
    if health.get(flag) is not True:
      issues.append(f"{flag} is not true")

  expected_paths = {
    "model_path": repo_root / "backend" / "models" / "roi-rotaug-e30-640.pt",
    "digit_model_path": repo_root / "backend" / "models" / "digit_classifier.pt",
    "strip_digit_model_path": repo_root / "backend" / "models" / "digit_strip_reader.pt",
    "strip_digit_23xx_model_path": (
      repo_root / "backend" / "models" / "digit_strip_reader_23xx.pt"
    )
  }
  for field, expected_path in expected_paths.items():
    actual_value = health.get(field)
    try:
      actual_path = Path(actual_value).expanduser().resolve() if actual_value else None
    except (OSError, TypeError):
      actual_path = None
    if actual_path != expected_path.resolve():
      issues.append(
        f"{field} is {actual_path or 'unknown'}, expected {expected_path.resolve()}"
      )
  return issues


def probe_backend(repo_root: Path = REPO_ROOT) -> tuple[bool, bool, str]:
  response = fetch_url(HEALTH_URL)
  if response is None:
    return False, False, "unreachable"
  status, body = response
  if status < 200 or status >= 300:
    return True, False, f"health check returned HTTP {status}"
  try:
    health = json.loads(body.decode("utf-8"))
  except (UnicodeDecodeError, json.JSONDecodeError):
    return True, False, "health check returned invalid JSON"
  issues = backend_health_issues(health, repo_root)
  if issues:
    return True, False, "; ".join(issues)
  return True, True, "ready"


def probe_frontend() -> tuple[bool, bool, str]:
  response = fetch_url(FRONTEND_URL)
  if response is None:
    return False, False, "unreachable"
  status, body = response
  if status < 200 or status >= 300:
    return True, False, f"frontend returned HTTP {status}"
  if b"<title>Jarvis - Personal Assistant</title>" not in body:
    return True, False, "port 8000 is serving a different application"
  return True, True, "ready"


def process_command(pid: int) -> str:
  try:
    result = subprocess.run(
      ["/bin/ps", "-p", str(pid), "-o", "command="],
      check=False,
      capture_output=True,
      text=True,
      timeout=2
    )
  except (OSError, subprocess.SubprocessError):
    return ""
  return result.stdout.strip() if result.returncode == 0 else ""


def command_matches(command: str, markers: list[str]) -> bool:
  return bool(command) and all(marker in command for marker in markers)


def managed_process_matches(
  pid: Any,
  markers: list[str],
  command_reader: Callable[[int], str] = process_command
) -> bool:
  if not isinstance(pid, int) or pid <= 1:
    return False
  return command_matches(command_reader(pid), markers)


def ensure_prerequisites(repo_root: Path = REPO_ROOT) -> None:
  expected = [
    repo_root / "backend" / ".venv" / "bin" / "uvicorn",
    repo_root / "backend" / "models" / "roi-rotaug-e30-640.pt",
    repo_root / "backend" / "models" / "digit_classifier.pt",
    repo_root / "backend" / "models" / "digit_strip_reader.pt",
    repo_root / "backend" / "models" / "digit_strip_reader_23xx.pt"
  ]
  missing = [str(path.relative_to(repo_root)) for path in expected if not path.exists()]
  if missing:
    raise LauncherError(
      "Jarvis is missing required local components: " + ", ".join(missing)
    )
  if not os.access(expected[0], os.X_OK):
    raise LauncherError("backend/.venv/bin/uvicorn is not executable")


def prune_stale_state(
  runtime_root: Path,
  state: dict[str, Any],
  specs: dict[str, dict[str, Any]]
) -> None:
  changed = False
  for service_name, entry in list(state["services"].items()):
    spec = specs.get(service_name)
    if (
      not spec
      or not isinstance(entry, dict)
      or not managed_process_matches(entry.get("pid"), spec["markers"])
    ):
      state["services"].pop(service_name, None)
      changed = True
  if changed:
    save_state(runtime_root, state)


def spawn_service(
  service_name: str,
  runtime_root: Path,
  state: dict[str, Any],
  spec: dict[str, Any],
  repo_root: Path = REPO_ROOT
) -> int:
  runtime_root.mkdir(parents=True, exist_ok=True)
  log_path = runtime_root / spec["log_name"]
  with log_path.open("ab", buffering=0) as output:
    child = subprocess.Popen(
      spec["command"],
      cwd=repo_root,
      stdin=subprocess.DEVNULL,
      stdout=output,
      stderr=subprocess.STDOUT,
      start_new_session=True,
      close_fds=True
    )
  state["services"][service_name] = {
    "pid": child.pid,
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "log_path": str(log_path)
  }
  save_state(runtime_root, state)
  return child.pid


def stop_managed_service(
  service_name: str,
  runtime_root: Path,
  state: dict[str, Any],
  spec: dict[str, Any],
  quiet: bool = False
) -> bool:
  entry = state["services"].get(service_name)
  if not isinstance(entry, dict):
    return False

  pid = entry.get("pid")
  if not managed_process_matches(pid, spec["markers"]):
    state["services"].pop(service_name, None)
    save_state(runtime_root, state)
    if not quiet:
      print(f"Left an unrecognized {service_name} process untouched.")
    return False

  try:
    process_group = os.getpgid(pid)
    if process_group == pid:
      os.killpg(process_group, signal.SIGTERM)
    else:
      os.kill(pid, signal.SIGTERM)
  except ProcessLookupError:
    pass
  except PermissionError as error:
    raise LauncherError(f"Could not stop the {service_name} service: {error}") from error

  for _ in range(50):
    if not managed_process_matches(pid, spec["markers"]):
      break
    time.sleep(0.1)
  else:
    try:
      process_group = os.getpgid(pid)
      if process_group == pid:
        os.killpg(process_group, signal.SIGKILL)
      else:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
      pass

  state["services"].pop(service_name, None)
  save_state(runtime_root, state)
  if not quiet:
    print(f"Stopped Jarvis {service_name}.")
  return True


def log_tail(runtime_root: Path, specs: dict[str, dict[str, Any]], lines: int = 12) -> str:
  chunks = []
  for service_name, spec in specs.items():
    path = runtime_root / spec["log_name"]
    if not path.exists():
      continue
    try:
      content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
      continue
    if content:
      chunks.append(f"{service_name} log:\n" + "\n".join(content[-lines:]))
  return "\n\n".join(chunks)


def wait_until_ready(
  repo_root: Path,
  runtime_root: Path,
  state: dict[str, Any],
  specs: dict[str, dict[str, Any]],
  timeout_seconds: float = 60
) -> None:
  deadline = time.monotonic() + timeout_seconds
  last_detail = ""
  while time.monotonic() < deadline:
    frontend_reachable, frontend_ready, frontend_detail = probe_frontend()
    backend_reachable, backend_ready, backend_detail = probe_backend(repo_root)
    if frontend_ready and backend_ready:
      return

    for service_name, entry in state["services"].items():
      spec = specs[service_name]
      if not managed_process_matches(entry.get("pid"), spec["markers"]):
        tail = log_tail(runtime_root, specs)
        raise LauncherError(
          f"Jarvis {service_name} exited before startup completed."
          + (f"\n\n{tail}" if tail else "")
        )

    last_detail = (
      f"frontend={frontend_detail if frontend_reachable else 'unreachable'}, "
      f"backend={backend_detail if backend_reachable else 'unreachable'}"
    )
    time.sleep(0.25)

  tail = log_tail(runtime_root, specs)
  raise LauncherError(
    f"Jarvis did not become ready within {int(timeout_seconds)} seconds ({last_detail})."
    + (f"\n\n{tail}" if tail else "")
  )


def start_jarvis(no_open: bool = False, repo_root: Path = REPO_ROOT) -> None:
  ensure_prerequisites(repo_root)
  runtime_root = runtime_dir(repo_root)
  specs = service_specs(repo_root)
  state = load_state(runtime_root, repo_root)
  prune_stale_state(runtime_root, state, specs)
  started_now = []

  try:
    backend_reachable, backend_ready, backend_detail = probe_backend(repo_root)
    if backend_reachable and not backend_ready:
      if "backend" in state["services"]:
        stop_managed_service("backend", runtime_root, state, specs["backend"], quiet=True)
      else:
        raise LauncherError(
          f"Port 8001 is already in use by an incompatible service: {backend_detail}"
        )
    if not backend_ready and "backend" not in state["services"]:
      spawn_service("backend", runtime_root, state, specs["backend"], repo_root)
      started_now.append("backend")

    frontend_reachable, frontend_ready, frontend_detail = probe_frontend()
    if frontend_reachable and not frontend_ready:
      if "frontend" in state["services"]:
        stop_managed_service("frontend", runtime_root, state, specs["frontend"], quiet=True)
      else:
        raise LauncherError(
          f"Port 8000 is already in use by an incompatible service: {frontend_detail}"
        )
    if not frontend_ready and "frontend" not in state["services"]:
      spawn_service("frontend", runtime_root, state, specs["frontend"], repo_root)
      started_now.append("frontend")

    wait_until_ready(repo_root, runtime_root, state, specs)
  except Exception:
    for service_name in reversed(started_now):
      stop_managed_service(
        service_name,
        runtime_root,
        state,
        specs[service_name],
        quiet=True
      )
    raise

  if not no_open and os.environ.get("JARVIS_NO_OPEN") != "1":
    result = subprocess.run(
      ["/usr/bin/open", FRONTEND_URL],
      check=False,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL
    )
    if result.returncode != 0:
      print(f"Jarvis is ready. Open {FRONTEND_URL} in your browser.")
      return

  print(f"Jarvis is ready at {FRONTEND_URL}")


def stop_jarvis(repo_root: Path = REPO_ROOT) -> None:
  runtime_root = runtime_dir(repo_root)
  specs = service_specs(repo_root)
  state = load_state(runtime_root, repo_root)
  prune_stale_state(runtime_root, state, specs)

  stopped = False
  for service_name in ("frontend", "backend"):
    stopped = (
      stop_managed_service(service_name, runtime_root, state, specs[service_name])
      or stopped
    )

  frontend_reachable, _, _ = probe_frontend()
  backend_reachable, _, _ = probe_backend(repo_root)
  if frontend_reachable or backend_reachable:
    if not stopped:
      print(
        "Jarvis-compatible services are running, but this launcher did not start them, "
        "so they were left untouched."
      )
    else:
      print("Launcher-managed services stopped; another local service is still using a Jarvis port.")
    return

  path = state_path(runtime_root)
  try:
    path.unlink()
  except FileNotFoundError:
    pass
  print("Jarvis is stopped.")


def status_jarvis(repo_root: Path = REPO_ROOT) -> int:
  frontend_reachable, frontend_ready, frontend_detail = probe_frontend()
  backend_reachable, backend_ready, backend_detail = probe_backend(repo_root)
  if frontend_ready and backend_ready:
    print(f"Jarvis is ready at {FRONTEND_URL}")
    return 0
  if frontend_reachable or backend_reachable:
    print(
      "Jarvis is only partially ready: "
      f"frontend={frontend_detail}, backend={backend_detail}"
    )
    return 1
  print("Jarvis is stopped.")
  return 1


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Manage the local Jarvis app.")
  subparsers = parser.add_subparsers(dest="action", required=True)
  start_parser = subparsers.add_parser("start", help="Start Jarvis and open it.")
  start_parser.add_argument(
    "--no-open",
    action="store_true",
    help="Start and verify Jarvis without opening a browser."
  )
  subparsers.add_parser("stop", help="Stop services started by this launcher.")
  subparsers.add_parser("status", help="Show whether Jarvis is ready.")
  return parser


def main(argv: list[str] | None = None) -> int:
  args = build_parser().parse_args(argv)
  try:
    if args.action == "start":
      start_jarvis(no_open=args.no_open)
      return 0
    if args.action == "stop":
      stop_jarvis()
      return 0
    return status_jarvis()
  except LauncherError as error:
    print(f"Jarvis launcher error: {error}", file=sys.stderr)
    print(f"Logs: {runtime_dir()}", file=sys.stderr)
    return 1
  except KeyboardInterrupt:
    print("Jarvis launcher interrupted.", file=sys.stderr)
    return 130


if __name__ == "__main__":
  raise SystemExit(main())
