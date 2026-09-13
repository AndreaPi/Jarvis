import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parent / "jarvis_local.py"
SPEC = importlib.util.spec_from_file_location("jarvis_local", SCRIPT_PATH)
jarvis_local = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(jarvis_local)


class JarvisLocalLauncherTests(unittest.TestCase):
  def setUp(self):
    self.repo_root = Path("/tmp/example-jarvis").resolve()

  def valid_health(self):
    return {
      "ok": True,
      "ready": True,
      "roi_ready": True,
      "digit_ready": True,
      "strip_digit_ready": True,
      "strip_digit_23xx_ready": True,
      "model_path": str(
        self.repo_root / "backend" / "models" / "roi-rotaug-e30-640.pt"
      ),
      "digit_model_path": str(
        self.repo_root / "backend" / "models" / "digit_classifier.pt"
      ),
      "strip_digit_model_path": str(
        self.repo_root / "backend" / "models" / "digit_strip_reader.pt"
      ),
      "strip_digit_23xx_model_path": str(
        self.repo_root / "backend" / "models" / "digit_strip_reader_23xx.pt"
      )
    }

  def test_health_requires_ready_models_and_canonical_checkpoints(self):
    cases = (
      ({}, None),
      ({"strip_digit_23xx_ready": False}, "strip_digit_23xx_ready is not true"),
      ({"model_path": str(self.repo_root / "backend" / "runs" / "challenger.pt")},
       "challenger.pt"),
    )
    for overrides, expected_issue in cases:
      with self.subTest(overrides=overrides):
        health = {**self.valid_health(), **overrides}
        issues = jarvis_local.backend_health_issues(health, self.repo_root)
        if expected_issue is None:
          self.assertEqual(issues, [])
        else:
          self.assertTrue(any(expected_issue in issue for issue in issues), issues)

  def test_process_identity_requires_every_marker(self):
    command = (
      "/tmp/example-jarvis/backend/.venv/bin/python "
      "/tmp/example-jarvis/backend/.venv/bin/uvicorn "
      "backend.app:app --host 127.0.0.1 --port 8001"
    )
    self.assertTrue(
      jarvis_local.command_matches(
        command,
        ["/tmp/example-jarvis", "uvicorn", "backend.app:app", "--port 8001"]
      )
    )
    self.assertFalse(
      jarvis_local.command_matches(
        command,
        ["/tmp/example-jarvis", "http.server", "8000"]
      )
    )

  def test_unknown_pid_identity_is_never_managed(self):
    self.assertFalse(
      jarvis_local.managed_process_matches(
        4242,
        ["uvicorn", "backend.app:app"],
        command_reader=lambda _pid: "python -m http.server 8000"
      )
    )

  def test_state_round_trip_is_scoped_to_repository(self):
    with tempfile.TemporaryDirectory() as directory:
      runtime_root = Path(directory)
      state = jarvis_local.blank_state(self.repo_root)
      state["services"]["backend"] = {"pid": 4242}
      jarvis_local.save_state(runtime_root, state)
      loaded = jarvis_local.load_state(runtime_root, self.repo_root)
      self.assertEqual(loaded, state)

      raw = json.loads((runtime_root / "state.json").read_text(encoding="utf-8"))
      self.assertEqual(raw["repo_root"], str(self.repo_root))
      different_repo = Path("/tmp/other-jarvis").resolve()
      self.assertEqual(
        jarvis_local.load_state(runtime_root, different_repo),
        jarvis_local.blank_state(different_repo)
      )


if __name__ == "__main__":
  unittest.main()
