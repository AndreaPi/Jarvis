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

  def test_accepts_complete_canonical_health(self):
    self.assertEqual(
      jarvis_local.backend_health_issues(self.valid_health(), self.repo_root),
      []
    )

  def test_rejects_missing_shadow_model_readiness(self):
    health = self.valid_health()
    health["strip_digit_23xx_ready"] = False
    self.assertIn(
      "strip_digit_23xx_ready is not true",
      jarvis_local.backend_health_issues(health, self.repo_root)
    )

  def test_rejects_noncanonical_checkpoint(self):
    health = self.valid_health()
    health["model_path"] = str(self.repo_root / "backend" / "runs" / "challenger.pt")
    issues = jarvis_local.backend_health_issues(health, self.repo_root)
    self.assertTrue(any("challenger.pt" in issue for issue in issues))

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
