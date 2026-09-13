import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class OrientationReviewRetentionTests(unittest.TestCase):
  def test_clean_extraction_preserves_review_log_and_overrides(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      output = root / "digits"
      manifests = output / "manifests"
      manifests.mkdir(parents=True)
      retained = {
        "orientation_reviews.csv": "filename,decision\nsample.jpeg,rejected\n",
        "direction_overrides.csv": "filename,rotation\nsample.jpeg,90\n",
      }
      for name, contents in retained.items():
        (manifests / name).write_text(contents)
      (output / "stale.txt").write_text("old derivative")
      readings = root / "readings.csv"
      readings.write_text("filename,value\n")
      roi = root / "roi"
      roi.mkdir()
      subprocess.run([
        sys.executable, str(Path(__file__).with_name("extract_digit_windows.py")),
        "--csv", str(readings), "--roi-dataset-dir", str(roi),
        "--out-dir", str(output), "--clean",
      ], check=True, capture_output=True, text=True)
      for name, contents in retained.items():
        with self.subTest(manifest=name):
          self.assertEqual((manifests / name).read_text(), contents)
      self.assertFalse((output / "stale.txt").exists())


if __name__ == "__main__":
  unittest.main()
