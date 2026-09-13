from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().with_name("monitor-thermal.sh")


def sample(number, pressure):
  return (
    f"*** Sampled system activity (sample {number}) ***\n"
    "**** Thermal pressure ****\n"
    f"Current pressure level: {pressure}\n"
    f"Sample {number} detail\n\n"
  )


class ThermalMonitorTests(unittest.TestCase):
  def test_display_modes_preserve_event_only_log(self):
    for pressures in (
      ("Nominal", "Nominal", "Heavy", "Nominal", "Moderate"),
      ("Heavy", "Nominal", "Moderate"),
      ("Nominal", "Nominal"),
    ):
      blocks = [sample(i, pressure) for i, pressure in enumerate(pressures)]
      events = "".join(b for b, p in zip(blocks, pressures) if p != "Nominal")
      for verbose in (0, 1):
        with self.subTest(pressures=pressures, verbose=verbose):
          with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "events.log"
            log.write_text("existing event\n")
            result = subprocess.run(
              ["/bin/bash", "-c", 'source "$1"; filter_thermal_events "$2" "$3"',
               "test", str(SCRIPT), str(log), str(verbose)],
              input="".join(blocks), capture_output=True, text=True, timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = "".join(
              b for i, (b, p) in enumerate(zip(blocks, pressures))
              if verbose or i == 0 or p != "Nominal"
            )
            self.assertEqual(result.stdout, expected)
            self.assertEqual(log.read_text(), "existing event\n" + events)

  @unittest.skipUnless(sys.platform == "darwin", "macOS command wrapper")
  def test_verbose_flags_and_log_path_with_mocked_sampler(self):
    fixture = sample(0, "Nominal") + sample(1, "Nominal") + sample(2, "Heavy")
    for options in ([], ["-v"], ["--verbose"]):
      for flags_first in (True, False):
        with self.subTest(options=options, flags_first=flags_first):
          with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "thermal events.log"
            args = options + [str(log)] if flags_first else [str(log)] + options
            result = subprocess.run(
              ["/bin/bash", "-c", '''
source "$1"
shift
sudo() {
  if [[ "$1" == "-v" ]]; then return 0; fi
  cat
}
main "$@"
''', "test", str(SCRIPT), *args],
              input=fixture, capture_output=True, text=True, timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(sample(0, "Nominal"), result.stdout)
            self.assertEqual(sample(1, "Nominal") in result.stdout, bool(options))
            self.assertIn(sample(2, "Heavy"), result.stdout)
            self.assertEqual(log.read_text(), sample(2, "Heavy"))


if __name__ == "__main__":
  unittest.main()
