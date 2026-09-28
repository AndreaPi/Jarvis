from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

try:
  from .thermal_pause import ThermalGate, install_yolo_thermal_callbacks
except ImportError:
  from thermal_pause import ThermalGate, install_yolo_thermal_callbacks


def publish(path, action, age=0):
  temporary = path.with_suffix('.new')
  temporary.write_text(json.dumps({"action": action, "reason": "test", "updated_monotonic": time.clock_gettime(time.CLOCK_MONOTONIC)-age}))
  temporary.replace(path)


class ThermalPauseTests(unittest.TestCase):
  def test_yolo_callbacks_preserve_training_state_and_continuation(self):
    def run(directory, pause):
      torch.manual_seed(812)
      model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Dropout(.2))
      optimizer = torch.optim.Adam(model.parameters(), lr=.01)
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
      callbacks = {}
      adapter = SimpleNamespace(add_callback=lambda event, cb: callbacks.setdefault(event, []).append(cb))
      control = directory / 'thermal-control.json'
      publish(control, 'run')
      trainer = SimpleNamespace(device=torch.device('cpu'), train_time_start=time.time(),
                                epoch_time_start=time.time(), stopper={'best_epoch': 2, 'best_fitness': .5})
      def invoke(name):
        for cb in callbacks.get(name, []): cb(trainer)
      with patch.dict(os.environ, {'JARVIS_THERMAL_CONTROL': str(control)}):
        install_yolo_thermal_callbacks(adapter)
        invoke('on_train_start')
        for batch in range(4):
          if pause and batch == 3:
            checkpoint = directory / 'last.pt'
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'stopper': trainer.stopper}, checkpoint)
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            rng = torch.get_rng_state().clone()
            weights = copy.deepcopy(model.state_dict())
            gradients = [p.grad.clone() for p in model.parameters()]
            opt = copy.deepcopy(optimizer.state_dict())
            sched, stopper = copy.deepcopy(scheduler.state_dict()), copy.deepcopy(trainer.stopper)
            publish(control, 'pause')
            def resume():
              publish(control, 'run')
            timer = threading.Timer(.25, resume)
            timer.start()
            try:
              invoke('on_train_batch_start')
            finally:
              timer.join()
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for key in weights: self.assertTrue(torch.equal(weights[key], model.state_dict()[key]))
            for expected, param in zip(gradients, model.parameters()): self.assertTrue(torch.equal(expected, param.grad))
            self.assertEqual(sched, scheduler.state_dict())
            self.assertEqual(stopper, trainer.stopper)
            self.assertEqual(before, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(opt['param_groups'], optimizer.state_dict()['param_groups'])
            for key, values in opt['state'].items():
              for name, value in values.items(): self.assertTrue(torch.equal(value, optimizer.state_dict()['state'][key][name]))
          else:
            invoke('on_train_batch_start')
          model(torch.ones(1, 2)).square().mean().backward()
          if batch % 2:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        self.assertIn('on_val_batch_start', callbacks)
      return model.state_dict(), optimizer.state_dict(), scheduler.state_dict()
    with tempfile.TemporaryDirectory() as tmp:
      baseline, paused = Path(tmp)/'baseline', Path(tmp)/'paused'
      baseline.mkdir(); paused.mkdir()
      first, second = run(baseline, False), run(paused, True)
      for key in first[0]: self.assertTrue(torch.equal(first[0][key], second[0][key]))
      self.assertEqual(first[2], second[2])
      for key, values in first[1]['state'].items():
        for name, value in values.items(): self.assertTrue(torch.equal(value, second[1]['state'][key][name]))

  def test_stale_or_missing_telemetry_blocks_and_pause_is_interruptible(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp)/'control.json'
      gate = ThermalGate(path)
      self.assertFalse(gate.decision()[0])
      for contents in ('[]', '{bad', '{"action":"run","updated_monotonic":"nan"}'):
        path.write_text(contents)
        self.assertFalse(gate.decision()[0])
      publish(path, 'run', age=20)
      self.assertFalse(gate.decision()[0])
      publish(path, 'pause')
      process = subprocess.Popen(
        [sys.executable, '-u', '-c', 'from thermal_pause import thermal_checkpoint; thermal_checkpoint(); print("unexpected")'],
        cwd=Path(__file__).parent, env={**os.environ, 'JARVIS_THERMAL_CONTROL': str(path)},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
      try:
        self.assertIn('paused', process.stdout.readline())
        process.send_signal(signal.SIGINT)
        out, err = process.communicate(timeout=5)
        self.assertNotEqual(process.returncode, 0)
        self.assertNotIn('unexpected', out)
      finally:
        if process.poll() is None: process.kill(); process.wait()
        process.stdout.close(); process.stderr.close()


if __name__ == '__main__':
  unittest.main()
