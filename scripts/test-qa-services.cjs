const path = require('node:path');
const test = require('node:test');
const assert = require('node:assert/strict');

const { validateQaBackendHealth } = require('./lib/qa-services.cjs');

const ROOT_DIR = path.resolve(__dirname, '..');
const validHealth = () => ({
  roi_ready: true,
  digit_ready: true,
  model_path: path.join(ROOT_DIR, 'backend', 'models', 'roi-rotaug-e30-640.pt'),
  digit_model_path: path.join(ROOT_DIR, 'backend', 'models', 'digit_classifier.pt')
});

test('requires ready models and canonical checkpoints', () => {
  assert.doesNotThrow(() => validateQaBackendHealth(validHealth(), ROOT_DIR));
  for (const [overrides, error] of [
    [{ digit_ready: false }, /digit_ready is not true/],
    [{ model_path: path.join(ROOT_DIR, 'backend', 'runs', 'challenger.pt') },
      /ROI checkpoint is .*challenger\.pt/]
  ]) {
    assert.throws(
      () => validateQaBackendHealth({ ...validHealth(), ...overrides }, ROOT_DIR),
      error,
      JSON.stringify(overrides)
    );
  }
});
