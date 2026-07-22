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

test('accepts the canonical ready backend', () => {
  assert.doesNotThrow(() => validateQaBackendHealth(validHealth(), ROOT_DIR));
});

test('rejects a backend whose required models are not ready', () => {
  const health = validHealth();
  health.digit_ready = false;
  assert.throws(
    () => validateQaBackendHealth(health, ROOT_DIR),
    /digit_ready is not true/
  );
});

test('rejects a backend serving a non-canonical checkpoint', () => {
  const health = validHealth();
  health.model_path = path.join(ROOT_DIR, 'backend', 'runs', 'challenger.pt');
  assert.throws(
    () => validateQaBackendHealth(health, ROOT_DIR),
    /ROI checkpoint is .*challenger\.pt/
  );
});
