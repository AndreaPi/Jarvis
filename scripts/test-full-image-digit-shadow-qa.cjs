const assert = require('node:assert/strict');
const test = require('node:test');
const { trackedProcess, waitFor, validateCheckpointProvenance, validateShadowBackendHealth } = require('./export-full-image-digit-shadow-qa.cjs');
const { selectValidationRows, buildRows, summarize } = require('./export-full-image-digit-shadow-qa.cjs');
const { validateFrontendSource } = require('./export-full-image-digit-shadow-qa.cjs');

test('UI benchmark rejects another checkout before using its frontend or readings', async () => {
  const fs = require('node:fs');
  const path = require('node:path');
  const crypto = require('node:crypto');
  const root = path.resolve(__dirname, '..');
  const fetchFromCheckout = async (url) => fs.readFileSync(path.join(root, new URL(url).pathname));
  const hashes = await validateFrontendSource('http://127.0.0.1:8000', fetchFromCheckout);
  assert.equal(hashes['src/ocr/pipeline.js'],
    crypto.createHash('sha256').update(fs.readFileSync(path.join(root, 'src/ocr/pipeline.js'))).digest('hex'));
  assert.ok(hashes['assets/meter_readings.csv']);
  for (const changed of ['src/ocr/pipeline.js', 'src/testset/run-test-set.js', 'assets/meter_readings.csv']) {
    assert.ok(hashes[changed], changed);
    await assert.rejects(validateFrontendSource('http://127.0.0.1:8000', async (url) => (
      new URL(url).pathname === `/${changed}` ? Buffer.from('another checkout') : fetchFromCheckout(url)
    )), /Frontend serves different checkout content/);
  }
});

test('UI validation excludes retired fold sources while preserving full diagnostic and provenance', () => {
  const fs = require('node:fs');
  const os = require('node:os');
  const path = require('node:path');
  const crypto = require('node:crypto');
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'jarvis-ui-exclusions-'));
  try {
    const folds = path.join(root, 'folds.csv');
    const original = 'filename,fold\nactive.jpg,0\nretired.jpg,0\ntraining.jpg,1\n';
    fs.writeFileSync(folds, original);
    fs.writeFileSync(path.join(root, 'dataset_provenance.json'), JSON.stringify({
      selected_fold: 0, cv_folds_sha256: crypto.createHash('sha256').update(original).digest('hex')
    }));
    const exclusions = path.join(root, 'exclusions.csv');
    fs.writeFileSync(exclusions, 'filename,scope,reason,retention\nretired.jpg,full_image_digit_detection,"defocus, obsolete domain",legacy_stress\n');
    const verified = validateCheckpointProvenance(path.join(root, 'best.pt'), folds, '0', exclusions);
    const rows = buildRows(['active.jpg', 'retired.jpg', 'training.jpg', 'new.jpg'].map((filename) => ({
      filename, expected: '1234', detected: filename === 'retired.jpg' ? '' : '1234'
    })), new Map(Object.entries(verified.fold_assignments)));
    const selected = selectValidationRows(rows, verified);
    assert.deepEqual(selected.map((row) => row.filename), ['active.jpg']);
    assert.equal(summarize(selected, 'production_value').no_read_count, 0);
    assert.equal(summarize(rows, 'production_value').no_read_count, 1);
    assert.equal(verified.fold_assignments['retired.jpg'], 0);
    assert.equal(fs.readFileSync(folds, 'utf8'), original);
    assert.equal(verified.evaluation_exclusions.sha256,
      crypto.createHash('sha256').update(fs.readFileSync(exclusions)).digest('hex'));
    assert.throws(() => selectValidationRows(rows.filter((row) => row.filename !== 'active.jpg'), verified), /No active UI rows/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('failed frontend/backend readiness stops the process before losing its handle', async () => {
  for (const label of ['frontend', 'shadow backend']) {
    const tracked = trackedProcess(process.execPath, ['-e', 'setInterval(() => {}, 1000)']);
    try {
      await assert.rejects(waitFor(async () => { throw new Error('health incomplete'); }, tracked, label, 10), /was not ready.*health incomplete/);
      assert.ok(tracked.child.exitCode !== null || tracked.child.signalCode !== null);
      assert.throws(() => process.kill(tracked.child.pid, 0), { code: 'ESRCH' });
    } finally {
      await tracked.stop();
    }
  }
});

test('early exit and spawn failure are reported and cleaned up', async () => {
  const exited = {
    child: { exitCode: 1, signalCode: null },
    output: () => ({ stdout: '', stderr: 'invalid checkpoint' }),
    stop: async () => { exited.stopped = true; }
  };
  await assert.rejects(waitFor(async () => true, exited, 'backend'), /exited early.*\nstdout:\n\nstderr:\ninvalid checkpoint/);
  assert.equal(exited.stopped, true);
  const missing = trackedProcess('/nonexistent-jarvis-qa-python', []);
  await assert.rejects(waitFor(async () => { throw new Error('not ready'); }, missing, 'backend', 2000), /could not start/);
});

test('successful readiness leaves process ownership with the caller', async () => {
  const tracked = {
    child: { exitCode: null, signalCode: null },
    stop: async () => { throw new Error('must not stop a ready service'); }
  };
  assert.deepEqual(await waitFor(async () => ({ ready: true }), tracked, 'backend'), { ready: true });
});


test('UI verifies original checkpoint membership before starting services', () => {
  const fs = require('node:fs');
  const os = require('node:os');
  const path = require('node:path');
  const crypto = require('node:crypto');
  const { spawnSync } = require('node:child_process');
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'jarvis-provenance-'));
  try {
    // Directory name deliberately disagrees with original validation fold.
    const run = path.join(root, 'misleading-fold0');
    fs.mkdirSync(path.join(run, 'weights'), { recursive: true });
    const checkpoint = path.join(run, 'weights', 'best.pt');
    fs.writeFileSync(checkpoint, 'unused model');
    const folds = path.join(root, 'folds.csv');
    const original = 'filename,fold\nvalidation.jpg,4\ntraining.jpg,1\n';
    fs.writeFileSync(folds, original);
    const provenancePath = path.join(run, 'dataset_provenance.json');
    const provenance = {
      selected_fold: 4,
      cv_folds_sha256: crypto.createHash('sha256').update(original).digest('hex')
    };
    fs.writeFileSync(provenancePath, JSON.stringify(provenance));
    for (const requestedFold of [undefined, '4']) {
      const verified = validateCheckpointProvenance(checkpoint, folds, requestedFold);
      assert.equal(verified.selected_fold, 4);
      assert.deepEqual(verified.fold_assignments, { 'validation.jpg': 4, 'training.jpg': 1 });
      assert.equal(verified.cv_folds_sha256, provenance.cv_folds_sha256);
    }
    const cases = [
      { fold: '1', metadata: provenance, csv: original, error: /validation fold 4/ },
      { fold: '4junk', metadata: provenance, csv: original, error: /invalid int value/ },
      { metadata: null, csv: original, error: /Missing checkpoint training provenance/ },
      { metadata: { selected_fold: 4 }, csv: original, error: /original cv_folds_sha256/ },
      { metadata: provenance, csv: 'filename,fold\nvalidation.jpg,1\ntraining.jpg,4\n',
        error: /does not match/ }
    ];
    for (const scenario of cases) {
      fs.writeFileSync(folds, scenario.csv);
      fs.rmSync(provenancePath, { force: true });
      if (scenario.metadata) fs.writeFileSync(provenancePath, JSON.stringify(scenario.metadata));
      const env = {
        ...process.env,
        FULL_IMAGE_DIGIT_SHADOW_MODEL_PATH: checkpoint,
        FULL_IMAGE_DIGIT_SHADOW_CV_FOLDS_PATH: folds,
        // Any attempt to reach a service would fail with a different error.
        JARVIS_FRONTEND_URL: 'invalid://must-not-start-services',
        JARVIS_SHADOW_BACKEND_URL: 'invalid://must-not-start-services'
      };
      delete env.FULL_IMAGE_DIGIT_SHADOW_VALIDATION_FOLD;
      if (scenario.fold !== undefined) env.FULL_IMAGE_DIGIT_SHADOW_VALIDATION_FOLD = scenario.fold;
      const result = spawnSync(process.execPath,
        [path.join(__dirname, 'export-full-image-digit-shadow-qa.cjs')],
        { env, encoding: 'utf8', timeout: 10000 });
      assert.equal(result.status, 1, result.stderr);
      assert.match(result.stderr, scenario.error);
      assert.doesNotMatch(result.stderr, /Protocol .* not supported/);
    }
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});


test('shadow benchmark requires both canonical primary models and its selected shadow', () => {
  const path = require('node:path');
  const root = path.resolve(__dirname, '..');
  const checkpoint = path.join(root, 'backend/runs/shadow/weights/best.pt');
  const healthy = {
    roi_ready: true, digit_ready: true,
    model_path: path.join(root, 'backend/models/roi-rotaug-e30-640.pt'),
    digit_model_path: path.join(root, 'backend/models/digit_classifier.pt'),
    full_image_digit_shadow_ready: true,
    full_image_digit_shadow_model_path: checkpoint
  };
  assert.equal(validateShadowBackendHealth(healthy, checkpoint), healthy);
  for (const [field, value, error] of [
    ['model_path', path.join(root, 'challenger.pt'), /ROI checkpoint/],
    ['digit_model_path', path.join(root, 'challenger.pt'), /digit checkpoint/],
    ['roi_ready', false, /roi_ready/],
    ['digit_ready', false, /digit_ready/],
    ['full_image_digit_shadow_ready', false, /shadow_ready/],
    ['full_image_digit_shadow_model_path', path.join(root, 'other.pt'), /unexpected shadow/],
    ['full_image_digit_shadow_model_path', undefined, /unexpected shadow/]
  ]) {
    assert.throws(() => validateShadowBackendHealth({ ...healthy, [field]: value }, checkpoint), error);
  }
});
