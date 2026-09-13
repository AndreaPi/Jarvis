#!/usr/bin/env node

const crypto = require('node:crypto');
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const http = require('node:http');
const path = require('node:path');
const { spawn, execFileSync } = require('node:child_process');
const { validateQaBackendHealth } = require('./lib/qa-services.cjs');

const ROOT_DIR = path.resolve(__dirname, '..');
const FRONTEND_URL = process.env.JARVIS_FRONTEND_URL || 'http://127.0.0.1:8000';
const BACKEND_URL = process.env.JARVIS_SHADOW_BACKEND_URL || 'http://127.0.0.1:8101';
const CHECKPOINT_PATH = path.resolve(
  process.env.FULL_IMAGE_DIGIT_SHADOW_MODEL_PATH
    || path.join(
      ROOT_DIR,
      'backend/runs/full-image-digit-detector-balanced48-crops-fold4/weights/best.pt'
    )
);
const OUTPUT_ROOT = path.join(ROOT_DIR, 'output', 'full-image-digit-shadow-qa');
const CV_FOLDS_PATH = path.resolve(
  process.env.FULL_IMAGE_DIGIT_SHADOW_CV_FOLDS_PATH
    || path.join(ROOT_DIR, 'backend/data/full_image_digit_dataset/manifests/cv_folds.csv')
);

const SOURCE_EXCLUSIONS_PATH = path.resolve(
  process.env.FULL_IMAGE_DIGIT_SHADOW_SOURCE_EXCLUSIONS_PATH
    || path.join(ROOT_DIR, 'backend/data/full_image_digit_dataset/manifests/source_exclusions.csv')
);

const validateCheckpointProvenance = (checkpointPath, foldsPath, requestedFold, exclusionsPath = SOURCE_EXCLUSIONS_PATH) => {
  const args = [
    path.join(ROOT_DIR, 'backend/full_image_checkpoint_provenance.py'),
    '--checkpoint', checkpointPath, '--folds', foldsPath, '--source-exclusions', exclusionsPath
  ];
  if (requestedFold !== undefined) {
    // Pass the exact value: argparse rejects partial numbers such as "4junk".
    args.push('--fold', String(requestedFold));
  }
  const output = execFileSync('python3', args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  return JSON.parse(output);
};

const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

const timestampId = () => {
  const now = new Date();
  return [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, '0'),
    String(now.getDate()).padStart(2, '0'),
    '-',
    String(now.getHours()).padStart(2, '0'),
    String(now.getMinutes()).padStart(2, '0'),
    String(now.getSeconds()).padStart(2, '0')
  ].join('');
};

const requestJson = (url) => new Promise((resolve, reject) => {
  const request = http.get(url, (response) => {
    let body = '';
    response.setEncoding('utf8');
    response.on('data', (chunk) => {
      body += chunk;
    });
    response.on('end', () => {
      if ((response.statusCode || 0) < 200 || (response.statusCode || 0) >= 300) {
        reject(new Error(`HTTP ${response.statusCode || 0}`));
        return;
      }
      try {
        resolve(JSON.parse(body));
      } catch (error) {
        reject(error);
      }
    });
  });
  request.on('error', reject);
  request.setTimeout(3000, () => request.destroy(new Error('timeout')));
});

const requestBytes = (url) => new Promise((resolve, reject) => {
  const request = http.get(url, (response) => {
    const chunks = [];
    const status = response.statusCode || 0;
    response.on('data', (chunk) => chunks.push(chunk));
    response.on('error', reject);
    response.on('end', () => {
      if (status >= 200 && status < 300) {
        resolve(Buffer.concat(chunks));
      } else {
        reject(new Error(`HTTP ${status}`));
      }
    });
  });
  request.on('error', reject);
  request.setTimeout(3000, () => request.destroy(new Error('timeout')));
});

const requestOk = async (url) => { await requestBytes(url); return true; };

const validateFrontendSource = async (url, fetchBytes = requestBytes) => {
  const files = ['index.html', 'app.js', 'styles.css', 'assets/meter_readings.csv'];
  const collectScripts = (directory) => {
    for (const entry of fs.readdirSync(path.join(ROOT_DIR, directory), { withFileTypes: true })) {
      const relative = `${directory}/${entry.name}`;
      if (entry.isDirectory()) collectScripts(relative);
      else if (entry.name.endsWith('.js')) files.push(relative);
    }
  };
  collectScripts('src');
  const hashes = {};
  for (const relative of files.sort()) {
    const expected = fs.readFileSync(path.join(ROOT_DIR, relative));
    const served = await fetchBytes(new URL(relative, `${url.replace(/\/$/, '')}/`).href);
    if (!expected.equals(served)) {
      throw new Error(`Frontend serves different checkout content: ${relative}. Use JARVIS_FRONTEND_URL for this checkout.`);
    }
    hashes[relative] = crypto.createHash('sha256').update(expected).digest('hex');
  }
  return hashes;
};

const trackedProcess = (command, args, options = {}) => {
  const child = spawn(command, args, {
    cwd: options.cwd || ROOT_DIR,
    env: options.env || process.env,
    stdio: ['ignore', 'pipe', 'pipe']
  });
  let stdout = '';
  let stderr = '';
  let startupError = null;
  child.on('error', (error) => { startupError = error; });
  child.stdout.on('data', (chunk) => {
    stdout = `${stdout}${chunk}`.slice(-8000);
  });
  child.stderr.on('data', (chunk) => {
    stderr = `${stderr}${chunk}`.slice(-8000);
  });
  return {
    child,
    output: () => ({ stdout, stderr }),
    startupError: () => startupError,
    stop: async () => {
      if (!child.pid || child.exitCode !== null || child.signalCode !== null) {
        return;
      }
      child.kill('SIGTERM');
      for (let index = 0; index < 30; index += 1) {
        if (child.exitCode !== null || child.signalCode !== null) {
          return;
        }
        await sleep(100);
      }
      if (child.exitCode === null && child.signalCode === null) {
        child.kill('SIGKILL');
      }
    }
  };
};

const waitFor = async (probe, tracked, label, timeoutMs = 120000) => {
  try {
    return await waitForReady(probe, tracked, label, timeoutMs);
  } catch (error) {
    // Startup failed before the caller could receive this process handle.
    if (tracked) {
      await tracked.stop();
    }
    throw error;
  }
};

const waitForReady = async (probe, tracked, label, timeoutMs) => {
  const deadline = Date.now() + timeoutMs;
  let lastError = null;
  while (Date.now() < deadline) {
    if (tracked?.startupError?.()) {
      throw new Error(`${label} could not start: ${tracked.startupError().message}`);
    }
    if (tracked && (tracked.child.exitCode !== null || tracked.child.signalCode !== null)) {
      const output = tracked.output();
      throw new Error(
        `${label} exited early.\nstdout:\n${output.stdout}\nstderr:\n${output.stderr}`
      );
    }
    try {
      return await probe();
    } catch (error) {
      lastError = error;
    }
    await sleep(400);
  }
  const output = tracked ? tracked.output() : { stdout: '', stderr: '' };
  throw new Error(
    `${label} was not ready: ${String(lastError || 'timeout')}\n`
      + `stdout:\n${output.stdout}\nstderr:\n${output.stderr}`
  );
};

const ensureFrontend = async () => {
  try {
    await requestOk(FRONTEND_URL);
    return null;
  } catch {
    // Start a disposable frontend below.
  }
  const parsed = new URL(FRONTEND_URL);
  const frontend = trackedProcess(
    'python3',
    ['-m', 'http.server', parsed.port || '8000', '--bind', parsed.hostname],
  );
  await waitFor(() => requestOk(FRONTEND_URL), frontend, 'frontend');
  return frontend;
};

const validateShadowBackendHealth = (payload, checkpointPath) => {
  validateQaBackendHealth(payload, ROOT_DIR);
  if (payload.full_image_digit_shadow_ready !== true) {
    throw new Error('full_image_digit_shadow_ready is not true');
  }
  if (typeof payload.full_image_digit_shadow_model_path !== 'string'
      || path.resolve(payload.full_image_digit_shadow_model_path) !== checkpointPath) {
    throw new Error(`unexpected shadow model: ${payload.full_image_digit_shadow_model_path}`);
  }
  return payload;
};

const startBackend = async () => {
  const parsed = new URL(BACKEND_URL);
  const backend = trackedProcess(
    path.join(ROOT_DIR, 'backend/.venv/bin/uvicorn'),
    ['backend.app:app', '--host', parsed.hostname, '--port', parsed.port || '8101'],
    {
      env: {
        ...process.env,
        FULL_IMAGE_DIGIT_SHADOW_MODEL_PATH: CHECKPOINT_PATH,
        FULL_IMAGE_DIGIT_SHADOW_DEVICE: process.env.FULL_IMAGE_DIGIT_SHADOW_DEVICE || 'cpu',
        ROI_DEVICE: process.env.ROI_DEVICE || 'cpu',
        DIGIT_DEVICE: process.env.DIGIT_DEVICE || 'cpu',
        MPLCONFIGDIR: process.env.MPLCONFIGDIR || '/tmp/jarvis-shadow-matplotlib'
      }
    }
  );
  const health = await waitFor(async () => {
    const payload = await requestJson(`${BACKEND_URL}/health`);
    return validateShadowBackendHealth(payload, CHECKPOINT_PATH);
  }, backend, 'shadow backend');
  return { backend, health };
};

const parseReading = (value) => {
  const normalized = String(value || '').trim();
  return /^\d{4}$/.test(normalized) ? normalized : '';
};

const summarize = (rows, valueKey) => {
  const readings = rows
    .map((row) => ({ expected: parseReading(row.expected), value: parseReading(row[valueKey]) }))
    .filter((row) => row.expected);
  const readable = readings.filter((row) => row.value);
  const errors = readable.map((row) => Math.abs(Number(row.expected) - Number(row.value)));
  const exact = readable.filter((row) => row.expected === row.value).length;
  return {
    image_count: readings.length,
    readable_count: readable.length,
    no_read_count: readings.length - readable.length,
    exact_match_count: exact,
    exact_match_rate: readings.length ? exact / readings.length : 0,
    readable_mae: errors.length ? errors.reduce((sum, value) => sum + value, 0) / errors.length : null
  };
};

const runUiBenchmark = async () => {
  const { chromium } = require('playwright');
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  try {
    await page.addInitScript((backendUrl) => {
      window.__JARVIS_OCR_CONFIG_OVERRIDE__ = {
        neuralRoi: {
          endpoint: `${backendUrl}/roi/detect`
        },
        digitClassifier: {
          enabled: true,
          endpoint: `${backendUrl}/digit/predict-cells`
        },
        digitStripReader: {
          enabled: false
        },
        digitStripReader23xx: {
          enabled: false
        },
        fullImageDigitShadow: {
          enabled: true,
          endpoint: `${backendUrl}/digit/predict-full-image-shadow`,
          timeoutMs: 15000,
          shadowOnly: true,
          disableAfterFailures: 100
        }
      };
    }, BACKEND_URL);
    await page.goto(FRONTEND_URL, { waitUntil: 'networkidle' });
    await page.waitForSelector('#run-test-btn', { timeout: 30000 });
    await page.evaluate(() => {
      window.__jarvisOcrSelectionLogs = [];
      window.__jarvisLastTestSetHistogram = null;
      window.__jarvisLastTestSetRows = null;
    });
    await page.click('#run-test-btn');
    await page.waitForFunction(() => {
      const status = document.getElementById('test-status');
      return status && String(status.textContent || '').trim().startsWith('Done.');
    }, undefined, { timeout: 900000 });
    return await page.evaluate(() => ({
      status: String(document.getElementById('test-status')?.textContent || '').trim(),
      histogram: window.__jarvisLastTestSetHistogram || null,
      rows: Array.isArray(window.__jarvisLastTestSetRows)
        ? window.__jarvisLastTestSetRows
        : []
    }));
  } finally {
    await page.close();
    await browser.close();
  }
};

const sha256 = async (filePath) => {
  const hash = crypto.createHash('sha256');
  await new Promise((resolve, reject) => {
    const stream = fs.createReadStream(filePath);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('end', resolve);
    stream.on('error', reject);
  });
  return hash.digest('hex');
};

const buildRows = (uiRows, cvFolds) => uiRows.map((row) => {
  const shadow = row.selectionLog && row.selectionLog.fullImageDigitShadow
    ? row.selectionLog.fullImageDigitShadow
    : null;
  const shadowValue = shadow && shadow.value ? shadow.value : '';
  const candidates = shadow && Array.isArray(shadow.candidates) ? shadow.candidates : [];
  return {
    filename: row.filename,
    expected: row.expected,
    cv_fold: cvFolds.has(row.filename) ? cvFolds.get(row.filename) : null,
    production_value: row.detected || '',
    production_absolute_error: Number.isFinite(row.absoluteError) ? row.absoluteError : null,
    shadow_value: shadowValue,
    shadow_absolute_error: shadowValue
      ? Math.abs(Number(row.expected) - Number(shadowValue))
      : null,
    shadow_exact: shadowValue === row.expected,
    shadow_no_read: !shadowValue,
    orientation_oracle_hit: candidates.some((candidate) => candidate.value === row.expected),
    selected_rotation: shadow ? shadow.selectedRotation : null,
    orientation_source: shadow ? shadow.orientationSource : null,
    confidence: shadow ? shadow.confidence : null,
    detection_count: shadow ? shadow.detectionCount : 0,
    reason: shadow ? shadow.reason : 'no-shadow-log',
    candidates
  };
});

const writeReport = async (payload) => {
  const outputDir = path.join(OUTPUT_ROOT, timestampId());
  await fsp.mkdir(outputDir, { recursive: true });
  await fsp.writeFile(
    path.join(outputDir, 'summary.json'),
    `${JSON.stringify(payload, null, 2)}\n`,
    'utf8'
  );
  const lines = [
    '# Full-Image Digit Shadow UI Benchmark',
    '',
    `Generated ${payload.generated_at}.`,
    '',
    `- Production: ${payload.production_metrics.exact_match_count}/${payload.production_metrics.image_count} exact, ${payload.production_metrics.no_read_count} no-read, MAE ${payload.production_metrics.readable_mae}.`,
    `- Shadow: ${payload.shadow_metrics.exact_match_count}/${payload.shadow_metrics.image_count} exact, ${payload.shadow_metrics.no_read_count} no-read, MAE ${payload.shadow_metrics.readable_mae}.`,
    `- Orientation-oracle hits: ${payload.orientation_oracle_hit_count}/${payload.shadow_metrics.image_count}.`,
    `- Runtime digit settings: confidence ${payload.runtime_settings.confidence}, NMS IoU ${payload.runtime_settings.iou}, image size ${payload.runtime_settings.imgsz}.`,
    '',
    `Leakage-safe active checkpoint fold ${payload.checkpoint_validation_fold} (current source exclusions applied):`,
    '',
    `- Production: ${payload.validation_slice.production_metrics.exact_match_count}/${payload.validation_slice.production_metrics.image_count} exact, ${payload.validation_slice.production_metrics.no_read_count} no-read, MAE ${payload.validation_slice.production_metrics.readable_mae}.`,
    `- Shadow: ${payload.validation_slice.shadow_metrics.exact_match_count}/${payload.validation_slice.shadow_metrics.image_count} exact, ${payload.validation_slice.shadow_metrics.no_read_count} no-read, MAE ${payload.validation_slice.shadow_metrics.readable_mae}.`,
    '',
    `The complete ${payload.shadow_metrics.image_count}-image comparison is a development diagnostic, not an unbiased generalization estimate: ${payload.known_training_overlap_count} mapped images belong to folds used to train this checkpoint, and ${payload.unmapped_image_count} images have no assignment in the original training manifest.`,
    '',
    'The shadow is orientation-assisted by the current primary OCR angle and never changes the selected reading.',
    '',
    '| Image | CV fold | Expected | Production | Shadow | Shadow error | Rotation |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ...payload.rows.map((row) => (
      `| ${row.filename} | ${row.cv_fold ?? 'n/a'} | ${row.expected} | ${row.production_value || 'NO-READ'} | ${row.shadow_value || 'NO-READ'} | ${row.shadow_absolute_error ?? 'n/a'} | ${row.selected_rotation ?? 'n/a'} |`
    )),
    ''
  ];
  await fsp.writeFile(path.join(outputDir, 'README.md'), lines.join('\n'), 'utf8');
  return outputDir;
};

const selectValidationRows = (rows, provenance) => {
  const excluded = new Set(provenance.evaluation_exclusions.filenames);
  const selected = rows.filter((row) => (
    row.cv_fold === provenance.selected_fold && !excluded.has(row.filename)
  ));
  if (!selected.length) {
    throw new Error(`No active UI rows belong to checkpoint fold ${provenance.selected_fold}.`);
  }
  return selected;
};

const main = async () => {
  if (!fs.existsSync(CHECKPOINT_PATH)) {
    throw new Error(`Missing shadow checkpoint: ${CHECKPOINT_PATH}`);
  }
  const checkpointProvenance = validateCheckpointProvenance(
    CHECKPOINT_PATH, CV_FOLDS_PATH, process.env.FULL_IMAGE_DIGIT_SHADOW_VALIDATION_FOLD
  );
  const validationFold = checkpointProvenance.selected_fold;
  const cvFolds = new Map(Object.entries(checkpointProvenance.fold_assignments));
  let frontend = null;
  let backend = null;
  try {
    frontend = await ensureFrontend();
    const frontendSourceHashes = await validateFrontendSource(FRONTEND_URL);
    const started = await startBackend();
    backend = started.backend;
    const backendHealth = started.health;
    const ui = await runUiBenchmark();
    const rows = buildRows(ui.rows, cvFolds);
    const validationRows = selectValidationRows(rows, checkpointProvenance);
    const payload = {
      version: 1,
      generated_at: new Date().toISOString(),
      ui_status: ui.status,
      frontend_source_sha256: frontendSourceHashes,
      checkpoint: CHECKPOINT_PATH,
      checkpoint_sha256: await sha256(CHECKPOINT_PATH),
      checkpoint_validation_fold: validationFold,
      checkpoint_training_provenance: checkpointProvenance,
      evaluation_exclusions: checkpointProvenance.evaluation_exclusions,
      runtime_settings: {
        confidence: backendHealth.full_image_digit_shadow_confidence,
        iou: backendHealth.full_image_digit_shadow_iou,
        imgsz: backendHealth.full_image_digit_shadow_imgsz,
        max_detections: backendHealth.full_image_digit_shadow_max_detections,
        roi_expand_x: backendHealth.full_image_digit_shadow_roi_expand_x,
        roi_expand_y: backendHealth.full_image_digit_shadow_roi_expand_y
      },
      production_metrics: summarize(rows, 'production_value'),
      shadow_metrics: summarize(rows, 'shadow_value'),
      orientation_oracle_hit_count: rows.filter((row) => row.orientation_oracle_hit).length,
      validation_slice: {
        production_metrics: summarize(validationRows, 'production_value'),
        shadow_metrics: summarize(validationRows, 'shadow_value'),
        orientation_oracle_hit_count: validationRows.filter(
          (row) => row.orientation_oracle_hit
        ).length
      },
      known_training_overlap_count: rows.filter((row) => (
        Number.isFinite(row.cv_fold) && row.cv_fold !== validationFold
      )).length,
      unmapped_image_count: rows.filter((row) => !Number.isFinite(row.cv_fold)).length,
      rows
    };
    const outputDir = await writeReport(payload);
    process.stdout.write(`${JSON.stringify({ output: outputDir, ...payload }, null, 2)}\n`);
  } finally {
    if (backend) {
      await backend.stop();
    }
    if (frontend) {
      await frontend.stop();
    }
  }
};

if (require.main === module) {
  main().catch((error) => {
    process.stderr.write(`${error && error.stack ? error.stack : error}\n`);
    process.exitCode = 1;
  });
}

module.exports = { trackedProcess, waitFor, validateCheckpointProvenance, validateShadowBackendHealth, selectValidationRows, buildRows, summarize, validateFrontendSource };
