#!/usr/bin/env node

const crypto = require('node:crypto');
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const http = require('node:http');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { chromium } = require('playwright');

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
const CV_FOLDS_PATH = path.join(
  ROOT_DIR,
  'backend/data/full_image_digit_dataset/manifests/cv_folds.csv'
);
const checkpointFoldMatch = path.basename(path.dirname(path.dirname(CHECKPOINT_PATH))).match(/fold(\d+)/);
const CHECKPOINT_VALIDATION_FOLD = Number.parseInt(
  process.env.FULL_IMAGE_DIGIT_SHADOW_VALIDATION_FOLD
    || (checkpointFoldMatch ? checkpointFoldMatch[1] : ''),
  10
);

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

const requestOk = (url) => new Promise((resolve, reject) => {
  const request = http.get(url, (response) => {
    response.resume();
    const status = response.statusCode || 0;
    if (status >= 200 && status < 300) {
      resolve(true);
    } else {
      reject(new Error(`HTTP ${status}`));
    }
  });
  request.on('error', reject);
  request.setTimeout(3000, () => request.destroy(new Error('timeout')));
});

const trackedProcess = (command, args, options = {}) => {
  const child = spawn(command, args, {
    cwd: options.cwd || ROOT_DIR,
    env: options.env || process.env,
    stdio: ['ignore', 'pipe', 'pipe']
  });
  let stdout = '';
  let stderr = '';
  child.stdout.on('data', (chunk) => {
    stdout = `${stdout}${chunk}`.slice(-8000);
  });
  child.stderr.on('data', (chunk) => {
    stderr = `${stderr}${chunk}`.slice(-8000);
  });
  return {
    child,
    output: () => ({ stdout, stderr }),
    stop: async () => {
      if (child.exitCode !== null || child.signalCode !== null) {
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
  const deadline = Date.now() + timeoutMs;
  let lastError = null;
  while (Date.now() < deadline) {
    if (tracked && tracked.child.exitCode !== null) {
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
    if (!payload.roi_ready || !payload.digit_ready || !payload.full_image_digit_shadow_ready) {
      throw new Error(
        `health incomplete: roi=${!!payload.roi_ready} digit=${!!payload.digit_ready} `
          + `shadow=${!!payload.full_image_digit_shadow_ready}`
      );
    }
    if (path.resolve(payload.full_image_digit_shadow_model_path) !== CHECKPOINT_PATH) {
      throw new Error(`unexpected shadow model: ${payload.full_image_digit_shadow_model_path}`);
    }
    return payload;
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

const readCvFolds = async () => {
  const lines = (await fsp.readFile(CV_FOLDS_PATH, 'utf8'))
    .split(/\r?\n/)
    .filter(Boolean);
  const headers = (lines.shift() || '').split(',');
  const filenameIndex = headers.indexOf('filename');
  const foldIndex = headers.indexOf('fold');
  if (filenameIndex < 0 || foldIndex < 0) {
    throw new Error(`Invalid CV fold manifest: ${CV_FOLDS_PATH}`);
  }
  return new Map(lines.map((line) => {
    const values = line.split(',');
    return [values[filenameIndex], Number.parseInt(values[foldIndex], 10)];
  }));
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
    `Leakage-safe checkpoint fold ${payload.checkpoint_validation_fold}:`,
    '',
    `- Production: ${payload.validation_slice.production_metrics.exact_match_count}/${payload.validation_slice.production_metrics.image_count} exact, ${payload.validation_slice.production_metrics.no_read_count} no-read, MAE ${payload.validation_slice.production_metrics.readable_mae}.`,
    `- Shadow: ${payload.validation_slice.shadow_metrics.exact_match_count}/${payload.validation_slice.shadow_metrics.image_count} exact, ${payload.validation_slice.shadow_metrics.no_read_count} no-read, MAE ${payload.validation_slice.shadow_metrics.readable_mae}.`,
    '',
    `The complete ${payload.shadow_metrics.image_count}-image comparison is a development diagnostic, not an unbiased generalization estimate: ${payload.known_training_overlap_count} mapped images belong to folds used to train this checkpoint, and ${payload.unmapped_image_count} images have no active CV assignment.`,
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

const main = async () => {
  if (!fs.existsSync(CHECKPOINT_PATH)) {
    throw new Error(`Missing shadow checkpoint: ${CHECKPOINT_PATH}`);
  }
  let frontend = null;
  let backend = null;
  try {
    frontend = await ensureFrontend();
    const started = await startBackend();
    backend = started.backend;
    const backendHealth = started.health;
    const ui = await runUiBenchmark();
    const cvFolds = await readCvFolds();
    const rows = buildRows(ui.rows, cvFolds);
    if (!Number.isFinite(CHECKPOINT_VALIDATION_FOLD)) {
      throw new Error(
        'Cannot infer the checkpoint validation fold; set '
          + 'FULL_IMAGE_DIGIT_SHADOW_VALIDATION_FOLD.'
      );
    }
    const validationRows = rows.filter((row) => row.cv_fold === CHECKPOINT_VALIDATION_FOLD);
    if (!validationRows.length) {
      throw new Error(`No UI rows belong to checkpoint fold ${CHECKPOINT_VALIDATION_FOLD}.`);
    }
    const payload = {
      version: 1,
      generated_at: new Date().toISOString(),
      ui_status: ui.status,
      checkpoint: CHECKPOINT_PATH,
      checkpoint_sha256: await sha256(CHECKPOINT_PATH),
      checkpoint_validation_fold: CHECKPOINT_VALIDATION_FOLD,
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
        Number.isFinite(row.cv_fold) && row.cv_fold !== CHECKPOINT_VALIDATION_FOLD
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

main().catch((error) => {
  process.stderr.write(`${error && error.stack ? error.stack : error}\n`);
  process.exitCode = 1;
});
