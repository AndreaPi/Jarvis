#!/usr/bin/env node

const fs = require('node:fs');
const fsp = require('node:fs/promises');
const path = require('node:path');
const http = require('node:http');
const { spawn } = require('node:child_process');
const { chromium } = require('playwright');

const ROOT_DIR = path.resolve(__dirname, '..');
const OUTPUT_ROOT = path.join(ROOT_DIR, 'output', 'roi-checkpoint-diff');
const FRONTEND_URL = process.env.JARVIS_FRONTEND_URL || 'http://127.0.0.1:8000';
const BACKEND_URL = process.env.JARVIS_BACKEND_URL || 'http://127.0.0.1:8001';
const BACKEND_HEALTH_URL = `${BACKEND_URL}/health`;
const ENABLE_DIGIT_FALLBACK = process.env.JARVIS_DIGIT_FALLBACK === '1';

const parseHttpUrl = (raw, fallbackRaw) => {
  try {
    return new URL(raw);
  } catch {
    return new URL(fallbackRaw);
  }
};

const toBindConfig = (urlValue, fallbackRaw, fallbackPort) => {
  const parsed = parseHttpUrl(urlValue, fallbackRaw);
  const host = parsed.hostname || '127.0.0.1';
  const parsedPort = Number.parseInt(parsed.port, 10);
  const port = Number.isFinite(parsedPort) ? parsedPort : fallbackPort;
  return {
    host,
    port
  };
};

const FRONTEND_BIND = toBindConfig(FRONTEND_URL, 'http://127.0.0.1:8000', 8000);
const BACKEND_BIND = toBindConfig(BACKEND_URL, 'http://127.0.0.1:8001', 8001);

const MODEL_RUNS = [
  {
    id: 'baseline',
    label: 'roi-rotaug-e30-640.pt',
    modelPath: path.join(ROOT_DIR, 'backend', 'models', 'roi-rotaug-e30-640.pt')
  },
  {
    id: 'challenger',
    label: 'roi.pt',
    modelPath: path.join(ROOT_DIR, 'backend', 'models', 'roi.pt')
  }
];

const STAGE_NAMES = {
  strip: '5. detected strip crop',
  ocr: '6. OCR input candidate'
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const timestampId = () => {
  const now = new Date();
  const parts = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, '0'),
    String(now.getDate()).padStart(2, '0'),
    String(now.getHours()).padStart(2, '0'),
    String(now.getMinutes()).padStart(2, '0'),
    String(now.getSeconds()).padStart(2, '0')
  ];
  return `${parts[0]}${parts[1]}${parts[2]}-${parts[3]}${parts[4]}${parts[5]}`;
};

const sanitizeFileToken = (input) => {
  return String(input || '')
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    || 'unknown';
};

const stripFileExt = (input) => String(input || '').replace(/\.[^.]+$/, '');

const toRelativeFromRoot = (absPath) => path.relative(ROOT_DIR, absPath).replace(/\\/g, '/');

const parseDataUrl = (dataUrl) => {
  if (typeof dataUrl !== 'string') {
    return null;
  }
  const match = dataUrl.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)$/);
  if (!match) {
    return null;
  }
  const mime = match[1];
  const base64 = match[2];
  let extension = 'bin';
  if (mime === 'image/jpeg') {
    extension = 'jpg';
  } else if (mime === 'image/png') {
    extension = 'png';
  } else if (mime === 'image/webp') {
    extension = 'webp';
  }
  return {
    mime,
    extension,
    buffer: Buffer.from(base64, 'base64')
  };
};

const tailText = (value, maxLength = 5000) => {
  const text = String(value || '');
  return text.length <= maxLength ? text : text.slice(text.length - maxLength);
};

const spawnTrackedProcess = (command, args, options = {}) => {
  const child = spawn(command, args, {
    cwd: options.cwd || ROOT_DIR,
    env: options.env || process.env,
    stdio: ['ignore', 'pipe', 'pipe']
  });
  let stdout = '';
  let stderr = '';
  const appendOutput = (chunk, target) => {
    if (!chunk) {
      return target;
    }
    return tailText(`${target}${chunk.toString()}`);
  };
  child.stdout.on('data', (chunk) => {
    stdout = appendOutput(chunk, stdout);
  });
  child.stderr.on('data', (chunk) => {
    stderr = appendOutput(chunk, stderr);
  });
  return {
    child,
    getStdout: () => stdout,
    getStderr: () => stderr,
    stop: async () => {
      if (child.exitCode !== null || child.signalCode !== null) {
        return;
      }
      child.kill('SIGTERM');
      for (let i = 0; i < 30; i += 1) {
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

const requestJson = (url) => {
  return new Promise((resolve, reject) => {
    const req = http.get(url, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => {
        body += chunk;
      });
      res.on('end', () => {
        const statusCode = res.statusCode || 0;
        if (statusCode < 200 || statusCode >= 300) {
          reject(new Error(`HTTP ${statusCode}`));
          return;
        }
        try {
          resolve(JSON.parse(body));
        } catch (error) {
          reject(error);
        }
      });
    });
    req.on('error', reject);
    req.setTimeout(2500, () => {
      req.destroy(new Error('timeout'));
    });
  });
};

const requestOk = (url) => {
  return new Promise((resolve, reject) => {
    const req = http.get(url, (res) => {
      const statusCode = res.statusCode || 0;
      res.resume();
      if (statusCode >= 200 && statusCode < 300) {
        resolve(true);
        return;
      }
      reject(new Error(`HTTP ${statusCode}`));
    });
    req.on('error', reject);
    req.setTimeout(2500, () => {
      req.destroy(new Error('timeout'));
    });
  });
};

const waitForBackendReady = async (expectedModelPath, timeoutMs, trackedBackend) => {
  const deadline = Date.now() + timeoutMs;
  const expectedResolved = path.resolve(expectedModelPath);
  let lastError = null;
  while (Date.now() < deadline) {
    if (trackedBackend.child.exitCode !== null) {
      throw new Error(
        [
          'Backend exited unexpectedly before /health was ready.',
          `stdout:\n${trackedBackend.getStdout()}`,
          `stderr:\n${trackedBackend.getStderr()}`
        ].join('\n')
      );
    }
    try {
      const health = await requestJson(BACKEND_HEALTH_URL);
      const modelPath = typeof health.model_path === 'string' ? path.resolve(health.model_path) : null;
      const ready = !!health.roi_ready;
      if (ready && modelPath === expectedResolved) {
        return health;
      }
      lastError = new Error(`health not ready for expected model (${modelPath || 'unknown'})`);
    } catch (error) {
      lastError = error;
    }
    await sleep(500);
  }
  throw new Error(
    [
      `Timed out waiting for backend /health to serve ${expectedResolved}.`,
      `Last error: ${lastError ? String(lastError.message || lastError) : 'unknown'}`,
      `stdout:\n${trackedBackend.getStdout()}`,
      `stderr:\n${trackedBackend.getStderr()}`
    ].join('\n')
  );
};

const ensureFrontendAvailable = async (timeoutMs = 120000) => {
  try {
    await requestOk(FRONTEND_URL);
    return {
      reused: true,
      process: null
    };
  } catch {
    // Start local server below.
  }

  const tracked = spawnTrackedProcess(
    'python3',
    ['-m', 'http.server', String(FRONTEND_BIND.port), '--bind', FRONTEND_BIND.host],
    {
    cwd: ROOT_DIR
    }
  );

  const deadline = Date.now() + timeoutMs;
  let lastError = null;
  while (Date.now() < deadline) {
    if (tracked.child.exitCode !== null) {
      throw new Error(
        [
          'Frontend server failed to start.',
          `stdout:\n${tracked.getStdout()}`,
          `stderr:\n${tracked.getStderr()}`
        ].join('\n')
      );
    }
    try {
      await requestOk(FRONTEND_URL);
      return {
        reused: false,
        process: tracked
      };
    } catch (error) {
      lastError = error;
    }
    await sleep(400);
  }

  await tracked.stop();
  throw new Error(`Timed out waiting for frontend server: ${String(lastError || 'unknown error')}`);
};

const computeTopRejectReason = (selectionLog) => {
  if (!selectionLog || !Array.isArray(selectionLog.rejectSummary) || !selectionLog.rejectSummary.length) {
    return null;
  }
  const top = selectionLog.rejectSummary[0];
  if (!top || !top.reason) {
    return null;
  }
  return String(top.reason);
};

const runUiTestSet = async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  try {
    if (ENABLE_DIGIT_FALLBACK) {
      await page.addInitScript(() => {
        window.__JARVIS_OCR_CONFIG_OVERRIDE__ = {
          digitClassifier: {
            enabled: true,
            fallbackOnNoDigitsOnly: true
          }
        };
      });
    }
    await page.goto(FRONTEND_URL, { waitUntil: 'networkidle' });
    await page.waitForSelector('#run-test-btn', { timeout: 30000 });
    await page.evaluate(() => {
      window.__jarvisOcrSelectionLogs = [];
      window.__jarvisLastTestSetHistogram = null;
      const debugToggle = document.getElementById('debug-overlay-toggle');
      if (debugToggle && !debugToggle.checked) {
        debugToggle.checked = true;
        debugToggle.dispatchEvent(new Event('change', { bubbles: true }));
      }
      const clearButton = document.getElementById('clear-debug-btn');
      if (clearButton) {
        clearButton.click();
      }
    });

    await page.click('#run-test-btn');
    await page.waitForFunction(() => {
      const statusEl = document.getElementById('test-status');
      if (!statusEl) {
        return false;
      }
      const text = (statusEl.textContent || '').trim();
      return text.startsWith('Done.');
    }, undefined, { timeout: 600000 });

    const payload = await page.evaluate(() => {
      const text = (node) => (node && node.textContent ? node.textContent.trim() : '');
      const tableRows = [];
      const table = document.querySelector('#test-results table');
      if (table) {
        const rows = Array.from(table.querySelectorAll('tr')).slice(1);
        rows.forEach((row) => {
          const cells = Array.from(row.querySelectorAll('td')).map((cell) => text(cell));
          if (cells.length < 6) {
            return;
          }
          tableRows.push({
            filename: cells[0],
            expected: cells[1],
            detected: cells[2] === '—' ? '' : cells[2],
            absoluteError: cells[3] === '—' ? '' : cells[3],
            failureReason: cells[4] === '—' ? '' : cells[4],
            result: cells[5]
          });
        });
      }

      const sessions = Array.from(document.querySelectorAll('.debug-session')).map((session) => {
        const label = text(session.querySelector('.debug-session-title'));
        const stages = Array.from(session.querySelectorAll('.debug-stage')).map((stage) => ({
          name: text(stage.querySelector('.debug-stage-name')),
          dataUrl: stage.querySelector('img') ? stage.querySelector('img').src : ''
        }));
        return { label, stages };
      });

      return {
        status: text(document.getElementById('test-status')),
        rows: tableRows,
        histogram: window.__jarvisLastTestSetHistogram || null,
        selectionLogs: Array.isArray(window.__jarvisOcrSelectionLogs) ? window.__jarvisOcrSelectionLogs : [],
        sessions
      };
    });
    return payload;
  } finally {
    await page.close();
    await browser.close();
  }
};

const exportStageImages = async (sessions, runOutputDir, runId) => {
  const stageDir = path.join(runOutputDir, 'stages');
  await fsp.mkdir(stageDir, { recursive: true });
  const stageIndex = new Map();

  for (const session of sessions || []) {
    const label = String(session && session.label ? session.label : '').trim();
    if (!label) {
      continue;
    }
    const key = label;
    const nameBase = sanitizeFileToken(stripFileExt(label));
    const stages = Array.isArray(session.stages) ? session.stages : [];
    const wanted = stages.filter((stage) => stage && (
      stage.name === STAGE_NAMES.strip
      || stage.name === STAGE_NAMES.ocr
    ));
    const entry = {};
    for (const stage of wanted) {
      const parsed = parseDataUrl(stage.dataUrl);
      if (!parsed) {
        continue;
      }
      const stageSlug = stage.name === STAGE_NAMES.strip ? 'stage5_strip' : 'stage6_ocr';
      const fileName = `${nameBase}_${stageSlug}_${runId}.${parsed.extension}`;
      const absPath = path.join(stageDir, fileName);
      await fsp.writeFile(absPath, parsed.buffer);
      const relPath = toRelativeFromRoot(absPath);
      if (stage.name === STAGE_NAMES.strip) {
        entry.stage5 = relPath;
      } else if (stage.name === STAGE_NAMES.ocr) {
        entry.stage6 = relPath;
      }
    }
    stageIndex.set(key, entry);
  }
  return stageIndex;
};

const rowsByFilename = (rows) => {
  const map = new Map();
  for (const row of rows || []) {
    if (!row || !row.filename) {
      continue;
    }
    map.set(String(row.filename), row);
  }
  return map;
};

const logsByImage = (logs) => {
  const map = new Map();
  for (const log of logs || []) {
    if (!log || !log.image) {
      continue;
    }
    map.set(String(log.image), log);
  }
  return map;
};

const classifyReason = (reason) => {
  const text = String(reason || '').toLowerCase();
  if (!text || text === '—') {
    return null;
  }
  if (text === 'mismatch') {
    return 'mismatch';
  }
  if (text.includes('ocr-no-digits')) {
    return 'ocr-no-digits';
  }
  if (text.includes('no-detection')) {
    return 'no-detection';
  }
  return text;
};

const parseMeterValue = (value) => {
  const digits = String(value || '').replace(/\D/g, '');
  if (!digits) {
    return null;
  }
  const parsed = Number.parseInt(digits, 10);
  return Number.isFinite(parsed) ? parsed : null;
};

const computeRowAbsoluteError = (row) => {
  if (!row) {
    return null;
  }
  const expectedValue = parseMeterValue(row.expected);
  const detectedValue = parseMeterValue(row.detected);
  if (!Number.isFinite(expectedValue) || !Number.isFinite(detectedValue)) {
    return null;
  }
  return Math.abs(expectedValue - detectedValue);
};

const comparisonErrorValue = (row) => {
  const absoluteError = computeRowAbsoluteError(row);
  return Number.isFinite(absoluteError) ? absoluteError : Number.POSITIVE_INFINITY;
};

const buildFailureHistogram = (rows) => {
  const counts = new Map();
  for (const row of rows || []) {
    if (!row || row.result === 'Pass') {
      continue;
    }
    const reason = row.failureReason || 'unknown';
    counts.set(reason, (counts.get(reason) || 0) + 1);
  }
  return [...counts.entries()]
    .map(([reason, count]) => ({ reason, count }))
    .sort((a, b) => b.count - a.count || a.reason.localeCompare(b.reason));
};

const computeMetrics = (rows) => {
  const total = rows.length;
  let exactMatchCount = 0;
  let noReadCount = 0;
  let maeSum = 0;
  let maeRows = 0;
  let mismatch = 0;
  let ocrNoDigits = 0;
  let noDetection = 0;

  for (const row of rows) {
    const hasDetectedValue = Boolean(String(row.detected || '').trim());
    if (!hasDetectedValue) {
      noReadCount += 1;
    }

    if (row.result === 'Pass') {
      exactMatchCount += 1;
    } else {
      const reasonClass = classifyReason(row.failureReason);
      if (reasonClass === 'mismatch') {
        mismatch += 1;
      }
      if (reasonClass === 'ocr-no-digits') {
        ocrNoDigits += 1;
      }
      if (reasonClass === 'no-detection') {
        noDetection += 1;
      }
    }

    const absoluteError = computeRowAbsoluteError(row);
    if (Number.isFinite(absoluteError)) {
      maeSum += absoluteError;
      maeRows += 1;
    }
  }

  const mae = maeRows ? maeSum / maeRows : null;
  const exactMatchRate = total ? exactMatchCount / total : 0;
  const noReadRate = total ? noReadCount / total : 0;

  return {
    total,
    correct: exactMatchCount,
    failed: total - exactMatchCount,
    accuracy: exactMatchRate,
    exactMatchCount,
    exactMatchRate,
    noReadCount,
    noReadRate,
    mae,
    maeRows,
    maeSum,
    mismatch,
    ocrNoDigits,
    noDetection,
    failureReasons: buildFailureHistogram(rows)
  };
};

const formatPct = (value) => `${(value * 100).toFixed(1)}%`;
const formatSigned = (value, digits = 3) => `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
const formatMae = (value) => (Number.isFinite(value) ? value.toFixed(2) : 'n/a');

const markdownEscape = (value) => String(value || '').replace(/\|/g, '\\|');

const buildComparison = (baselineRun, challengerRun) => {
  const baselineRows = rowsByFilename(baselineRun.rows);
  const challengerRows = rowsByFilename(challengerRun.rows);
  const baselineLogs = logsByImage(baselineRun.selectionLogs);
  const challengerLogs = logsByImage(challengerRun.selectionLogs);

  const names = [...new Set([...baselineRows.keys(), ...challengerRows.keys()])].sort((a, b) => a.localeCompare(b));
  const rows = names.map((filename) => {
    const base = baselineRows.get(filename) || null;
    const next = challengerRows.get(filename) || null;
    const baseLog = baselineLogs.get(filename) || null;
    const nextLog = challengerLogs.get(filename) || null;
    return {
      filename,
      expected: (base && base.expected) || (next && next.expected) || '',
      baseline: base ? {
        detected: base.detected || '',
        failureReason: base.failureReason || '',
        result: base.result || '',
        absoluteError: base.absoluteError || '',
        topRejectReason: computeTopRejectReason(baseLog)
      } : null,
      challenger: next ? {
        detected: next.detected || '',
        failureReason: next.failureReason || '',
        result: next.result || '',
        absoluteError: next.absoluteError || '',
        topRejectReason: computeTopRejectReason(nextLog)
      } : null
    };
  });

  const EPSILON = 1e-9;
  const improved = rows.filter((row) => {
    if (!row.baseline || !row.challenger) {
      return false;
    }
    const baselineError = comparisonErrorValue({
      expected: row.expected,
      detected: row.baseline.detected
    });
    const challengerError = comparisonErrorValue({
      expected: row.expected,
      detected: row.challenger.detected
    });
    return challengerError < baselineError - EPSILON;
  }).length;
  const regressed = rows.filter((row) => {
    if (!row.baseline || !row.challenger) {
      return false;
    }
    const baselineError = comparisonErrorValue({
      expected: row.expected,
      detected: row.baseline.detected
    });
    const challengerError = comparisonErrorValue({
      expected: row.expected,
      detected: row.challenger.detected
    });
    return challengerError > baselineError + EPSILON;
  }).length;

  return {
    rows,
    improved,
    regressed
  };
};

const renderMarkdownReport = ({
  generatedAt,
  fallbackEnabled,
  baselineRun,
  challengerRun,
  baselineMetrics,
  challengerMetrics,
  comparisonRows,
  improved,
  regressed,
  outputDir
}) => {
  const lines = [];
  lines.push('# ROI Checkpoint Comparison');
  lines.push('');
  lines.push(`- Generated: ${generatedAt}`);
  lines.push(`- Baseline model: \`${baselineRun.model.label}\` (\`${baselineRun.health.model_path}\`)`);
  lines.push(`- Challenger model: \`${challengerRun.model.label}\` (\`${challengerRun.health.model_path}\`)`);
  lines.push(`- Digit classifier fallback: \`${fallbackEnabled ? 'enabled' : 'disabled'}\``);
  lines.push(`- Output directory: \`${toRelativeFromRoot(outputDir)}\``);
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push('| Metric | Baseline | Challenger | Delta |');
  lines.push('| --- | ---: | ---: | ---: |');
  lines.push(`| MAE (lower is better) | ${formatMae(baselineMetrics.mae)} | ${formatMae(challengerMetrics.mae)} | ${Number.isFinite(baselineMetrics.mae) && Number.isFinite(challengerMetrics.mae) ? formatSigned(challengerMetrics.mae - baselineMetrics.mae, 2) : 'n/a'} |`);
  lines.push(`| MAE rows | ${baselineMetrics.maeRows}/${baselineMetrics.total} | ${challengerMetrics.maeRows}/${challengerMetrics.total} | ${challengerMetrics.maeRows - baselineMetrics.maeRows} |`);
  lines.push(`| Exact Match (guardrail) | ${baselineMetrics.exactMatchCount}/${baselineMetrics.total} (${formatPct(baselineMetrics.exactMatchRate)}) | ${challengerMetrics.exactMatchCount}/${challengerMetrics.total} (${formatPct(challengerMetrics.exactMatchRate)}) | ${formatSigned((challengerMetrics.exactMatchRate - baselineMetrics.exactMatchRate) * 100, 1)}% |`);
  lines.push(`| No-read (guardrail, lower is better) | ${baselineMetrics.noReadCount}/${baselineMetrics.total} (${formatPct(baselineMetrics.noReadRate)}) | ${challengerMetrics.noReadCount}/${challengerMetrics.total} (${formatPct(challengerMetrics.noReadRate)}) | ${formatSigned((challengerMetrics.noReadRate - baselineMetrics.noReadRate) * 100, 1)}% |`);
  lines.push(`| mismatch | ${baselineMetrics.mismatch} | ${challengerMetrics.mismatch} | ${challengerMetrics.mismatch - baselineMetrics.mismatch} |`);
  lines.push(`| ocr-no-digits | ${baselineMetrics.ocrNoDigits} | ${challengerMetrics.ocrNoDigits} | ${challengerMetrics.ocrNoDigits - baselineMetrics.ocrNoDigits} |`);
  lines.push(`| no-detection | ${baselineMetrics.noDetection} | ${challengerMetrics.noDetection} | ${challengerMetrics.noDetection - baselineMetrics.noDetection} |`);
  lines.push(`| Improved images (by absolute error) | - | ${improved} | - |`);
  lines.push(`| Regressed images (by absolute error) | - | ${regressed} | - |`);
  lines.push('');
  lines.push('## Failure Histogram');
  lines.push('');
  lines.push('### Baseline');
  lines.push('');
  lines.push('| Reason | Count |');
  lines.push('| --- | ---: |');
  baselineMetrics.failureReasons.forEach((entry) => {
    lines.push(`| ${markdownEscape(entry.reason)} | ${entry.count} |`);
  });
  lines.push('');
  lines.push('### Challenger');
  lines.push('');
  lines.push('| Reason | Count |');
  lines.push('| --- | ---: |');
  challengerMetrics.failureReasons.forEach((entry) => {
    lines.push(`| ${markdownEscape(entry.reason)} | ${entry.count} |`);
  });
  lines.push('');
  lines.push('## Per-image Diff');
  lines.push('');
  lines.push('| File | Expected | Baseline detected | Challenger detected | Baseline abs error | Challenger abs error | Baseline reason | Challenger reason | Baseline reject | Challenger reject | Baseline stage5 | Challenger stage5 | Baseline stage6 | Challenger stage6 |');
  lines.push('| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- | --- |');

  comparisonRows.forEach((row) => {
    const base = row.baseline || {};
    const next = row.challenger || {};
    const baseStage = baselineRun.stageIndex.get(row.filename) || {};
    const nextStage = challengerRun.stageIndex.get(row.filename) || {};
    const baseAbsoluteError = computeRowAbsoluteError({
      expected: row.expected,
      detected: base.detected || ''
    });
    const nextAbsoluteError = computeRowAbsoluteError({
      expected: row.expected,
      detected: next.detected || ''
    });
    const cols = [
      markdownEscape(row.filename),
      markdownEscape(row.expected),
      markdownEscape(base.detected || '—'),
      markdownEscape(next.detected || '—'),
      markdownEscape(Number.isFinite(baseAbsoluteError) ? String(baseAbsoluteError) : '—'),
      markdownEscape(Number.isFinite(nextAbsoluteError) ? String(nextAbsoluteError) : '—'),
      markdownEscape(base.failureReason || '—'),
      markdownEscape(next.failureReason || '—'),
      markdownEscape(base.topRejectReason || '—'),
      markdownEscape(next.topRejectReason || '—'),
      markdownEscape(baseStage.stage5 || '—'),
      markdownEscape(nextStage.stage5 || '—'),
      markdownEscape(baseStage.stage6 || '—'),
      markdownEscape(nextStage.stage6 || '—')
    ];
    lines.push(`| ${cols.join(' | ')} |`);
  });

  lines.push('');
  lines.push('## Notes');
  lines.push('');
  lines.push('- `topRejectReason` comes from each image selection log (`rejectSummary[0].reason`).');
  lines.push('- Stage snapshot paths are written relative to repository root and saved from debug overlay image data.');
  lines.push('');

  return `${lines.join('\n')}\n`;
};

const runCheckpoint = async (model, runDir) => {
  const backendProc = spawnTrackedProcess(
    './.venv/bin/uvicorn',
    ['app:app', '--host', BACKEND_BIND.host, '--port', String(BACKEND_BIND.port)],
    {
      cwd: path.join(ROOT_DIR, 'backend'),
      env: {
        ...process.env,
        ROI_MODEL_PATH: model.modelPath
      }
    }
  );

  try {
    const health = await waitForBackendReady(model.modelPath, 120000, backendProc);
    const uiResult = await runUiTestSet();
    const stageIndex = await exportStageImages(uiResult.sessions || [], runDir, model.id);
    return {
      model,
      health,
      rows: uiResult.rows || [],
      histogram: uiResult.histogram || null,
      selectionLogs: uiResult.selectionLogs || [],
      status: uiResult.status || '',
      stageIndex,
      backendStdoutTail: backendProc.getStdout(),
      backendStderrTail: backendProc.getStderr()
    };
  } finally {
    await backendProc.stop();
  }
};

const writeJson = async (filePath, value) => {
  await fsp.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
};

const ensureModelFilesExist = () => {
  const missing = MODEL_RUNS.filter((run) => !fs.existsSync(run.modelPath));
  if (!missing.length) {
    return;
  }
  const list = missing.map((item) => `- ${item.modelPath}`).join('\n');
  throw new Error(`Missing required model files:\n${list}`);
};

const run = async () => {
  ensureModelFilesExist();
  const runId = timestampId();
  const modeSuffix = ENABLE_DIGIT_FALLBACK ? 'fallback-on' : 'fallback-off';
  const outputDir = path.join(OUTPUT_ROOT, `${runId}-${modeSuffix}`);
  await fsp.mkdir(outputDir, { recursive: true });

  const frontend = await ensureFrontendAvailable();

  try {
    const baselineDir = path.join(outputDir, MODEL_RUNS[0].id);
    const challengerDir = path.join(outputDir, MODEL_RUNS[1].id);
    await fsp.mkdir(baselineDir, { recursive: true });
    await fsp.mkdir(challengerDir, { recursive: true });

    const baselineRun = await runCheckpoint(MODEL_RUNS[0], baselineDir);
    const challengerRun = await runCheckpoint(MODEL_RUNS[1], challengerDir);

    const baselineMetrics = computeMetrics(baselineRun.rows);
    const challengerMetrics = computeMetrics(challengerRun.rows);
    const comparison = buildComparison(baselineRun, challengerRun);

    const generatedAt = new Date().toISOString();
    const reportJson = {
      generatedAt,
      fallbackEnabled: ENABLE_DIGIT_FALLBACK,
      outputDir: toRelativeFromRoot(outputDir),
      baseline: {
        model: baselineRun.model,
        health: baselineRun.health,
        status: baselineRun.status,
        metrics: baselineMetrics,
        rows: baselineRun.rows
      },
      challenger: {
        model: challengerRun.model,
        health: challengerRun.health,
        status: challengerRun.status,
        metrics: challengerMetrics,
        rows: challengerRun.rows
      },
      comparison: {
        improved: comparison.improved,
        regressed: comparison.regressed,
        rows: comparison.rows
      }
    };

    const markdown = renderMarkdownReport({
      generatedAt,
      fallbackEnabled: ENABLE_DIGIT_FALLBACK,
      baselineRun,
      challengerRun,
      baselineMetrics,
      challengerMetrics,
      comparisonRows: comparison.rows,
      improved: comparison.improved,
      regressed: comparison.regressed,
      outputDir
    });

    const jsonPath = path.join(outputDir, 'roi-diff-report.json');
    const mdPath = path.join(outputDir, 'roi-diff-report.md');
    await writeJson(jsonPath, reportJson);
    await fsp.writeFile(mdPath, markdown, 'utf8');

    const consoleSummary = {
      outputDir: toRelativeFromRoot(outputDir),
      fallbackEnabled: ENABLE_DIGIT_FALLBACK,
      markdownReport: toRelativeFromRoot(mdPath),
      jsonReport: toRelativeFromRoot(jsonPath),
      baselinePrimary: {
        mae: Number.isFinite(baselineMetrics.mae) ? Number(baselineMetrics.mae.toFixed(4)) : null,
        maeRows: baselineMetrics.maeRows
      },
      challengerPrimary: {
        mae: Number.isFinite(challengerMetrics.mae) ? Number(challengerMetrics.mae.toFixed(4)) : null,
        maeRows: challengerMetrics.maeRows
      },
      guardrails: {
        baselineExactMatchRate: Number((baselineMetrics.exactMatchRate * 100).toFixed(2)),
        challengerExactMatchRate: Number((challengerMetrics.exactMatchRate * 100).toFixed(2)),
        baselineNoReadRate: Number((baselineMetrics.noReadRate * 100).toFixed(2)),
        challengerNoReadRate: Number((challengerMetrics.noReadRate * 100).toFixed(2))
      },
      baselineFailureMix: {
        mismatch: baselineMetrics.mismatch,
        ocrNoDigits: baselineMetrics.ocrNoDigits,
        noDetection: baselineMetrics.noDetection
      },
      challengerFailureMix: {
        mismatch: challengerMetrics.mismatch,
        ocrNoDigits: challengerMetrics.ocrNoDigits,
        noDetection: challengerMetrics.noDetection
      },
      improved: comparison.improved,
      regressed: comparison.regressed
    };

    process.stdout.write(`${JSON.stringify(consoleSummary, null, 2)}\n`);
  } finally {
    if (frontend.process) {
      await frontend.process.stop();
    }
  }
};

run().catch((error) => {
  process.stderr.write(`${error && error.stack ? error.stack : String(error)}\n`);
  process.exitCode = 1;
});
