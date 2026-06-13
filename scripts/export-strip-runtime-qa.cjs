#!/usr/bin/env node

const fs = require('node:fs');
const fsp = require('node:fs/promises');
const path = require('node:path');
const http = require('node:http');
const { spawn } = require('node:child_process');
const { chromium } = require('@playwright/test');

const ROOT_DIR = path.resolve(__dirname, '..');
const FRONTEND_URL = process.env.JARVIS_FRONTEND_URL || 'http://127.0.0.1:8000';
const BACKEND_URL = process.env.JARVIS_BACKEND_URL || 'http://127.0.0.1:8001';
const OUTPUT_ROOT = path.join(ROOT_DIR, 'output', 'strip-runtime-qa');
const DEFAULT_TARGETS = [
  'meter_20260327.JPEG',
  'meter_20260409.JPEG',
  'meter_20260413.JPEG',
  'meter_20260416.JPEG',
  'meter_20260420.JPEG',
  'meter_20260423.JPEG',
  'meter_20260427.JPEG'
];
const STAGE_EXPORTS = [
  { name: '5. detected strip crop', slug: 'stage5-strip', title: 'Stage 5 strip' },
  { name: '6. OCR input candidate', slug: 'stage6-ocr', title: 'Stage 6 OCR input' },
  { name: '7. classifier cell crops', slug: 'stage7-cells', title: 'Stage 7 cells' },
  { name: '8. strip reader input', slug: 'stage8-strip-reader', title: 'Stage 8 strip reader' }
];

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

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

const sanitizeFileToken = (input) => String(input || '')
  .trim()
  .replace(/[^a-zA-Z0-9._-]+/g, '_')
  .replace(/^_+|_+$/g, '')
  || 'unknown';

const stripFileExt = (input) => String(input || '').replace(/\.[^.]+$/, '');

const parseDataUrl = (dataUrl) => {
  if (typeof dataUrl !== 'string') {
    return null;
  }
  const match = dataUrl.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)$/);
  if (!match) {
    return null;
  }
  const mime = match[1];
  const extension = mime === 'image/png' ? 'png' : 'jpg';
  return {
    extension,
    buffer: Buffer.from(match[2], 'base64')
  };
};

const parseCsv = (text) => {
  const rows = [];
  const lines = String(text || '').split(/\r?\n/).filter((line) => line.trim());
  if (!lines.length) {
    return rows;
  }
  const headers = lines[0].split(',').map((header) => header.trim());
  for (const line of lines.slice(1)) {
    const values = line.split(',');
    const row = {};
    headers.forEach((header, index) => {
      row[header] = (values[index] || '').trim();
    });
    rows.push(row);
  }
  return rows;
};

const requestOk = (url) => new Promise((resolve) => {
  const request = http.get(url, (response) => {
    response.resume();
    resolve((response.statusCode || 0) >= 200 && (response.statusCode || 0) < 300);
  });
  request.on('error', () => resolve(false));
  request.setTimeout(1000, () => {
    request.destroy();
    resolve(false);
  });
});

const spawnTrackedProcess = (command, args, options = {}) => {
  const child = spawn(command, args, {
    cwd: options.cwd || ROOT_DIR,
    env: options.env || process.env,
    stdio: ['ignore', 'pipe', 'pipe']
  });
  child.stdout.on('data', (chunk) => process.stdout.write(`[${options.label || command}] ${chunk}`));
  child.stderr.on('data', (chunk) => process.stderr.write(`[${options.label || command}] ${chunk}`));
  return {
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

const ensureServices = async () => {
  const processes = [];
  if (!(await requestOk(FRONTEND_URL))) {
    processes.push(spawnTrackedProcess('npm', ['run', 'serve'], {
      label: 'frontend'
    }));
  }
  if (!(await requestOk(`${BACKEND_URL}/health`))) {
    processes.push(spawnTrackedProcess(
      path.join(ROOT_DIR, 'backend', '.venv', 'bin', 'uvicorn'),
      ['backend.app:app', '--host', '127.0.0.1', '--port', '8001'],
      { label: 'backend' }
    ));
  }

  for (let attempt = 0; attempt < 80; attempt += 1) {
    if ((await requestOk(FRONTEND_URL)) && (await requestOk(`${BACKEND_URL}/health`))) {
      return processes;
    }
    await sleep(250);
  }
  throw new Error('Timed out waiting for frontend/backend services.');
};

const readMeters = async () => {
  const csvPath = path.join(ROOT_DIR, 'assets', 'meter_readings.csv');
  const rows = parseCsv(await fsp.readFile(csvPath, 'utf8'));
  const map = new Map();
  rows.forEach((row) => {
    if (row.filename) {
      map.set(row.filename, row.value || '');
    }
  });
  return map;
};

const readCanonicalManifest = async () => {
  const manifestPath = path.join(ROOT_DIR, 'backend', 'data', 'digit_dataset', 'manifests', 'canonical_windows.csv');
  const rows = parseCsv(await fsp.readFile(manifestPath, 'utf8'));
  const map = new Map();
  rows.forEach((row) => {
    if (row.filename && row.canonical_window_path) {
      map.set(row.filename, row);
    }
  });
  return map;
};

const writeDataUrlImage = async (stage, outputDir, filename, slug) => {
  const parsed = parseDataUrl(stage && stage.dataUrl);
  if (!parsed) {
    return null;
  }
  const outputPath = path.join(outputDir, `${sanitizeFileToken(stripFileExt(filename))}_${slug}.${parsed.extension}`);
  await fsp.writeFile(outputPath, parsed.buffer);
  return outputPath;
};

const copyCanonicalWindow = async (manifestRow, outputDir, filename) => {
  if (!manifestRow || !manifestRow.canonical_window_path) {
    return null;
  }
  const sourcePath = path.join(
    ROOT_DIR,
    'backend',
    'data',
    'digit_dataset',
    manifestRow.canonical_window_path
  );
  if (!fs.existsSync(sourcePath)) {
    return null;
  }
  const outputPath = path.join(outputDir, `${sanitizeFileToken(stripFileExt(filename))}_canonical.png`);
  await fsp.copyFile(sourcePath, outputPath);
  return outputPath;
};

const collectImageQa = async (page, filename, outputDir, readings, canonicalRows) => {
  await page.evaluate(() => {
    window.__jarvisOcrSelectionLogs = [];
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

  const runPayload = await page.evaluate(async ({ target }) => {
    const { runMeterOcr } = await import('/src/ocr/pipeline.js');
    const response = await fetch(`/assets/${target}`, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`Unable to fetch asset ${target}`);
    }
    const blob = await response.blob();
    const file = new File([blob], target, { type: blob.type || 'image/jpeg' });
    let result = null;
    let error = null;
    try {
      result = await runMeterOcr(file);
    } catch (caught) {
      error = caught && caught.message ? caught.message : String(caught);
    }
    const logs = Array.isArray(window.__jarvisOcrSelectionLogs) ? window.__jarvisOcrSelectionLogs : [];
    return {
      result: result ? { value: result.value || '' } : null,
      error,
      selectionLog: logs.length ? logs[logs.length - 1] : null
    };
  }, { target: filename });

  const session = await page.evaluate(() => {
    const text = (node) => (node && node.textContent ? node.textContent.trim() : '');
    const sessionEl = document.querySelector('.debug-session');
    if (!sessionEl) {
      return null;
    }
    return {
      label: text(sessionEl.querySelector('.debug-session-title')),
      stages: Array.from(sessionEl.querySelectorAll('.debug-stage')).map((stageEl) => ({
        name: text(stageEl.querySelector('.debug-stage-name')),
        dataUrl: stageEl.querySelector('img') ? stageEl.querySelector('img').src : ''
      }))
    };
  });

  const stageFiles = {};
  for (const stageConfig of STAGE_EXPORTS) {
    const matchingStages = (session && Array.isArray(session.stages) ? session.stages : [])
      .filter((stage) => stage && stage.name === stageConfig.name);
    const stage = matchingStages.length ? matchingStages[matchingStages.length - 1] : null;
    stageFiles[stageConfig.slug] = await writeDataUrlImage(stage, outputDir, filename, stageConfig.slug);
  }
  const canonicalPath = await copyCanonicalWindow(canonicalRows.get(filename), outputDir, filename);
  const expected = readings.get(filename) || '';
  const selected = runPayload.selectionLog && runPayload.selectionLog.selected
    ? runPayload.selectionLog.selected
    : null;
  const stripReader = runPayload.selectionLog && runPayload.selectionLog.stripReader
    ? runPayload.selectionLog.stripReader
    : null;
  const stripReader23xx = runPayload.selectionLog && runPayload.selectionLog.stripReader23xx
    ? runPayload.selectionLog.stripReader23xx
    : null;

  return {
    filename,
    expected,
    detected: selected && selected.value ? selected.value : '',
    primarySource: selected && selected.sourceLabel ? selected.sourceLabel : '',
    primaryMethod: selected && selected.method ? selected.method : '',
    stripValue: stripReader && stripReader.value ? stripReader.value : '',
    stripConfidence: Number.isFinite(stripReader && stripReader.confidence) ? stripReader.confidence : null,
    stripSource: stripReader && stripReader.sourceLabel ? stripReader.sourceLabel : '',
    stripHeadlineReason: stripReader && stripReader.headlineReason ? stripReader.headlineReason : '',
    stripCandidateCount: stripReader && Array.isArray(stripReader.candidates) ? stripReader.candidates.length : 0,
    stripConfidenceBestValue: stripReader && stripReader.confidenceBest && stripReader.confidenceBest.value
      ? stripReader.confidenceBest.value
      : '',
    stripSelectedSourceValue: stripReader && stripReader.selectedSourceCandidate && stripReader.selectedSourceCandidate.value
      ? stripReader.selectedSourceCandidate.value
      : '',
    strip23xxAccepted: Boolean(stripReader23xx && stripReader23xx.accepted),
    strip23xxValue: stripReader23xx && stripReader23xx.value ? stripReader23xx.value : '',
    strip23xxPredictedValue: stripReader23xx && stripReader23xx.predictedValue ? stripReader23xx.predictedValue : '',
    strip23xxConfidence: Number.isFinite(stripReader23xx && stripReader23xx.confidence) ? stripReader23xx.confidence : null,
    strip23xxGuardConfidence: Number.isFinite(stripReader23xx && stripReader23xx.guardConfidence)
      ? stripReader23xx.guardConfidence
      : null,
    strip23xxSource: stripReader23xx && stripReader23xx.sourceLabel ? stripReader23xx.sourceLabel : '',
    strip23xxHeadlineReason: stripReader23xx && stripReader23xx.headlineReason ? stripReader23xx.headlineReason : '',
    strip23xxCandidateCount: stripReader23xx && Array.isArray(stripReader23xx.candidates)
      ? stripReader23xx.candidates.length
      : 0,
    strip23xxConfidenceBestValue: stripReader23xx && stripReader23xx.confidenceBest && stripReader23xx.confidenceBest.predictedValue
      ? stripReader23xx.confidenceBest.predictedValue
      : '',
    strip23xxSelectedSourceValue: stripReader23xx && stripReader23xx.selectedSourceCandidate && stripReader23xx.selectedSourceCandidate.predictedValue
      ? stripReader23xx.selectedSourceCandidate.predictedValue
      : '',
    error: runPayload.error || '',
    canonicalPath,
    stageFiles
  };
};

const relative = (value) => value ? path.relative(ROOT_DIR, value).replace(/\\/g, '/') : '';

const htmlEscape = (value) => String(value ?? '')
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;');

const buildReportHtml = (rows, outputDir) => {
  const imageCell = (label, imagePath) => {
    if (!imagePath) {
      return `<div class="image-cell missing"><p>${htmlEscape(label)}</p><span>missing</span></div>`;
    }
    const relPath = path.relative(outputDir, imagePath).replace(/\\/g, '/');
    return `<div class="image-cell"><p>${htmlEscape(label)}</p><img src="${htmlEscape(relPath)}" /></div>`;
  };

  const rowHtml = rows.map((row) => {
    const confidence = Number.isFinite(row.stripConfidence) ? `${row.stripConfidence.toFixed(1)}%` : 'n/a';
    const strip23xxConfidence = Number.isFinite(row.strip23xxConfidence)
      ? `${row.strip23xxConfidence.toFixed(1)}%`
      : 'n/a';
    const strip23xxGuardConfidence = Number.isFinite(row.strip23xxGuardConfidence)
      ? `${row.strip23xxGuardConfidence.toFixed(1)}%`
      : 'n/a';
    const exactClass = row.detected === row.expected ? 'ok' : 'bad';
    const stripClass = row.stripValue === row.expected ? 'ok' : 'bad';
    const strip23xxClass = row.strip23xxAccepted && row.strip23xxValue === row.expected ? 'ok' : 'bad';
    return `
      <section class="qa-row">
        <header>
          <h2>${htmlEscape(row.filename)}</h2>
          <div class="meta">
            <span>Expected <strong>${htmlEscape(row.expected)}</strong></span>
            <span class="${exactClass}">Primary <strong>${htmlEscape(row.detected || 'no-read')}</strong></span>
            <span class="${stripClass}">Strip <strong>${htmlEscape(row.stripValue || 'n/a')}</strong> (${htmlEscape(confidence)})</span>
            <span class="${strip23xxClass}">23xx <strong>${htmlEscape(row.strip23xxValue || 'abstain')}</strong> (${htmlEscape(strip23xxConfidence)})</span>
            <span>23xx predicted ${htmlEscape(row.strip23xxPredictedValue || 'n/a')}</span>
            <span>23xx guard ${htmlEscape(strip23xxGuardConfidence)}</span>
            <span>Primary source ${htmlEscape(row.primarySource || 'n/a')}</span>
            <span>Strip source ${htmlEscape(row.stripSource || 'n/a')}</span>
            <span>Strip rule ${htmlEscape(row.stripHeadlineReason || 'n/a')}</span>
            <span>Strip probes ${htmlEscape(row.stripCandidateCount)}</span>
            <span>Best-by-confidence ${htmlEscape(row.stripConfidenceBestValue || 'n/a')}</span>
            <span>Selected-source strip ${htmlEscape(row.stripSelectedSourceValue || 'n/a')}</span>
            <span>23xx source ${htmlEscape(row.strip23xxSource || 'n/a')}</span>
            <span>23xx rule ${htmlEscape(row.strip23xxHeadlineReason || 'n/a')}</span>
            <span>23xx probes ${htmlEscape(row.strip23xxCandidateCount)}</span>
            <span>23xx best-by-confidence ${htmlEscape(row.strip23xxConfidenceBestValue || 'n/a')}</span>
            <span>23xx selected-source ${htmlEscape(row.strip23xxSelectedSourceValue || 'n/a')}</span>
          </div>
          ${row.error ? `<p class="error">${htmlEscape(row.error)}</p>` : ''}
        </header>
        <div class="images">
          ${imageCell('Canonical training window', row.canonicalPath)}
          ${imageCell('Stage 5 strip crop', row.stageFiles['stage5-strip'])}
          ${imageCell('Stage 6 selected OCR input', row.stageFiles['stage6-ocr'])}
          ${imageCell('Stage 7 classifier cells', row.stageFiles['stage7-cells'])}
          ${imageCell('Stage 8 strip-reader input', row.stageFiles['stage8-strip-reader'])}
        </div>
      </section>
    `;
  }).join('\n');

  return `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Jarvis Strip Runtime QA</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #172033;
      --muted: #667085;
      --line: #d7dee8;
      --paper: #f7f2e8;
      --card: #ffffff;
      --good: #0f7a4f;
      --bad: #b42318;
    }
    body {
      margin: 0;
      padding: 28px;
      background: radial-gradient(circle at 10% 0%, #ffffff 0, #f8ead1 32%, var(--paper) 68%);
      color: var(--ink);
      font-family: Georgia, 'Times New Roman', serif;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 30px;
    }
    .subtitle {
      margin: 0 0 22px;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
    }
    .qa-row {
      break-inside: avoid;
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid var(--line);
      border-radius: 18px;
      margin: 0 0 22px;
      padding: 16px;
      box-shadow: 0 12px 34px rgba(23, 32, 51, 0.08);
    }
    h2 {
      margin: 0 0 8px;
      font-size: 19px;
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
    }
    .meta span {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 8px;
      background: #fff;
    }
    .meta .ok {
      color: var(--good);
      border-color: rgba(15, 122, 79, 0.35);
    }
    .meta .bad {
      color: var(--bad);
      border-color: rgba(180, 35, 24, 0.35);
    }
    .error {
      color: var(--bad);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
    }
    .images {
      display: grid;
      grid-template-columns: repeat(5, minmax(170px, 1fr));
      gap: 12px;
      margin-top: 14px;
      align-items: start;
    }
    .image-cell {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fbfcfe;
      padding: 8px;
      min-height: 100px;
    }
    .image-cell p {
      margin: 0 0 8px;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px;
    }
    .image-cell img {
      display: block;
      max-width: 100%;
      height: auto;
      image-rendering: auto;
      border-radius: 8px;
      background: #101828;
    }
    .missing {
      display: grid;
      place-items: center;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
  </style>
</head>
<body>
  <h1>Jarvis Strip Runtime QA</h1>
  <p class="subtitle">Compare canonical training windows against browser runtime stages 5, 6, 7, and 8. Generated ${htmlEscape(new Date().toISOString())}.</p>
  ${rowHtml}
</body>
</html>`;
};

const writeReport = async (rows, outputDir) => {
  const reportPath = path.join(outputDir, 'strip-runtime-qa.html');
  const jsonPath = path.join(outputDir, 'summary.json');
  const screenshotPath = path.join(outputDir, 'strip-runtime-contact-sheet.png');
  await fsp.writeFile(reportPath, buildReportHtml(rows, outputDir), 'utf8');
  await fsp.writeFile(jsonPath, JSON.stringify({
    generatedAt: new Date().toISOString(),
    rows: rows.map((row) => ({
      ...row,
      canonicalPath: relative(row.canonicalPath),
      stageFiles: Object.fromEntries(Object.entries(row.stageFiles).map(([key, value]) => [key, relative(value)]))
    }))
  }, null, 2), 'utf8');

  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 2200 }, deviceScaleFactor: 1 });
    await page.goto(`file://${reportPath}`, { waitUntil: 'networkidle' });
    await page.screenshot({ path: screenshotPath, fullPage: true });
  } finally {
    await browser.close();
  }
  return { reportPath, jsonPath, screenshotPath };
};

const main = async () => {
  const targets = process.argv.slice(2).length ? process.argv.slice(2) : DEFAULT_TARGETS;
  const outputDir = path.join(OUTPUT_ROOT, timestampId());
  const imageDir = path.join(outputDir, 'images');
  await fsp.mkdir(imageDir, { recursive: true });

  const readings = await readMeters();
  const canonicalRows = await readCanonicalManifest();
  const processes = await ensureServices();
  const browser = await chromium.launch({ headless: true });
  const rows = [];
  try {
    const page = await browser.newPage();
    await page.goto(FRONTEND_URL, { waitUntil: 'networkidle' });
    await page.waitForSelector('#debug-overlay-toggle', { timeout: 30000 });
    for (const filename of targets) {
      console.log(`Inspecting ${filename}`);
      rows.push(await collectImageQa(page, filename, imageDir, readings, canonicalRows));
    }
  } finally {
    await browser.close();
    for (const processHandle of processes.reverse()) {
      await processHandle.stop();
    }
  }

  const report = await writeReport(rows, outputDir);
  console.log(JSON.stringify({
    outputDir: relative(outputDir),
    report: relative(report.reportPath),
    summary: relative(report.jsonPath),
    contactSheet: relative(report.screenshotPath),
    rows: rows.map((row) => ({
      filename: row.filename,
      expected: row.expected,
      primary: row.detected || 'no-read',
      strip: row.stripValue || 'n/a',
      stripConfidence: row.stripConfidence,
      stripSource: row.stripSource,
      stripHeadlineReason: row.stripHeadlineReason,
      stripCandidateCount: row.stripCandidateCount,
      stripConfidenceBestValue: row.stripConfidenceBestValue,
      stripSelectedSourceValue: row.stripSelectedSourceValue,
      strip23xxAccepted: row.strip23xxAccepted,
      strip23xxValue: row.strip23xxValue || 'n/a',
      strip23xxPredictedValue: row.strip23xxPredictedValue || 'n/a',
      strip23xxConfidence: row.strip23xxConfidence,
      strip23xxGuardConfidence: row.strip23xxGuardConfidence,
      strip23xxSource: row.strip23xxSource,
      strip23xxHeadlineReason: row.strip23xxHeadlineReason,
      strip23xxCandidateCount: row.strip23xxCandidateCount,
      strip23xxConfidenceBestValue: row.strip23xxConfidenceBestValue,
      strip23xxSelectedSourceValue: row.strip23xxSelectedSourceValue
    }))
  }, null, 2));
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
