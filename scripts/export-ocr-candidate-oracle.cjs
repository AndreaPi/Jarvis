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
const OUTPUT_ROOT = path.join(ROOT_DIR, 'output', 'ocr-candidate-oracle');
const MAX_PRIMARY_CANDIDATES = Number.isFinite(Number.parseInt(process.env.JARVIS_ORACLE_MAX_CANDIDATES || '', 10))
  ? Number.parseInt(process.env.JARVIS_ORACLE_MAX_CANDIDATES, 10)
  : 20;

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

const parseCsv = (text) => {
  const lines = String(text || '').split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) {
    return [];
  }
  const headers = lines[0].split(',').map((header) => header.trim());
  return lines.slice(1).map((line) => {
    const values = line.split(',');
    const row = {};
    headers.forEach((header, index) => {
      row[header] = (values[index] || '').trim();
    });
    return row;
  }).filter((row) => row.filename && row.value);
};

const parseValue = (value) => {
  const digits = String(value || '').replace(/\D/g, '');
  if (!digits) {
    return null;
  }
  const parsed = Number.parseInt(digits, 10);
  return Number.isFinite(parsed) ? parsed : null;
};

const absoluteError = (expected, detected) => {
  const expectedValue = parseValue(expected);
  const detectedValue = parseValue(detected);
  if (!Number.isFinite(expectedValue) || !Number.isFinite(detectedValue)) {
    return null;
  }
  return Math.abs(expectedValue - detectedValue);
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
    processes.push(spawnTrackedProcess('npm', ['run', 'serve'], { label: 'frontend' }));
  }
  if (!(await requestOk(`${BACKEND_URL}/health`))) {
    processes.push(spawnTrackedProcess(
      path.join(ROOT_DIR, 'backend', '.venv', 'bin', 'uvicorn'),
      ['backend.app:app', '--host', '127.0.0.1', '--port', '8001'],
      { label: 'backend' }
    ));
  }

  for (let attempt = 0; attempt < 100; attempt += 1) {
    if ((await requestOk(FRONTEND_URL)) && (await requestOk(`${BACKEND_URL}/health`))) {
      return processes;
    }
    await sleep(250);
  }
  throw new Error('Timed out waiting for frontend/backend services.');
};

const sourceKind = (sourceLabel) => {
  const label = String(sourceLabel || '');
  if (!label) {
    return 'unknown';
  }
  if (label.includes('-edge')) {
    return 'edge';
  }
  if (label.includes('-base') || label === 'scan-roi' || label === 'raw-fallback-roi') {
    return 'base';
  }
  return 'other';
};

const extractAngle = (sourceLabel, fallbackAngle = null) => {
  if (Number.isFinite(fallbackAngle)) {
    return fallbackAngle;
  }
  const tokens = String(sourceLabel || '').split('-');
  for (const token of tokens) {
    const parsed = Number.parseInt(token, 10);
    if (Number.isFinite(parsed) && parsed % 90 === 0) {
      return ((parsed % 360) + 360) % 360;
    }
  }
  return null;
};

const summarizeCandidate = (entry) => {
  const result = entry && entry.result ? entry.result : null;
  return {
    stage: entry && entry.stage ? entry.stage : '',
    sourceLabel: entry && entry.sourceLabel ? entry.sourceLabel : '',
    sourceKind: sourceKind(entry && entry.sourceLabel),
    angle: extractAngle(entry && entry.sourceLabel, result && result.angle),
    width: Number.isFinite(entry && entry.width) ? entry.width : null,
    height: Number.isFinite(entry && entry.height) ? entry.height : null,
    fallbackScore: Number.isFinite(entry && entry.fallbackScore) ? entry.fallbackScore : null,
    fallbackAspect: Number.isFinite(entry && entry.fallbackAspect) ? entry.fallbackAspect : null,
    value: result && result.value ? result.value : '',
    confidence: Number.isFinite(result && result.confidence) ? result.confidence : null,
    score: Number.isFinite(result && result.score) ? result.score : null,
    cellDigits: Array.isArray(result && result.cellDigits) ? result.cellDigits : [],
    cellConfidences: Array.isArray(result && result.cellConfidences) ? result.cellConfidences : [],
    rejectReasons: Array.isArray(entry && entry.rejects)
      ? entry.rejects.map((reject) => reject && reject.reason ? String(reject.reason) : 'unknown')
      : []
  };
};

const compareByScoreThenConfidence = (a, b) => (
  (b.score ?? -Infinity) - (a.score ?? -Infinity)
  || (b.confidence ?? -Infinity) - (a.confidence ?? -Infinity)
  || String(a.sourceLabel).localeCompare(String(b.sourceLabel))
);

const compareByConfidenceThenScore = (a, b) => (
  (b.confidence ?? -Infinity) - (a.confidence ?? -Infinity)
  || (b.score ?? -Infinity) - (a.score ?? -Infinity)
  || String(a.sourceLabel).localeCompare(String(b.sourceLabel))
);

const bestBy = (items, comparator) => {
  if (!Array.isArray(items) || !items.length) {
    return null;
  }
  return [...items].sort(comparator)[0] || null;
};

const summarizeBySource = (candidates, expected) => {
  const groups = new Map();
  candidates.forEach((candidate) => {
    const key = `${candidate.sourceKind}:${candidate.angle ?? 'n/a'}`;
    const existing = groups.get(key) || [];
    existing.push(candidate);
    groups.set(key, existing);
  });
  return [...groups.entries()].map(([key, group]) => {
    const [kind, angle] = key.split(':');
    const bestScore = bestBy(group, compareByScoreThenConfidence);
    const bestConfidence = bestBy(group, compareByConfidenceThenScore);
    const expectedMatches = group.filter((candidate) => candidate.value === expected);
    return {
      kind,
      angle,
      count: group.length,
      bestScore: bestScore ? {
        value: bestScore.value,
        sourceLabel: bestScore.sourceLabel,
        score: bestScore.score,
        confidence: bestScore.confidence
      } : null,
      bestConfidence: bestConfidence ? {
        value: bestConfidence.value,
        sourceLabel: bestConfidence.sourceLabel,
        score: bestConfidence.score,
        confidence: bestConfidence.confidence
      } : null,
      expectedMatchCount: expectedMatches.length,
      expectedSources: expectedMatches.map((candidate) => candidate.sourceLabel)
    };
  }).sort((a, b) => (
    String(a.kind).localeCompare(String(b.kind))
    || String(a.angle).localeCompare(String(b.angle))
  ));
};

const analyzeSelectionLog = (row, productionLog, oracleLog, productionError = '', oracleError = '') => {
  const selected = productionLog && productionLog.selected ? productionLog.selected : null;
  const selectedValue = selected && selected.value ? selected.value : '';
  const candidates = oracleLog && Array.isArray(oracleLog.candidateTrace)
    ? oracleLog.candidateTrace.map(summarizeCandidate)
    : [];
  const readableCandidates = candidates.filter((candidate) => candidate.value);
  const expectedCandidates = readableCandidates.filter((candidate) => candidate.value === row.value);
  const selectedSourceCandidates = selected && selected.sourceLabel
    ? readableCandidates.filter((candidate) => candidate.sourceLabel === selected.sourceLabel)
    : [];
  const scoreBest = bestBy(readableCandidates, compareByScoreThenConfidence);
  const confidenceBest = bestBy(readableCandidates, compareByConfidenceThenScore);
  const expectedBest = bestBy(expectedCandidates, compareByScoreThenConfidence);
  const selectedSourceBest = bestBy(selectedSourceCandidates, compareByScoreThenConfidence);
  const selectedExact = selectedValue === row.value;
  const oracleAvailable = expectedCandidates.length > 0;
  const failureClass = selectedExact
    ? 'selected-correct'
    : (
      productionError
        ? 'ocr-error'
        : (oracleAvailable ? 'misranked-candidate' : 'no-correct-candidate')
    );

  return {
    filename: row.filename,
    expected: row.value,
    selected: selectedValue,
    selectedExact,
    selectedSourceLabel: selected && selected.sourceLabel ? selected.sourceLabel : '',
    selectedMethod: selected && selected.method ? selected.method : '',
    selectedScore: Number.isFinite(selected && selected.score) ? selected.score : null,
    selectedConfidence: Number.isFinite(selected && selected.confidence) ? selected.confidence : null,
    absoluteError: absoluteError(row.value, selectedValue),
    failureClass,
    oracleAvailable,
    candidateCount: candidates.length,
    readableCandidateCount: readableCandidates.length,
    expectedCandidateCount: expectedCandidates.length,
    scoreBest,
    confidenceBest,
    selectedSourceBest,
    expectedBest,
    bySource: summarizeBySource(readableCandidates, row.value),
    candidates,
    productionError,
    oracleError
  };
};

const readRows = async () => {
  const csvPath = path.join(ROOT_DIR, 'assets', 'meter_readings.csv');
  return parseCsv(await fsp.readFile(csvPath, 'utf8'));
};

const runImage = async (page, row, options = {}) => {
  return page.evaluate(async ({ filename, options: browserOptions }) => {
    const maxPrimaryCandidates = Number.isFinite(browserOptions && browserOptions.maxPrimaryCandidates)
      ? browserOptions.maxPrimaryCandidates
      : 4;
    window.__jarvisOcrSelectionLogs = [];
    const { OCR_CONFIG } = await import('/src/ocr/config.js');
    OCR_CONFIG.digitClassifier = {
      ...OCR_CONFIG.digitClassifier,
      maxPrimaryCandidates,
      forceInitialPreviewCandidate: false
    };
    OCR_CONFIG.digitStripReader = {
      ...OCR_CONFIG.digitStripReader,
      enabled: false
    };
    OCR_CONFIG.digitStripReader23xx = {
      ...OCR_CONFIG.digitStripReader23xx,
      enabled: false
    };
    const { runMeterOcr } = await import('/src/ocr/pipeline.js');
    const response = await fetch(`/assets/${filename}`, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`Unable to fetch asset ${filename}`);
    }
    const blob = await response.blob();
    const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });
    let result = null;
    let error = '';
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
  }, {
    filename: row.filename,
    options: {
      maxPrimaryCandidates: Number.isFinite(options.maxPrimaryCandidates)
        ? options.maxPrimaryCandidates
        : undefined
    }
  });
};

const markdownEscape = (value) => String(value ?? '')
  .replace(/\|/g, '\\|')
  .replace(/\n/g, ' ');

const candidateLabel = (candidate) => {
  if (!candidate) {
    return 'n/a';
  }
  const bits = [
    candidate.value || 'no-read',
    candidate.sourceLabel || 'unknown'
  ];
  if (Number.isFinite(candidate.score)) {
    bits.push(`score ${candidate.score.toFixed(3)}`);
  }
  if (Number.isFinite(candidate.confidence)) {
    bits.push(`conf ${candidate.confidence.toFixed(1)}`);
  }
  return bits.join(' / ');
};

const summarizeRows = (rows) => {
  const total = rows.length;
  const selectedCorrect = rows.filter((row) => row.selectedExact).length;
  const oracleAvailable = rows.filter((row) => row.oracleAvailable).length;
  const misranked = rows.filter((row) => row.failureClass === 'misranked-candidate').length;
  const noCorrectCandidate = rows.filter((row) => row.failureClass === 'no-correct-candidate').length;
  const ocrErrors = rows.filter((row) => row.failureClass === 'ocr-error').length;
  const noRead = rows.filter((row) => !row.selected).length;
  const errors = rows.map((row) => row.absoluteError).filter((value) => Number.isFinite(value));
  const mae = errors.length ? errors.reduce((sum, value) => sum + value, 0) / errors.length : null;
  const oracleErrors = rows
    .map((row) => (row.oracleAvailable ? 0 : row.absoluteError))
    .filter((value) => Number.isFinite(value));
  const oracleMae = oracleErrors.length
    ? oracleErrors.reduce((sum, value) => sum + value, 0) / oracleErrors.length
    : null;
  return {
    total,
    selectedCorrect,
    oracleAvailable,
    misranked,
    noCorrectCandidate,
    ocrErrors,
    noRead,
    mae,
    oracleMae
  };
};

const buildMarkdown = (summary, rows, outputDir) => {
  const lines = [];
  lines.push('# OCR Candidate Oracle Benchmark');
  lines.push('');
  lines.push(`Generated: ${new Date().toISOString()}`);
  lines.push(`Oracle max primary candidates: ${MAX_PRIMARY_CANDIDATES}`);
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push(`- Selected exact: ${summary.selectedCorrect}/${summary.total}`);
  lines.push(`- Oracle candidate available: ${summary.oracleAvailable}/${summary.total}`);
  lines.push(`- Misranked candidate cases: ${summary.misranked}`);
  lines.push(`- No correct candidate cases: ${summary.noCorrectCandidate}`);
  lines.push(`- OCR errors: ${summary.ocrErrors}`);
  lines.push(`- Normal selected MAE: ${Number.isFinite(summary.mae) ? summary.mae.toFixed(2) : 'n/a'}`);
  lines.push(`- Oracle-if-present MAE: ${Number.isFinite(summary.oracleMae) ? summary.oracleMae.toFixed(2) : 'n/a'}`);
  lines.push('');
  lines.push('## Rows');
  lines.push('');
  lines.push('| File | Expected | Selected | Class | Candidate match? | Selected source | Score-best | Confidence-best | Expected-best | Candidate count |');
  lines.push('|---|---:|---:|---|---|---|---|---|---|---:|');
  rows.forEach((row) => {
    lines.push([
      markdownEscape(row.filename),
      markdownEscape(row.expected),
      markdownEscape(row.selected || 'no-read'),
      markdownEscape(row.failureClass),
      row.oracleAvailable ? `yes (${row.expectedCandidateCount})` : 'no',
      markdownEscape(row.selectedSourceLabel || 'n/a'),
      markdownEscape(candidateLabel(row.scoreBest)),
      markdownEscape(candidateLabel(row.confidenceBest)),
      markdownEscape(candidateLabel(row.expectedBest)),
      String(row.candidateCount)
    ].join(' | ').replace(/^/, '| ').replace(/$/, ' |'));
  });
  lines.push('');
  lines.push('## Misranked Cases');
  lines.push('');
  rows.filter((row) => row.failureClass === 'misranked-candidate').forEach((row) => {
    lines.push(`### ${row.filename}`);
    lines.push('');
    lines.push(`- Expected: \`${row.expected}\``);
    lines.push(`- Selected: \`${row.selected || 'no-read'}\` from \`${row.selectedSourceLabel || 'n/a'}\``);
    lines.push(`- Expected-best: ${candidateLabel(row.expectedBest)}`);
    lines.push(`- Score-best: ${candidateLabel(row.scoreBest)}`);
    lines.push(`- Confidence-best: ${candidateLabel(row.confidenceBest)}`);
    lines.push('- By source:');
    row.bySource.forEach((entry) => {
      lines.push(`  - ${entry.kind}/${entry.angle}: bestScore=${candidateLabel(entry.bestScore)}, bestConfidence=${candidateLabel(entry.bestConfidence)}, expectedMatches=${entry.expectedMatchCount}`);
    });
    lines.push('');
  });
  lines.push('## Artifacts');
  lines.push('');
  lines.push(`- JSON: \`${path.relative(ROOT_DIR, path.join(outputDir, 'oracle-candidate-summary.json')).replace(/\\/g, '/')}\``);
  return `${lines.join('\n')}\n`;
};

const main = async () => {
  const outputDir = path.join(OUTPUT_ROOT, timestampId());
  await fsp.mkdir(outputDir, { recursive: true });
  const rows = await readRows();
  const processes = await ensureServices();
  const browser = await chromium.launch({ headless: true });
  const analyzedRows = [];
  try {
    const page = await browser.newPage();
    await page.goto(FRONTEND_URL, { waitUntil: 'networkidle' });
    for (const row of rows) {
      process.stdout.write(`Oracle inspecting ${row.filename}\n`);
      const productionRun = await runImage(page, row, { maxPrimaryCandidates: 4 });
      const oracleRun = await runImage(page, row, { maxPrimaryCandidates: MAX_PRIMARY_CANDIDATES });
      analyzedRows.push(analyzeSelectionLog(
        row,
        productionRun.selectionLog,
        oracleRun.selectionLog,
        productionRun.error,
        oracleRun.error
      ));
    }
  } finally {
    await browser.close();
    for (const processHandle of processes.reverse()) {
      await processHandle.stop();
    }
  }

  const summary = summarizeRows(analyzedRows);
  const jsonPath = path.join(outputDir, 'oracle-candidate-summary.json');
  const reportPath = path.join(outputDir, 'oracle-candidate-report.md');
  await fsp.writeFile(jsonPath, JSON.stringify({
    generatedAt: new Date().toISOString(),
    maxPrimaryCandidates: MAX_PRIMARY_CANDIDATES,
    productionMaxPrimaryCandidates: 4,
    summary,
    rows: analyzedRows
  }, null, 2), 'utf8');
  await fsp.writeFile(reportPath, buildMarkdown(summary, analyzedRows, outputDir), 'utf8');
  console.log(JSON.stringify({
    outputDir: path.relative(ROOT_DIR, outputDir).replace(/\\/g, '/'),
    report: path.relative(ROOT_DIR, reportPath).replace(/\\/g, '/'),
    summary: path.relative(ROOT_DIR, jsonPath).replace(/\\/g, '/'),
    metrics: summary
  }, null, 2));
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
