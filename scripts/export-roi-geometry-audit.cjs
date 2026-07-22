#!/usr/bin/env node

const fsp = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require('@playwright/test');
const { ensureQaServices } = require('./lib/qa-services.cjs');

const ROOT_DIR = path.resolve(__dirname, '..');
const FRONTEND_URL = process.env.JARVIS_FRONTEND_URL || 'http://127.0.0.1:8000';
const BACKEND_URL = process.env.JARVIS_BACKEND_URL || 'http://127.0.0.1:8001';
const OUTPUT_ROOT = path.join(ROOT_DIR, 'output', 'roi-geometry-audit');
const MAX_PRIMARY_CANDIDATES = 20;
const MAX_DIAGNOSTIC_CANDIDATES = 36;

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

const htmlEscape = (value) => String(value ?? '')
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;');

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

const relativeFrom = (from, target) => path.relative(from, target).replaceAll(path.sep, '/');

const ensureServices = () => ensureQaServices({
  rootDir: ROOT_DIR,
  frontendUrl: FRONTEND_URL,
  backendUrl: BACKEND_URL
});

const runImage = async (page, row, options = {}) => page.evaluate(async ({ filename, options: browserOptions }) => {
  window.__jarvisOcrSelectionLogs = [];
  const { OCR_CONFIG } = await import('/src/ocr/config.js');
  OCR_CONFIG.digitClassifier = {
    ...OCR_CONFIG.digitClassifier,
    maxPrimaryCandidates: browserOptions.maxPrimaryCandidates,
    decodeDiagnosticCandidates: browserOptions.decodeDiagnosticCandidates === true,
    maxDiagnosticCandidates: browserOptions.maxDiagnosticCandidates,
    forceInitialPreviewCandidate: false
  };
  OCR_CONFIG.digitStripReader = { ...OCR_CONFIG.digitStripReader, enabled: false };
  OCR_CONFIG.digitStripReader23xx = { ...OCR_CONFIG.digitStripReader23xx, enabled: false };
  const { runMeterOcr } = await import('/src/ocr/pipeline.js');
  const response = await fetch(`/assets/${filename}`, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Unable to fetch asset ${filename}`);
  }
  const blob = await response.blob();
  const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });
  let error = '';
  try {
    await runMeterOcr(file);
  } catch (caught) {
    error = caught && caught.message ? caught.message : String(caught);
  }
  const logs = Array.isArray(window.__jarvisOcrSelectionLogs) ? window.__jarvisOcrSelectionLogs : [];
  return {
    error,
    selectionLog: logs.length ? logs[logs.length - 1] : null
  };
}, {
  filename: row.filename,
  options: {
    maxPrimaryCandidates: options.maxPrimaryCandidates || 4,
    decodeDiagnosticCandidates: options.decodeDiagnosticCandidates === true,
    maxDiagnosticCandidates: options.maxDiagnosticCandidates || MAX_DIAGNOSTIC_CANDIDATES
  }
});

const candidateValue = (entry) => entry && entry.result && entry.result.value ? entry.result.value : '';

const candidateVariantValues = (entry) => (
  entry
  && entry.result
  && Array.isArray(entry.result.variantCandidates)
    ? entry.result.variantCandidates.map((variant) => variant && variant.value).filter(Boolean)
    : []
);

const candidateMatchesExpected = (entry, expected) => (
  candidateValue(entry) === expected || candidateVariantValues(entry).includes(expected)
);

const sourceFamilyFor = (entry) => {
  const label = entry && entry.sourceLabel ? entry.sourceLabel : '';
  const probeKind = entry && entry.probeKind ? entry.probeKind : '';
  if (probeKind === 'register-localization' || label.includes('-regloc-')) {
    return 'regloc';
  }
  if (probeKind === 'normalization' || label.includes('-normprobe-')) {
    return 'normprobe';
  }
  if (label.includes('-edge-context-')) {
    return 'edge-context';
  }
  if (label.includes('-edge')) {
    return 'edge';
  }
  if (label.includes('-base')) {
    return 'base';
  }
  return 'other';
};

const extractAngle = (sourceLabel, geometry = null) => {
  if (geometry && Number.isFinite(geometry.angle)) {
    return geometry.angle;
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

const incrementCount = (counts, key) => {
  const safeKey = key || 'unknown';
  counts[safeKey] = (counts[safeKey] || 0) + 1;
  return counts;
};

const formatCountMap = (counts) => {
  const entries = Object.entries(counts || {});
  if (!entries.length) {
    return 'none';
  }
  return entries
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([key, value]) => `${key}:${value}`)
    .join(' ');
};

const rectText = (rect) => {
  if (!rect) {
    return '';
  }
  return `${rect.x},${rect.y} ${rect.width}x${rect.height}`;
};

const frameText = (frame) => {
  if (!frame) {
    return '';
  }
  return `L${frame.left} R${frame.right} T${frame.top} B${frame.bottom}`;
};

const clipFlags = (geometry) => {
  if (geometry && geometry.family === 'base') {
    return [];
  }
  const frame = geometry && geometry.cropFrame ? geometry.cropFrame : null;
  if (!frame) {
    return [];
  }
  const flags = [];
  if (frame.left <= 0.015) flags.push('left-edge');
  if (frame.right >= 0.985) flags.push('right-edge');
  if (frame.top <= 0.015) flags.push('top-edge');
  if (frame.bottom >= 0.985) flags.push('bottom-edge');
  return flags;
};

const summarizeCandidate = (candidate, expected) => {
  const result = candidate && candidate.result ? candidate.result : null;
  const geometry = candidate && candidate.geometry ? candidate.geometry : null;
  return {
    stage: candidate && candidate.stage ? candidate.stage : '',
    sourceLabel: candidate && candidate.sourceLabel ? candidate.sourceLabel : '',
    sourceFamily: sourceFamilyFor(candidate),
    angle: extractAngle(candidate && candidate.sourceLabel, geometry),
    diagnosticOnly: candidate && candidate.diagnosticOnly === true,
    probeKind: candidate && candidate.probeKind ? candidate.probeKind : null,
    width: Number.isFinite(candidate && candidate.width) ? candidate.width : null,
    height: Number.isFinite(candidate && candidate.height) ? candidate.height : null,
    value: result && result.value ? result.value : '',
    score: Number.isFinite(result && result.score) ? result.score : null,
    confidence: Number.isFinite(result && result.confidence) ? result.confidence : null,
    cropMode: result && result.cropMode ? result.cropMode : '',
    cropRatio: Number.isFinite(result && result.cropRatio) ? result.cropRatio : null,
    cellDigits: Array.isArray(result && result.cellDigits) ? result.cellDigits : [],
    splitGeometry: result && result.splitGeometry ? result.splitGeometry : null,
    expectedHit: candidateMatchesExpected(candidate, expected),
    geometry,
    clipFlags: clipFlags(geometry)
  };
};

const groupByAngle = (candidates) => {
  const groups = new Map();
  candidates.forEach((candidate) => {
    const key = Number.isFinite(candidate.angle) ? String(candidate.angle) : 'unknown';
    const existing = groups.get(key) || {
      angle: key,
      candidateCount: 0,
      readableCount: 0,
      familyCounts: {},
      edgeRects: [],
      rotatedSize: null,
      clippedCandidateCount: 0
    };
    existing.candidateCount += 1;
    if (candidate.value) {
      existing.readableCount += 1;
    }
    incrementCount(existing.familyCounts, candidate.sourceFamily);
    if (candidate.geometry && candidate.geometry.edgeRect) {
      const token = rectText(candidate.geometry.edgeRect);
      if (token && !existing.edgeRects.includes(token)) {
        existing.edgeRects.push(token);
      }
    }
    if (!existing.rotatedSize && candidate.geometry && candidate.geometry.rotatedSize) {
      existing.rotatedSize = candidate.geometry.rotatedSize;
    }
    if (candidate.clipFlags.length) {
      existing.clippedCandidateCount += 1;
    }
    groups.set(key, existing);
  });
  return [...groups.values()].sort((a, b) => String(a.angle).localeCompare(String(b.angle)));
};

const inferGeometryBucket = (row) => {
  if (!row.roiGeometry) {
    return 'missing-roi-geometry';
  }
  const hasEdgeRect = row.candidates.some((candidate) => candidate.geometry && candidate.geometry.edgeRect);
  if (!hasEdgeRect) {
    return 'edge-window-missing';
  }
  const clipped = row.candidates.filter((candidate) => candidate.clipFlags.length).length;
  if (clipped >= Math.ceil(row.candidates.length * 0.4)) {
    return 'crop-family-boundary-clipped';
  }
  const normprobeCount = row.sourceFamilyCounts.normprobe || 0;
  if (normprobeCount >= 8) {
    return 'edge-window-present-normalization-insufficient';
  }
  return 'edge-window-present-expected-still-absent';
};

const buildReportHtml = (rows, outputDir) => `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ROI Geometry Audit</title>
  <style>
    body { margin: 24px; font-family: Inter, system-ui, sans-serif; color: #172033; background: #f8fafc; }
    h1 { margin: 0 0 8px; font-size: 28px; }
    h2 { margin: 28px 0 8px; font-size: 20px; }
    .summary, .row { background: #fff; border: 1px solid #d8e0ea; border-radius: 8px; padding: 16px; margin: 16px 0; }
    .meta { display: flex; flex-wrap: wrap; gap: 8px 16px; color: #42526b; font-size: 13px; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #e8eef7; color: #253858; font-size: 12px; }
    .grid { display: grid; grid-template-columns: 220px 1fr; gap: 16px; align-items: start; }
    img { max-width: 220px; border-radius: 6px; border: 1px solid #d8e0ea; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 12px; }
    th, td { border-bottom: 1px solid #e6ebf2; padding: 6px 8px; text-align: left; vertical-align: top; }
    th { color: #42526b; font-weight: 700; background: #f1f5f9; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
    details { margin-top: 10px; }
    summary { cursor: pointer; font-weight: 700; color: #1f4f82; }
  </style>
</head>
<body>
  <h1>ROI Geometry Audit</h1>
  <div class="summary">
    <div class="meta">
      <span>Focused rows: <strong>${rows.length}</strong></span>
      <span>Bucket counts: <strong>${htmlEscape(formatCountMap(rows.reduce((counts, row) => incrementCount(counts, row.geometryBucket), {})))}</strong></span>
    </div>
  </div>
  ${rows.map((row) => `
    <section class="row">
      <h2>${htmlEscape(row.filename)} <span class="pill">${htmlEscape(row.geometryBucket)}</span></h2>
      <div class="grid">
        <img src="${htmlEscape(relativeFrom(outputDir, path.join(ROOT_DIR, 'assets', row.filename)))}" alt="${htmlEscape(row.filename)}">
        <div>
          <div class="meta">
            <span>Expected <strong>${htmlEscape(row.expected)}</strong></span>
            <span>Selected <strong>${htmlEscape(row.productionSelected || 'no-read')}</strong></span>
            <span>Selected source <code>${htmlEscape(row.productionSelectedSource || '')}</code></span>
            <span>Families <strong>${htmlEscape(formatCountMap(row.sourceFamilyCounts))}</strong></span>
          </div>
          <table>
            <thead><tr><th>ROI Field</th><th>Value</th></tr></thead>
            <tbody>
              <tr><td>base size</td><td>${htmlEscape(row.roiGeometry && row.roiGeometry.baseSize ? `${row.roiGeometry.baseSize.width}x${row.roiGeometry.baseSize.height}` : '')}</td></tr>
              <tr><td>detection confidence</td><td>${htmlEscape(row.roiGeometry && row.roiGeometry.detection ? row.roiGeometry.detection.confidence : '')}</td></tr>
              <tr><td>detected rect</td><td><code>${htmlEscape(row.roiGeometry && row.roiGeometry.detection ? rectText(row.roiGeometry.detection.rect) : '')}</code></td></tr>
              <tr><td>expanded rect</td><td><code>${htmlEscape(row.roiGeometry ? rectText(row.roiGeometry.expandedRect) : '')}</code></td></tr>
            </tbody>
          </table>
          <table>
            <thead><tr><th>Angle</th><th>Rotated Size</th><th>Readable</th><th>Families</th><th>Edge Rects</th><th>Clipped</th></tr></thead>
            <tbody>
              ${row.angleSummaries.map((summary) => `
                <tr>
                  <td>${htmlEscape(summary.angle)}</td>
                  <td>${htmlEscape(summary.rotatedSize ? `${summary.rotatedSize.width}x${summary.rotatedSize.height}` : '')}</td>
                  <td>${summary.readableCount}/${summary.candidateCount}</td>
                  <td>${htmlEscape(formatCountMap(summary.familyCounts))}</td>
                  <td><code>${htmlEscape(summary.edgeRects.join(' | '))}</code></td>
                  <td>${summary.clippedCandidateCount}</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
          <details>
            <summary>Candidate Geometry</summary>
            <table>
              <thead><tr><th>Source</th><th>Family</th><th>Value</th><th>Score</th><th>Crop Rect</th><th>Frame</th><th>Flags</th></tr></thead>
              <tbody>
                ${row.candidates.map((candidate) => `
                  <tr>
                    <td><code>${htmlEscape(candidate.sourceLabel)}</code></td>
                    <td>${htmlEscape(candidate.sourceFamily)}${candidate.diagnosticOnly ? ' diag' : ''}</td>
                    <td>${htmlEscape(candidate.value)}</td>
                    <td>${htmlEscape(candidate.score ?? '')}</td>
                    <td><code>${htmlEscape(candidate.geometry ? rectText(candidate.geometry.cropRect) : '')}</code></td>
                    <td><code>${htmlEscape(candidate.geometry ? frameText(candidate.geometry.cropFrame) : '')}</code></td>
                    <td>${htmlEscape(candidate.clipFlags.join(', '))}</td>
                  </tr>
                `).join('')}
              </tbody>
            </table>
          </details>
        </div>
      </div>
    </section>
  `).join('')}
</body>
</html>`;

const main = async () => {
  const outputDir = path.join(OUTPUT_ROOT, timestampId());
  await fsp.mkdir(outputDir, { recursive: true });
  const rows = parseCsv(await fsp.readFile(path.join(ROOT_DIR, 'assets', 'meter_readings.csv'), 'utf8'));
  const processes = await ensureServices();
  const browser = await chromium.launch({ headless: true });
  const reportRows = [];
  try {
    const page = await browser.newPage();
    await page.goto(FRONTEND_URL, { waitUntil: 'networkidle' });
    for (const row of rows) {
      process.stdout.write(`ROI geometry audit inspecting ${row.filename}\n`);
      const productionRun = await runImage(page, row, { maxPrimaryCandidates: 4 });
      const oracleRun = await runImage(page, row, {
        maxPrimaryCandidates: MAX_PRIMARY_CANDIDATES,
        decodeDiagnosticCandidates: true,
        maxDiagnosticCandidates: MAX_DIAGNOSTIC_CANDIDATES
      });
      if (productionRun.error || oracleRun.error || !oracleRun.selectionLog) {
        continue;
      }
      const productionSelected = productionRun.selectionLog && productionRun.selectionLog.selected
        ? productionRun.selectionLog.selected
        : null;
      const productionSelectedValue = productionSelected && productionSelected.value ? productionSelected.value : '';
      if (productionSelectedValue === row.value) {
        continue;
      }
      const rawCandidates = Array.isArray(oracleRun.selectionLog.candidateTrace)
        ? oracleRun.selectionLog.candidateTrace
        : [];
      const readable = rawCandidates.filter((candidate) => candidateValue(candidate));
      if (readable.some((candidate) => candidateMatchesExpected(candidate, row.value))) {
        continue;
      }
      const candidates = rawCandidates.map((candidate) => summarizeCandidate(candidate, row.value));
      const sourceFamilyCounts = candidates.reduce((counts, candidate) => (
        incrementCount(counts, candidate.sourceFamily)
      ), {});
      const reportRow = {
        filename: row.filename,
        expected: row.value,
        productionSelected: productionSelectedValue,
        productionSelectedSource: productionSelected && productionSelected.sourceLabel ? productionSelected.sourceLabel : '',
        roiGeometry: oracleRun.selectionLog.roiGeometry || null,
        sourceFamilyCounts,
        candidates,
        angleSummaries: groupByAngle(candidates)
      };
      reportRow.geometryBucket = inferGeometryBucket(reportRow);
      reportRows.push(reportRow);
    }
  } finally {
    await browser.close();
    for (const processHandle of processes.reverse()) {
      await processHandle.stop();
    }
  }

  const reportPath = path.join(outputDir, 'roi-geometry-audit.html');
  const summaryPath = path.join(outputDir, 'summary.json');
  const geometryBucketCounts = reportRows.reduce((counts, row) => (
    incrementCount(counts, row.geometryBucket)
  ), {});
  await fsp.writeFile(reportPath, buildReportHtml(reportRows, outputDir), 'utf8');
  await fsp.writeFile(summaryPath, JSON.stringify({
    generatedAt: new Date().toISOString(),
    maxPrimaryCandidates: MAX_PRIMARY_CANDIDATES,
    maxDiagnosticCandidates: MAX_DIAGNOSTIC_CANDIDATES,
    rowCount: reportRows.length,
    geometryBucketCounts,
    rows: reportRows
  }, null, 2), 'utf8');

  console.log(JSON.stringify({
    outputDir: relativeFrom(ROOT_DIR, outputDir),
    report: relativeFrom(ROOT_DIR, reportPath),
    summary: relativeFrom(ROOT_DIR, summaryPath),
    rowCount: reportRows.length,
    geometryBucketCounts,
    rows: reportRows.map((row) => ({
      filename: row.filename,
      expected: row.expected,
      selected: row.productionSelected || 'no-read',
      selectedSource: row.productionSelectedSource,
      geometryBucket: row.geometryBucket,
      roiExpandedRect: row.roiGeometry ? row.roiGeometry.expandedRect : null,
      sourceFamilyCounts: row.sourceFamilyCounts,
      angleSummaries: row.angleSummaries
    }))
  }, null, 2));
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
