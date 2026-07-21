#!/usr/bin/env node

const fs = require('node:fs');
const fsp = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require('@playwright/test');
const { ensureQaServices } = require('./lib/qa-services.cjs');

const ROOT_DIR = path.resolve(__dirname, '..');
const FRONTEND_URL = process.env.JARVIS_FRONTEND_URL || 'http://127.0.0.1:8000';
const BACKEND_URL = process.env.JARVIS_BACKEND_URL || 'http://127.0.0.1:8001';
const OUTPUT_ROOT = path.join(ROOT_DIR, 'output', 'cell-crop-failure-qa');
const MAX_PRIMARY_CANDIDATES = 20;
const MAX_DIAGNOSTIC_CANDIDATES = 36;
const QA_FILES = new Set(String(process.env.CELL_CROP_QA_FILES || '')
  .split(',')
  .map((value) => value.trim())
  .filter(Boolean));

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

const parseDataUrl = (dataUrl) => {
  if (typeof dataUrl !== 'string') {
    return null;
  }
  const match = dataUrl.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)$/);
  if (!match) {
    return null;
  }
  return {
    extension: match[1] === 'image/png' ? 'png' : 'jpg',
    buffer: Buffer.from(match[2], 'base64')
  };
};

const writeDataUrl = async (dataUrl, outputPathBase) => {
  const parsed = parseDataUrl(dataUrl);
  if (!parsed) {
    return null;
  }
  const outputPath = `${outputPathBase}.${parsed.extension}`;
  await fsp.writeFile(outputPath, parsed.buffer);
  return outputPath;
};

const ensureServices = () => ensureQaServices({
  rootDir: ROOT_DIR,
  frontendUrl: FRONTEND_URL,
  backendUrl: BACKEND_URL
});

const runImage = async (page, row, options = {}) => {
  return page.evaluate(async ({ filename, options: browserOptions }) => {
    window.__jarvisOcrSelectionLogs = [];
    window.__JARVIS_EXPORT_CANDIDATE_IMAGES__ = browserOptions.exportCandidateImages === true;
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
    window.__JARVIS_EXPORT_CANDIDATE_IMAGES__ = false;
    return {
      error,
      selectionLog: logs.length ? logs[logs.length - 1] : null
    };
  }, {
    filename: row.filename,
    options: {
      maxPrimaryCandidates: options.maxPrimaryCandidates || 4,
      decodeDiagnosticCandidates: options.decodeDiagnosticCandidates === true,
      maxDiagnosticCandidates: options.maxDiagnosticCandidates || MAX_DIAGNOSTIC_CANDIDATES,
      exportCandidateImages: options.exportCandidateImages === true
    }
  });
};

const htmlEscape = (value) => String(value ?? '')
  .replace(/&/g, '&amp;')
  .replace(/</g, '&lt;')
  .replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;');

const candidateValue = (entry) => entry && entry.result && entry.result.value ? entry.result.value : '';

const candidateVariantValues = (entry) => (
  entry
  && entry.result
  && Array.isArray(entry.result.variantCandidates)
    ? entry.result.variantCandidates.map((variant) => variant && variant.value).filter(Boolean)
    : []
);

const candidateMatchKind = (entry, expected) => {
  if (candidateValue(entry) === expected) {
    return 'direct';
  }
  return candidateVariantValues(entry).includes(expected) ? 'variant' : '';
};

const candidateMatchesExpected = (entry, expected) => !!candidateMatchKind(entry, expected);

const isDiagnosticCandidate = (entry) => entry && entry.diagnosticOnly === true;

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

const incrementCount = (counts, key, amount = 1) => {
  const safeKey = key || 'unknown';
  counts[safeKey] = (counts[safeKey] || 0) + amount;
  return counts;
};

const countCandidateFamilies = (candidates) => candidates.reduce((counts, candidate) => (
  incrementCount(counts, candidate.sourceFamily || sourceFamilyFor(candidate))
), {});

const countExpectedHitFamilies = (candidates, expected) => candidates.reduce((counts, candidate) => {
  if (candidateMatchesExpected(candidate, expected)) {
    incrementCount(counts, candidate.sourceFamily || sourceFamilyFor(candidate));
  }
  return counts;
}, {});

const hasSplitProbeExpectedHit = (candidate, expected) => {
  if (!candidate || !candidate.result) {
    return false;
  }
  const result = candidate.result;
  if (
    result.value === expected
    && Number.isFinite(result.splitOffsetRatio)
    && Math.abs(result.splitOffsetRatio) > 0.0005
  ) {
    return true;
  }
  const variants = Array.isArray(result.variantCandidates) ? result.variantCandidates : [];
  return variants.some((variant) => (
    variant
    && variant.value === expected
    && Number.isFinite(variant.splitOffsetRatio)
    && Math.abs(variant.splitOffsetRatio) > 0.0005
  ));
};

const hasNonSplitExpectedHit = (candidate, expected) => {
  if (!candidate || !candidate.result) {
    return false;
  }
  const result = candidate.result;
  if (
    result.value === expected
    && (!Number.isFinite(result.splitOffsetRatio) || Math.abs(result.splitOffsetRatio) <= 0.0005)
  ) {
    return true;
  }
  const variants = Array.isArray(result.variantCandidates) ? result.variantCandidates : [];
  return variants.some((variant) => (
    variant
    && variant.value === expected
    && (!Number.isFinite(variant.splitOffsetRatio) || Math.abs(variant.splitOffsetRatio) <= 0.0005)
  ));
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

const summarizeComparisonEntry = (entry, expected, matchKind = '') => {
  if (!entry) {
    return null;
  }
  const result = entry.result || {};
  return {
    value: entry.value || result.value || '',
    sourceLabel: entry.sourceLabel || '',
    sourceFamily: entry.sourceFamily || sourceFamilyFor(entry),
    diagnosticOnly: isDiagnosticCandidate(entry),
    matchKind,
    isExpected: (entry.value || result.value || '') === expected,
    score: Number.isFinite(result.score) ? result.score : null,
    confidence: Number.isFinite(result.confidence) ? result.confidence : null,
    cropMode: result.cropMode || '',
    cropRatio: Number.isFinite(result.cropRatio) ? result.cropRatio : null,
    splitOffsetRatio: Number.isFinite(result.splitOffsetRatio) ? result.splitOffsetRatio : 0,
    splitMode: result.splitMode || '',
    geometryRankPenalty: Number.isFinite(result.geometryRankPenalty) ? result.geometryRankPenalty : null,
    selectionGeometryPenalty: Number.isFinite(result.selectionGeometryPenalty)
      ? result.selectionGeometryPenalty
      : null,
    geometryRankReasons: Array.isArray(result.geometryRankReasons) ? [...result.geometryRankReasons] : []
  };
};

const summarizeVariantComparisonEntry = (candidate, variant, expected) => ({
  value: variant && variant.value ? variant.value : '',
  sourceLabel: candidate && candidate.sourceLabel ? candidate.sourceLabel : '',
  sourceFamily: candidate && candidate.sourceFamily ? candidate.sourceFamily : sourceFamilyFor(candidate),
  diagnosticOnly: isDiagnosticCandidate(candidate),
  matchKind: 'variant',
  isExpected: !!(variant && variant.value === expected),
  score: variant && Number.isFinite(variant.score) ? variant.score : null,
  confidence: variant && Number.isFinite(variant.confidence) ? variant.confidence : null,
  cropMode: variant && variant.cropMode ? variant.cropMode : '',
  cropRatio: variant && Number.isFinite(variant.cropRatio) ? variant.cropRatio : null,
  splitOffsetRatio: variant && Number.isFinite(variant.splitOffsetRatio) ? variant.splitOffsetRatio : 0,
  splitMode: variant && variant.splitMode ? variant.splitMode : '',
  geometryRankPenalty: variant && Number.isFinite(variant.geometryRankPenalty) ? variant.geometryRankPenalty : null,
  selectionGeometryPenalty: null,
  geometryRankReasons: variant && Array.isArray(variant.geometryRankReasons)
    ? [...variant.geometryRankReasons]
    : []
});

const findBestExpectedCandidate = (candidates, expected) => {
  const matches = [];
  candidates.forEach((candidate) => {
    if (candidateValue(candidate) === expected) {
      matches.push(summarizeComparisonEntry(candidate, expected, 'direct'));
    }
    const variants = candidate && candidate.result && Array.isArray(candidate.result.variantCandidates)
      ? candidate.result.variantCandidates
      : [];
    variants.forEach((variant) => {
      if (variant && variant.value === expected) {
        matches.push(summarizeVariantComparisonEntry(candidate, variant, expected));
      }
    });
  });
  return matches
    .filter(Boolean)
    .sort((a, b) => (
      (b.score ?? -1) - (a.score ?? -1)
      || (b.confidence ?? -1) - (a.confidence ?? -1)
    ))[0] || null;
};

const describeSelectedVsExpected = (candidates, expected, productionSelected) => {
  const selectedSource = productionSelected && productionSelected.sourceLabel
    ? productionSelected.sourceLabel
    : '';
  const selectedCandidate = selectedSource
    ? candidates.find((candidate) => candidate && candidate.sourceLabel === selectedSource) || null
    : null;
  const selected = selectedCandidate
    ? summarizeComparisonEntry(selectedCandidate, expected, 'selected')
    : {
      value: productionSelected && productionSelected.value ? productionSelected.value : '',
      sourceLabel: selectedSource,
      matchKind: 'selected',
      isExpected: false,
      score: productionSelected && Number.isFinite(productionSelected.score) ? productionSelected.score : null,
      confidence: productionSelected && Number.isFinite(productionSelected.confidence)
        ? productionSelected.confidence
        : null,
      cropMode: productionSelected && productionSelected.cropMode ? productionSelected.cropMode : '',
      cropRatio: productionSelected && Number.isFinite(productionSelected.cropRatio) ? productionSelected.cropRatio : null,
      splitOffsetRatio: productionSelected && Number.isFinite(productionSelected.splitOffsetRatio)
        ? productionSelected.splitOffsetRatio
        : 0,
      splitMode: productionSelected && productionSelected.splitMode ? productionSelected.splitMode : '',
      geometryRankPenalty: productionSelected && Number.isFinite(productionSelected.geometryRankPenalty)
        ? productionSelected.geometryRankPenalty
        : null,
      selectionGeometryPenalty: productionSelected && Number.isFinite(productionSelected.selectionGeometryPenalty)
        ? productionSelected.selectionGeometryPenalty
        : null,
      geometryRankReasons: productionSelected && Array.isArray(productionSelected.geometryRankReasons)
        ? [...productionSelected.geometryRankReasons]
        : []
    };
  const bestExpected = findBestExpectedCandidate(candidates, expected);
  return {
    selected,
    bestExpected,
    scoreDelta: selected && bestExpected && Number.isFinite(selected.score) && Number.isFinite(bestExpected.score)
      ? selected.score - bestExpected.score
      : null
  };
};

const describeExpectedCoverage = (candidates, expected) => {
  for (const candidate of candidates) {
    if (candidateMatchKind(candidate, expected) === 'direct') {
      return {
        bucket: 'expected-present-as-candidate',
        sourceLabel: candidate.sourceLabel || '',
        matchKind: 'direct',
        diagnosticOnly: isDiagnosticCandidate(candidate)
      };
    }
  }
  for (const candidate of candidates) {
    const variants = candidate && candidate.result && Array.isArray(candidate.result.variantCandidates)
      ? candidate.result.variantCandidates
      : [];
    const matchedVariant = variants.find((variant) => variant && variant.value === expected) || null;
    if (matchedVariant) {
      return {
        bucket: 'expected-present-as-internal-variant',
        sourceLabel: candidate.sourceLabel || '',
        matchKind: 'variant',
        diagnosticOnly: isDiagnosticCandidate(candidate),
        cropMode: matchedVariant.cropMode || '',
        registerSelectionEligible: matchedVariant.registerSelectionEligible === true
      };
    }
  }
  return {
    bucket: 'expected-absent-from-expanded-candidates',
    sourceLabel: '',
    matchKind: 'absent'
  };
};

const confidenceText = (value) => Number.isFinite(value) ? `${value.toFixed(1)}%` : 'n/a';

const scoreText = (value) => Number.isFinite(value) ? value.toFixed(3) : 'n/a';

const describeFailureSplit = (coverage) => {
  const bucket = coverage && coverage.bucket ? coverage.bucket : '';
  if (bucket === 'expected-absent-from-expanded-candidates') {
    return {
      bucket: 'candidate-coverage',
      nextAction: 'upstream crop normalization',
      diagnosticHit: false
    };
  }
  if (coverage && coverage.diagnosticOnly) {
    return {
      bucket: 'shadow-coverage-gain',
      nextAction: 'promote-safe crop construction candidate',
      diagnosticHit: true
    };
  }
  return {
    bucket: 'selection-arbitration',
    nextAction: 'guarded selection review',
    diagnosticHit: false
  };
};

const comparisonText = (entry) => {
  if (!entry) {
    return 'none';
  }
  const reasons = Array.isArray(entry.geometryRankReasons) && entry.geometryRankReasons.length
    ? ` reasons ${entry.geometryRankReasons.join('+')}`
    : '';
  const crop = entry.cropMode
    ? ` crop ${entry.cropMode}${Number.isFinite(entry.cropRatio) ? ` ${entry.cropRatio.toFixed(3)}` : ''}`
    : '';
  const split = entry.splitMode
    ? ` split ${entry.splitMode}${Number.isFinite(entry.splitOffsetRatio) ? ` ${entry.splitOffsetRatio.toFixed(3)}` : ''}`
    : '';
  const diagnostic = entry.diagnosticOnly ? ' diagnostic' : '';
  const family = entry.sourceFamily ? ` family ${entry.sourceFamily}` : '';
  return `${entry.value || 'no-read'} @ ${entry.sourceLabel || 'unknown'}${family}${diagnostic} score ${scoreText(entry.score)} conf ${confidenceText(entry.confidence)}${crop}${split}${reasons}`;
};

const writeCandidateImages = async (row, candidate, rowDir, index) => {
  const token = `${String(index + 1).padStart(2, '0')}_${sanitizeFileToken(candidate.sourceLabel || 'candidate')}`;
  const debugImages = candidate.debugImages || {};
  const stripPath = await writeDataUrl(debugImages.strip, path.join(rowDir, `${token}_strip`));
  const cellSheetPath = await writeDataUrl(debugImages.cellSheet, path.join(rowDir, `${token}_cells`));
  const cellPaths = [];
  if (Array.isArray(debugImages.cells)) {
    for (let cellIndex = 0; cellIndex < debugImages.cells.length; cellIndex += 1) {
      const cellPath = await writeDataUrl(
        debugImages.cells[cellIndex],
        path.join(rowDir, `${token}_cell${cellIndex + 1}`)
      );
      cellPaths.push(cellPath);
    }
  }
  return {
    ...candidate,
    sourceFamily: candidate.sourceFamily || sourceFamilyFor(candidate),
    imageFiles: {
      strip: stripPath,
      cellSheet: cellSheetPath,
      cells: cellPaths
    }
  };
};

const relativeFrom = (fromDir, value) => value ? path.relative(fromDir, value).replace(/\\/g, '/') : '';

const buildReportHtml = (rows, outputDir) => {
  const imageBlock = (label, filePath) => {
    if (!filePath) {
      return `<div class="image missing"><strong>${htmlEscape(label)}</strong><span>missing</span></div>`;
    }
    return `<div class="image"><strong>${htmlEscape(label)}</strong><img src="${htmlEscape(relativeFrom(outputDir, filePath))}" /></div>`;
  };

  const rowHtml = rows.map((row) => {
    const candidateHtml = row.candidates.map((candidate, index) => {
      const result = candidate.result || {};
      const isSelected = row.productionSelectedSource === candidate.sourceLabel;
      const isReadable = !!candidate.value;
      const variantHtml = Array.isArray(result.variantCandidates) && result.variantCandidates.length
        ? `<p class="variants"><strong>internal variants:</strong> ${result.variantCandidates.map((variant) => {
          const score = Number.isFinite(variant.score) ? variant.score.toFixed(3) : 'n/a';
          const crop = `${variant.cropMode || 'n/a'}${Number.isFinite(variant.cropRatio) ? ` ${variant.cropRatio.toFixed(3)}` : ''}`;
          const split = variant.splitMode
            ? ` split ${variant.splitMode}${Number.isFinite(variant.splitOffsetRatio) ? ` ${variant.splitOffsetRatio.toFixed(3)}` : ''}`
            : '';
          const eligibility = variant.registerSelectionEligible ? ' eligible' : '';
          const marker = variant.value === row.expected ? ' *expected*' : '';
          return `${htmlEscape(variant.value || 'no-read')} (${score}, ${htmlEscape(crop)}${htmlEscape(split)}${eligibility})${marker}`;
        }).join(' | ')}</p>`
        : '';
      return `
        <article class="candidate ${candidateMatchesExpected(candidate, row.expected) ? 'expected' : ''} ${isSelected ? 'selected' : ''} ${isReadable ? '' : 'unreadable'}">
          <h3>${index + 1}. ${htmlEscape(candidate.sourceLabel || 'unknown')}</h3>
          <p class="meta">
            <span>family ${htmlEscape(candidate.sourceFamily || sourceFamilyFor(candidate))}</span>
            <span>value <b>${htmlEscape(candidate.value || 'no-read')}</b></span>
            <span>score ${Number.isFinite(result.score) ? result.score.toFixed(3) : 'n/a'}</span>
            <span>base ${Number.isFinite(result.baseScore) ? result.baseScore.toFixed(3) : 'n/a'}</span>
            <span>diag geom ${Number.isFinite(result.diagnosticGeometryScoreAdjustment) ? result.diagnosticGeometryScoreAdjustment.toFixed(3) : 'n/a'}</span>
            <span>confidence ${confidenceText(result.confidence)}</span>
            <span>digits ${htmlEscape(Array.isArray(result.cellDigits) ? result.cellDigits.join(' ') : 'n/a')}</span>
            <span>cells ${htmlEscape(Array.isArray(result.cellConfidences) ? result.cellConfidences.map((v) => confidenceText(v)).join(' / ') : 'n/a')}</span>
            <span>crop ${htmlEscape(result.cropMode || 'n/a')}${Number.isFinite(result.cropRatio) ? ` ${result.cropRatio.toFixed(3)}` : ''}</span>
            <span>split ${htmlEscape(result.splitMode || 'equal-cells')}${Number.isFinite(result.splitOffsetRatio) ? ` ${result.splitOffsetRatio.toFixed(3)}` : ''}</span>
            ${candidate.diagnosticOnly ? `<span class="pill">diagnostic ${htmlEscape(candidate.probeKind || 'candidate')}</span>` : ''}
            ${isSelected ? '<span class="pill">normal selected source</span>' : ''}
          </p>
          ${variantHtml}
          <div class="images">
            ${imageBlock('strip', candidate.imageFiles.strip)}
            ${imageBlock('cell sheet', candidate.imageFiles.cellSheet)}
          </div>
        </article>
      `;
    }).join('\n');
    return `
      <section class="row ${htmlEscape(row.coverage.bucket)}">
        <h2>${htmlEscape(row.filename)}</h2>
        <p class="meta">
          <span>expected <b>${htmlEscape(row.expected)}</b></span>
          <span>normal selected <b>${htmlEscape(row.productionSelected || 'no-read')}</b></span>
          <span>selected source ${htmlEscape(row.productionSelectedSource || 'n/a')}</span>
          <span>coverage ${htmlEscape(row.coverage.bucket)}</span>
          <span>split ${htmlEscape(row.failureSplit.bucket)}</span>
          <span>next ${htmlEscape(row.failureSplit.nextAction)}</span>
          <span>coverage source ${htmlEscape(row.coverage.sourceLabel || 'n/a')}</span>
          <span>families ${htmlEscape(formatCountMap(row.sourceFamilyCounts))}</span>
          <span>expected hit families ${htmlEscape(formatCountMap(row.expectedHitFamilyCounts))}</span>
          ${row.failureSplit.diagnosticHit ? '<span class="pill">shadow probe found expected</span>' : ''}
          <span>readable candidates ${row.readableCandidateCount}</span>
          <span>candidate count ${row.candidates.length}</span>
        </p>
        <p class="comparison">
          <strong>selected:</strong> ${htmlEscape(comparisonText(row.selectionComparison.selected))}
          <br />
          <strong>best expected:</strong> ${htmlEscape(comparisonText(row.selectionComparison.bestExpected))}
          <br />
          <strong>score delta:</strong> ${scoreText(row.selectionComparison.scoreDelta)}
        </p>
        ${candidateHtml}
      </section>
    `;
  }).join('\n');

  return `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Jarvis Cell Crop Failure QA</title>
  <style>
    body { margin: 0; padding: 28px; background: #f6efe2; color: #172033; font-family: Georgia, 'Times New Roman', serif; }
    h1 { margin: 0 0 8px; }
    .subtitle, .meta { color: #667085; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
    .row, .candidate { border: 1px solid #d7dee8; border-radius: 16px; background: rgba(255,255,255,0.86); padding: 14px; margin: 0 0 18px; }
    .candidate { box-shadow: 0 8px 24px rgba(23,32,51,0.06); }
    .candidate.expected { border-color: #0f7a4f; }
    .candidate.selected { border-color: #b45309; }
    .candidate.unreadable { opacity: 0.72; }
    .expected-present-as-candidate, .expected-present-as-internal-variant { border-color: #0f7a4f; }
    .expected-absent-from-expanded-candidates { border-color: #b42318; }
    .meta { display: flex; flex-wrap: wrap; gap: 8px; }
    .meta span { border: 1px solid #d7dee8; border-radius: 999px; padding: 4px 8px; background: #fff; }
    .comparison { color: #475467; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; line-height: 1.6; }
    .variants { color: #475467; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; line-height: 1.5; }
    .pill { color: #b45309; border-color: rgba(180,83,9,0.35) !important; }
    .images { display: grid; grid-template-columns: minmax(220px, 1fr) minmax(260px, 1fr); gap: 10px; align-items: start; margin-top: 10px; }
    .image { border: 1px solid #d7dee8; border-radius: 12px; background: #fbfcfe; padding: 8px; }
    .image strong { display: block; margin-bottom: 6px; color: #667085; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }
    img { display: block; max-width: 100%; height: auto; border-radius: 8px; background: #111827; }
    .missing { display: grid; place-items: center; min-height: 80px; }
  </style>
</head>
<body>
  <h1>Jarvis Cell Crop Failure QA</h1>
  <p class="subtitle">Rows where production OCR did not match the expected reading. Coverage buckets separate missing candidates from selection losses. Generated ${htmlEscape(new Date().toISOString())}.</p>
  ${rowHtml}
</body>
</html>`;
};

const main = async () => {
  const outputDir = path.join(OUTPUT_ROOT, timestampId());
  const imagesDir = path.join(outputDir, 'images');
  await fsp.mkdir(imagesDir, { recursive: true });
  const allRows = parseCsv(await fsp.readFile(path.join(ROOT_DIR, 'assets', 'meter_readings.csv'), 'utf8'));
  const rows = QA_FILES.size
    ? allRows.filter((row) => QA_FILES.has(row.filename))
    : allRows;
  if (QA_FILES.size && !rows.length) {
    throw new Error(`CELL_CROP_QA_FILES did not match any rows: ${[...QA_FILES].join(', ')}`);
  }
  const processes = await ensureServices();
  const browser = await chromium.launch({ headless: true });
  const reportRows = [];
  try {
    const page = await browser.newPage();
    await page.goto(FRONTEND_URL, { waitUntil: 'networkidle' });
    for (const row of rows) {
      process.stdout.write(`Cell QA inspecting ${row.filename}\n`);
      const productionRun = await runImage(page, row, { maxPrimaryCandidates: 4 });
      const oracleRun = await runImage(page, row, {
        maxPrimaryCandidates: MAX_PRIMARY_CANDIDATES,
        decodeDiagnosticCandidates: true,
        maxDiagnosticCandidates: MAX_DIAGNOSTIC_CANDIDATES,
        exportCandidateImages: true
      });
      if (productionRun.error || oracleRun.error || !oracleRun.selectionLog) {
        continue;
      }
      const rawCandidates = Array.isArray(oracleRun.selectionLog.candidateTrace)
        ? oracleRun.selectionLog.candidateTrace
        : [];
      const allCandidates = rawCandidates
        .map((candidate) => ({
          ...candidate,
          value: candidateValue(candidate),
          sourceFamily: sourceFamilyFor(candidate)
        }));
      const readable = allCandidates.filter((candidate) => candidate.value);
      const productionSelected = productionRun.selectionLog && productionRun.selectionLog.selected
        ? productionRun.selectionLog.selected
        : null;
      const productionSelectedValue = productionSelected && productionSelected.value ? productionSelected.value : '';
      if (productionSelectedValue === row.value) {
        continue;
      }
      const coverage = describeExpectedCoverage(readable, row.value);
      const selectionComparison = describeSelectedVsExpected(readable, row.value, productionSelected);
      const failureSplit = describeFailureSplit(coverage);
      const sourceFamilyCounts = countCandidateFamilies(allCandidates);
      const expectedHitFamilyCounts = countExpectedHitFamilies(readable, row.value);
      const rowDir = path.join(imagesDir, sanitizeFileToken(row.filename));
      await fsp.mkdir(rowDir, { recursive: true });
      const candidates = [];
      for (let index = 0; index < allCandidates.length; index += 1) {
        candidates.push(await writeCandidateImages(row, allCandidates[index], rowDir, index));
      }
      reportRows.push({
        filename: row.filename,
        expected: row.value,
        productionSelected: productionSelectedValue,
        productionSelectedSource: productionSelected && productionSelected.sourceLabel ? productionSelected.sourceLabel : '',
        coverage,
        failureSplit,
        selectionComparison,
        sourceFamilyCounts,
        expectedHitFamilyCounts,
        readableCandidateCount: readable.length,
        candidates
      });
    }
  } finally {
    await browser.close();
    for (const processHandle of processes.reverse()) {
      await processHandle.stop();
    }
  }

  const reportPath = path.join(outputDir, 'cell-crop-failure-qa.html');
  const summaryPath = path.join(outputDir, 'summary.json');
  const coverageCounts = reportRows.reduce((counts, row) => {
    counts[row.coverage.bucket] = (counts[row.coverage.bucket] || 0) + 1;
    return counts;
  }, {});
  const failureSplitCounts = reportRows.reduce((counts, row) => {
    const key = row.failureSplit && row.failureSplit.bucket ? row.failureSplit.bucket : 'unknown';
    counts[key] = (counts[key] || 0) + 1;
    return counts;
  }, {});
  const shadowProbeHitRowCount = reportRows.filter((row) => (
    row.failureSplit && row.failureSplit.diagnosticHit
  )).length;
  const registerLocalizationHitRowCount = reportRows.filter((row) => (
    row.expectedHitFamilyCounts && row.expectedHitFamilyCounts.regloc > 0
  )).length;
  const splitProbeHitRowCount = reportRows.filter((row) => (
    row.candidates.some((candidate) => hasSplitProbeExpectedHit(candidate, row.expected))
  )).length;
  const splitProbeOnlyHitRowCount = reportRows.filter((row) => (
    row.candidates.some((candidate) => hasSplitProbeExpectedHit(candidate, row.expected))
    && !row.candidates.some((candidate) => hasNonSplitExpectedHit(candidate, row.expected))
  )).length;
  const sourceFamilyExpectedHitCounts = reportRows.reduce((counts, row) => {
    Object.keys(row.expectedHitFamilyCounts || {}).forEach((family) => {
      incrementCount(counts, family);
    });
    return counts;
  }, {});
  await fsp.writeFile(reportPath, buildReportHtml(reportRows, outputDir), 'utf8');
  await fsp.writeFile(summaryPath, JSON.stringify({
    generatedAt: new Date().toISOString(),
    maxPrimaryCandidates: MAX_PRIMARY_CANDIDATES,
    maxDiagnosticCandidates: MAX_DIAGNOSTIC_CANDIDATES,
    rowCount: reportRows.length,
    expectedAbsentRowCount: coverageCounts['expected-absent-from-expanded-candidates'] || 0,
    expectedPresentRowCount: (
      (coverageCounts['expected-present-as-candidate'] || 0)
      + (coverageCounts['expected-present-as-internal-variant'] || 0)
    ),
    coverageCounts,
    failureSplitCounts,
    shadowProbeHitRowCount,
    registerLocalizationHitRowCount,
    splitProbeHitRowCount,
    splitProbeOnlyHitRowCount,
    sourceFamilyExpectedHitCounts,
    rows: reportRows.map((row) => ({
      ...row,
      candidates: row.candidates.map((candidate) => ({
        stage: candidate.stage,
        sourceLabel: candidate.sourceLabel,
        diagnosticOnly: candidate.diagnosticOnly === true,
        probeKind: candidate.probeKind || null,
        sourceFamily: candidate.sourceFamily || sourceFamilyFor(candidate),
        width: candidate.width,
        height: candidate.height,
        result: candidate.result,
        value: candidate.value,
        imageFiles: {
          strip: relativeFrom(ROOT_DIR, candidate.imageFiles.strip),
          cellSheet: relativeFrom(ROOT_DIR, candidate.imageFiles.cellSheet)
        }
      }))
    }))
  }, null, 2), 'utf8');

  console.log(JSON.stringify({
    outputDir: relativeFrom(ROOT_DIR, outputDir),
    report: relativeFrom(ROOT_DIR, reportPath),
    summary: relativeFrom(ROOT_DIR, summaryPath),
    rowCount: reportRows.length,
    expectedAbsentRowCount: coverageCounts['expected-absent-from-expanded-candidates'] || 0,
    expectedPresentRowCount: (
      (coverageCounts['expected-present-as-candidate'] || 0)
      + (coverageCounts['expected-present-as-internal-variant'] || 0)
    ),
    coverageCounts,
    failureSplitCounts,
    shadowProbeHitRowCount,
    registerLocalizationHitRowCount,
    splitProbeHitRowCount,
    splitProbeOnlyHitRowCount,
    sourceFamilyExpectedHitCounts,
    rows: reportRows.map((row) => ({
      filename: row.filename,
      expected: row.expected,
      selected: row.productionSelected || 'no-read',
      coverage: row.coverage.bucket,
      failureSplit: row.failureSplit ? row.failureSplit.bucket : '',
      nextAction: row.failureSplit ? row.failureSplit.nextAction : '',
      shadowProbeHit: !!(row.failureSplit && row.failureSplit.diagnosticHit),
      splitProbeHit: row.candidates.some((candidate) => hasSplitProbeExpectedHit(candidate, row.expected)),
      splitProbeOnlyHit: (
        row.candidates.some((candidate) => hasSplitProbeExpectedHit(candidate, row.expected))
        && !row.candidates.some((candidate) => hasNonSplitExpectedHit(candidate, row.expected))
      ),
      scoreDelta: row.selectionComparison && Number.isFinite(row.selectionComparison.scoreDelta)
        ? Number(row.selectionComparison.scoreDelta.toFixed(3))
        : null,
      selectedSource: row.selectionComparison && row.selectionComparison.selected
        ? row.selectionComparison.selected.sourceLabel
        : '',
      bestExpectedSource: row.selectionComparison && row.selectionComparison.bestExpected
        ? row.selectionComparison.bestExpected.sourceLabel
        : '',
      sourceFamilyCounts: row.sourceFamilyCounts,
      expectedHitFamilyCounts: row.expectedHitFamilyCounts,
      readableCandidates: row.readableCandidateCount,
      candidates: row.candidates.length
    }))
  }, null, 2));
};

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
