import { OCR_CONFIG } from './config.js';
import { setOcrDebugHooks, startDebugSession, addDebugStage, commitDebugSession } from './debug-hooks.js';
import {
  loadImageBitmap,
  drawImageToCanvas,
  preprocessCanvas,
  cropCanvas,
  normalizeRectToCanvas,
  drawOverlayCanvas,
  normalizeAngle
} from './canvas-utils.js';
import { buildDigitCandidates } from './alignment.js';
import { readDigitsByCells } from './recognition.js';
import { detectNeuralRoi } from './neural-roi.js';
import { predictDigitStrip } from './digit-classifier.js';

const resolveNeuralRoiRect = (canvas, roiDetection, roiConfig) => {
  const rawRect = normalizeRectToCanvas(canvas, {
    x: roiDetection.rect.x * canvas.width,
    y: roiDetection.rect.y * canvas.height,
    width: roiDetection.rect.width * canvas.width,
    height: roiDetection.rect.height * canvas.height
  });
  const expandX = Number.isFinite(roiConfig.expandX) ? roiConfig.expandX : 0;
  const expandY = Number.isFinite(roiConfig.expandY) ? roiConfig.expandY : 0;
  return normalizeRectToCanvas(canvas, {
    x: rawRect.x - rawRect.width * expandX,
    y: rawRect.y - rawRect.height * expandY,
    width: rawRect.width * (1 + expandX * 2),
    height: rawRect.height * (1 + expandY * 2)
  });
};

const addNeuralRoiDebugStages = (debugSession, baseCanvas, roiRect, roiDetection) => {
  if (!debugSession) {
    return;
  }
  const overlay = drawOverlayCanvas(baseCanvas, [
    {
      x: roiRect.x,
      y: roiRect.y,
      width: roiRect.width,
      height: roiRect.height,
      label: `neural roi ${(roiDetection.confidence * 100).toFixed(0)}%`,
      color: '#06b6d4'
    }
  ]);
  addDebugStage(debugSession, '0. neural roi detection', overlay);
};

const addNeuralRoiMissStage = (debugSession, baseCanvas, probe) => {
  if (!debugSession) {
    return;
  }
  const reason = probe && probe.reason ? probe.reason : 'unknown';
  const geometry = probe && probe.geometry ? probe.geometry : null;
  const geometrySuffix = geometry && geometry.metric
    ? ` (${geometry.metric}=${Number.isFinite(geometry.value) ? geometry.value.toFixed(3) : 'n/a'})`
    : '';
  const confidence = Number.isFinite(probe && probe.confidence) ? ` ${(probe.confidence * 100).toFixed(1)}%` : '';
  const overlay = drawOverlayCanvas(baseCanvas, [
    {
      x: Math.round(baseCanvas.width * 0.02),
      y: Math.round(baseCanvas.height * 0.02),
      width: Math.round(baseCanvas.width * 0.96),
      height: Math.round(baseCanvas.height * 0.96),
      label: `neural roi miss: ${reason}${geometrySuffix}${confidence}`,
      color: '#ef4444'
    }
  ]);
  addDebugStage(debugSession, '0. neural roi detection', overlay);
};

const formatNeuralRoiMissReason = (probe) => {
  const reason = probe && probe.reason ? probe.reason : 'unknown';
  const geometry = probe && probe.geometry ? probe.geometry : null;
  if (geometry && geometry.metric) {
    const value = Number.isFinite(geometry.value) ? geometry.value.toFixed(3) : 'n/a';
    return `${reason}:${geometry.metric}=${value}`;
  }
  return reason;
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const isEdgeSourceLabel = (sourceLabel) => (
  typeof sourceLabel === 'string' && sourceLabel.includes('-edge')
);
const isClassifierAvailabilityReject = (detail) => (
  !!detail && (
    detail.reason === 'classifier-unavailable'
    || detail.reason === 'classifier-disabled'
  )
);

const recordSelectionEvidence = (
  evidenceMap,
  reading,
  { sourceLabel = '', isTopPick = false } = {}
) => {
  if (!reading || !reading.value) {
    return;
  }

  const value = String(reading.value).replace(/\D/g, '');
  if (!value) {
    return;
  }

  const score = Number.isFinite(reading.score) ? reading.score : 0;
  const confidence = Number.isFinite(reading.confidence) ? reading.confidence : 0;
  const existing = evidenceMap.get(value) || {
    value,
    hits: 0,
    topHits: 0,
    totalScore: 0,
    bestScore: -1,
    bestConfidence: 0,
    sources: new Set(),
    edgeHits: 0,
    edgeTopHits: 0,
    nonEdgeHits: 0,
    nonEdgeTopHits: 0,
    edgeSources: new Set(),
    nonEdgeSources: new Set()
  };

  existing.hits += 1;
  existing.totalScore += score;
  if (isTopPick) {
    existing.topHits += 1;
  }
  if (sourceLabel) {
    existing.sources.add(sourceLabel);
    if (isEdgeSourceLabel(sourceLabel)) {
      existing.edgeHits += 1;
      existing.edgeSources.add(sourceLabel);
      if (isTopPick) {
        existing.edgeTopHits += 1;
      }
    } else {
      existing.nonEdgeHits += 1;
      existing.nonEdgeSources.add(sourceLabel);
      if (isTopPick) {
        existing.nonEdgeTopHits += 1;
      }
    }
  }
  if (score > existing.bestScore) {
    existing.bestScore = score;
  }
  if (confidence > existing.bestConfidence) {
    existing.bestConfidence = confidence;
  }

  evidenceMap.set(value, existing);
};

const rankSelectionEvidence = (evidenceMap) => {
  return [...evidenceMap.values()]
    .map((entry) => {
      const averageScore = entry.hits ? entry.totalScore / entry.hits : 0;
      const consensusBoost = clamp((entry.topHits - 1) * 0.12 + (entry.hits - entry.topHits) * 0.04, 0, 0.3);
      const sourceSpreadBoost = clamp((entry.sources.size - 1) * 0.02, 0, 0.08);
      const nonEdgeSupportBoost = clamp(entry.nonEdgeHits * 0.03 + entry.nonEdgeTopHits * 0.04, 0, 0.12);
      const edgeOnlyPenalty = entry.edgeHits > 0 && entry.nonEdgeHits === 0 ? 0.07 : 0;
      const preferredLengthBoost = entry.value.length === OCR_CONFIG.preferredDigits ? 0.05 : -0.08;
      const leadingZeroPenalty = (
        entry.value.length === OCR_CONFIG.preferredDigits
        && entry.value.startsWith('0')
      ) ? 0.07 : 0;
      const score = clamp(
        entry.bestScore * 0.58
          + averageScore * 0.27
          + consensusBoost
          + sourceSpreadBoost
          + nonEdgeSupportBoost
          + preferredLengthBoost
          - edgeOnlyPenalty
          - leadingZeroPenalty,
        0,
        0.99
      );

      return {
        ...entry,
        averageScore,
        score,
        sourceCount: entry.sources.size,
        edgeHits: entry.edgeHits,
        edgeTopHits: entry.edgeTopHits,
        nonEdgeHits: entry.nonEdgeHits,
        nonEdgeTopHits: entry.nonEdgeTopHits,
        edgeSourceCount: entry.edgeSources.size,
        nonEdgeSourceCount: entry.nonEdgeSources.size
      };
    })
    .sort((a, b) => b.score - a.score || b.topHits - a.topHits || b.hits - a.hits || b.bestScore - a.bestScore);
};

const buildSelectionSummary = (rankedEvidence, limit = 3) => {
  return rankedEvidence.slice(0, limit).map((entry) => ({
    value: entry.value,
    score: Number(entry.score.toFixed(3)),
    bestScore: Number(entry.bestScore.toFixed(3)),
    averageScore: Number(entry.averageScore.toFixed(3)),
    hits: entry.hits,
    topHits: entry.topHits,
    sourceCount: entry.sourceCount
  }));
};

const pushSelectionLog = (payload) => {
  if (typeof window !== 'undefined') {
    if (!Array.isArray(window.__jarvisOcrSelectionLogs)) {
      window.__jarvisOcrSelectionLogs = [];
    }
    window.__jarvisOcrSelectionLogs.push(payload);
    if (window.__jarvisOcrSelectionLogs.length > 300) {
      window.__jarvisOcrSelectionLogs.shift();
    }
  }
  console.info('[OCR] selection', JSON.stringify(payload));
};

const serializeCellConfidences = (confidences) => {
  if (!Array.isArray(confidences)) {
    return null;
  }
  return confidences.map((value) => {
    if (!Number.isFinite(value)) {
      return null;
    }
    return Number(value.toFixed(1));
  });
};

const serializeStripConfidence = (value) => {
  if (!Number.isFinite(value)) {
    return null;
  }
  const percent = value <= 1 ? value * 100 : value;
  return Number(clamp(percent, 0, 100).toFixed(1));
};

const serializeStripConfidences = (confidences) => {
  if (!Array.isArray(confidences)) {
    return null;
  }
  return confidences.map((value) => serializeStripConfidence(value));
};

const isRepeatedDigitReading = (value) => {
  if (typeof value !== 'string' || value.length !== OCR_CONFIG.preferredDigits) {
    return false;
  }
  return new Set(value.split('')).size === 1;
};

const compareStripReaderConfidence = (a, b) => {
  const confidenceA = a && Number.isFinite(a.rawConfidence) ? a.rawConfidence : -1;
  const confidenceB = b && Number.isFinite(b.rawConfidence) ? b.rawConfidence : -1;
  return confidenceB - confidenceA;
};

const cloneStripReaderProbe = (probe) => {
  if (!probe || typeof probe !== 'object') {
    return null;
  }
  return {
    ...probe,
    digits: Array.isArray(probe.digits) ? [...probe.digits] : probe.digits,
    digitConfidences: Array.isArray(probe.digitConfidences)
      ? [...probe.digitConfidences]
      : probe.digitConfidences,
    topKByPosition: Array.isArray(probe.topKByPosition)
      ? probe.topKByPosition.map((entries) => (
        Array.isArray(entries)
          ? entries.map((entry) => ({ ...entry }))
          : entries
      ))
      : probe.topKByPosition
  };
};

const summarizeStripReaderBySource = (candidates, selectedSourceLabel = null) => {
  const sourceMap = new Map();
  candidates.forEach((candidate) => {
    if (!candidate) {
      return;
    }
    const sourceLabel = candidate.sourceLabel || 'unknown';
    const existing = sourceMap.get(sourceLabel) || {
      sourceLabel,
      attempts: 0,
      okCount: 0,
      matchesSelectedSource: !!selectedSourceLabel && sourceLabel === selectedSourceLabel,
      best: null,
      values: []
    };
    existing.attempts += 1;
    if (candidate.ok) {
      existing.okCount += 1;
      existing.values.push({
        value: candidate.value || null,
        confidence: Number.isFinite(candidate.confidence) ? candidate.confidence : null,
        rawConfidence: Number.isFinite(candidate.rawConfidence) ? candidate.rawConfidence : null,
        stage: candidate.stage || null
      });
      if (!existing.best || compareStripReaderConfidence(existing.best, candidate) > 0) {
        existing.best = candidate;
      }
    }
    sourceMap.set(sourceLabel, existing);
  });

  return [...sourceMap.values()]
    .map((entry) => ({
      sourceLabel: entry.sourceLabel,
      attempts: entry.attempts,
      okCount: entry.okCount,
      matchesSelectedSource: entry.matchesSelectedSource,
      bestValue: entry.best && entry.best.value ? entry.best.value : null,
      bestConfidence: entry.best && Number.isFinite(entry.best.confidence) ? entry.best.confidence : null,
      bestRawConfidence: entry.best && Number.isFinite(entry.best.rawConfidence) ? entry.best.rawConfidence : null,
      values: entry.values
    }))
    .sort((a, b) => (
      (b.matchesSelectedSource ? 1 : 0) - (a.matchesSelectedSource ? 1 : 0)
      || b.okCount - a.okCount
      || (b.bestRawConfidence ?? -1) - (a.bestRawConfidence ?? -1)
      || String(a.sourceLabel).localeCompare(String(b.sourceLabel))
    ));
};

const buildStripReaderSummary = (stripReaderTrace, finalResult) => {
  const rawCandidates = stripReaderTrace && Array.isArray(stripReaderTrace.candidates)
    ? stripReaderTrace.candidates
    : [];
  const candidates = rawCandidates
    .map((candidate) => cloneStripReaderProbe(candidate))
    .filter(Boolean);
  if (!candidates.length) {
    return null;
  }

  const selectedSourceLabel = finalResult && finalResult.sourceLabel ? finalResult.sourceLabel : null;
  const okCandidates = candidates.filter((candidate) => candidate.ok);
  const confidenceBest = okCandidates.length
    ? cloneStripReaderProbe([...okCandidates].sort(compareStripReaderConfidence)[0])
    : null;
  const selectedSourceCandidate = selectedSourceLabel
    ? cloneStripReaderProbe(
      okCandidates
        .filter((candidate) => candidate.sourceLabel === selectedSourceLabel)
        .sort(compareStripReaderConfidence)[0] || null
    )
    : null;
  const headline = selectedSourceCandidate || confidenceBest;
  if (!headline) {
    return {
      ok: false,
      reason: 'strip-reader-no-successful-candidate',
      preferredSourceLabel: selectedSourceLabel,
      selectionRule: 'selected-source-first-then-confidence',
      headlineReason: 'no-successful-candidate',
      confidenceBest: null,
      selectedSourceCandidate: null,
      bySource: summarizeStripReaderBySource(candidates, selectedSourceLabel),
      candidates
    };
  }

  return {
    ...headline,
    preferredSourceLabel: selectedSourceLabel,
    selectionRule: 'selected-source-first-then-confidence',
    headlineReason: selectedSourceCandidate ? 'selected-source' : 'highest-confidence',
    confidenceBest,
    selectedSourceCandidate,
    bySource: summarizeStripReaderBySource(candidates, selectedSourceLabel),
    candidates
  };
};

const resolveStripReaderDebug = (stripReaderTrace, finalResult) => {
  if (!stripReaderTrace) {
    return null;
  }
  const selectedSourceLabel = finalResult && finalResult.sourceLabel ? finalResult.sourceLabel : null;
  if (
    selectedSourceLabel
    && stripReaderTrace.debugBySource
    && typeof stripReaderTrace.debugBySource.get === 'function'
  ) {
    const selectedSourceDebug = stripReaderTrace.debugBySource.get(selectedSourceLabel);
    if (selectedSourceDebug && selectedSourceDebug.canvas) {
      return selectedSourceDebug;
    }
  }
  return stripReaderTrace.confidenceBestDebug || null;
};

const finalizeSelection = ({
  debugLabel,
  roiUsed,
  bestResult,
  evidenceMap,
  branchUsed,
  rejectSummary = [],
  candidateTrace = [],
  stripReaderTrace = null
}) => {
  const rankedEvidence = rankSelectionEvidence(evidenceMap);
  const evidenceBest = rankedEvidence[0] || null;
  const classifierConfig = OCR_CONFIG.digitClassifier || {};
  let finalResult = bestResult;
  let finalRejectReason = null;
  let finalRejectDetail = null;
  const carryMetadataFromTrace = (value) => {
    if (!value) {
      return null;
    }
    const matches = candidateTrace
      .filter((entry) => entry && entry.result && entry.result.value === value)
      .sort((a, b) => (b.result.score ?? 0) - (a.result.score ?? 0));
    if (!matches.length) {
      return null;
    }
    const bestMatch = matches[0];
    return {
      branch: branchUsed,
      method: bestMatch.result.method || null,
      sourceLabel: bestMatch.sourceLabel || null,
      preprocessMode: 'raw',
      angle: Number.isFinite(bestMatch.result.angle) ? bestMatch.result.angle : null,
      cellDigits: Array.isArray(bestMatch.result.cellDigits) ? bestMatch.result.cellDigits : null,
      cellConfidences: bestMatch.result.cellConfidences || null,
      confidence: Number.isFinite(bestMatch.result.confidence) ? bestMatch.result.confidence : 0
    };
  };

  if (evidenceBest) {
    const currentFinalSupport = finalResult
      ? rankedEvidence.find((entry) => entry.value === finalResult.value) || null
      : null;
    const lowDiversityEdgeResult = !!(
      finalResult
      && isEdgeSourceLabel(finalResult.sourceLabel)
      && typeof finalResult.value === 'string'
      && finalResult.value.length === OCR_CONFIG.preferredDigits
      && new Set(finalResult.value.split('')).size <= 2
    );
    const shouldPromoteLowDiversityEdgeFallback = !!(
      finalResult
      && evidenceBest.value !== finalResult.value
      && evidenceBest.nonEdgeHits >= 1
      && lowDiversityEdgeResult
      && evidenceBest.score >= (finalResult.score ?? -1) - 0.06
      && evidenceBest.bestConfidence >= 75
    );
    const preserveAgreedEdgeConsensus = !!(
      finalResult
      && currentFinalSupport
      && evidenceBest.value !== finalResult.value
      && isEdgeSourceLabel(finalResult.sourceLabel)
      && currentFinalSupport.topHits >= 2
      && currentFinalSupport.edgeTopHits >= 2
      && currentFinalSupport.nonEdgeHits === 0
      && evidenceBest.nonEdgeHits >= 1
      && evidenceBest.topHits === 1
      && evidenceBest.score <= (finalResult.score ?? 0) + 0.03
    );
    const shouldPromoteEvidence = (
      (
        !finalResult
        || evidenceBest.score >= (finalResult.score ?? -1) - 0.03
        || evidenceBest.topHits >= 2
      )
      && !preserveAgreedEdgeConsensus
      || shouldPromoteLowDiversityEdgeFallback
    );

    if (shouldPromoteEvidence) {
      const confidenceFromBest = finalResult && finalResult.value === evidenceBest.value
        ? (finalResult.confidence ?? 0)
        : 0;
      const carryMetadata = finalResult && finalResult.value === evidenceBest.value
        ? finalResult
        : carryMetadataFromTrace(evidenceBest.value);
      finalResult = {
        value: evidenceBest.value,
        confidence: Math.max(evidenceBest.bestConfidence, confidenceFromBest),
        areaRatio: finalResult && finalResult.value === evidenceBest.value
          ? (finalResult.areaRatio ?? 0.28)
          : 0.28,
        score: evidenceBest.score,
        branch: carryMetadata && carryMetadata.branch ? carryMetadata.branch : branchUsed,
        method: carryMetadata && carryMetadata.method ? carryMetadata.method : null,
        sourceLabel: carryMetadata && carryMetadata.sourceLabel ? carryMetadata.sourceLabel : null,
        preprocessMode: carryMetadata && carryMetadata.preprocessMode ? carryMetadata.preprocessMode : null,
        angle: carryMetadata && Number.isFinite(carryMetadata.angle) ? carryMetadata.angle : null,
        cellDigits: carryMetadata && Array.isArray(carryMetadata.cellDigits) ? carryMetadata.cellDigits : null,
        cellConfidences: carryMetadata ? serializeCellConfidences(carryMetadata.cellConfidences) : null
      };
    }
  }

  if (finalResult && isEdgeSourceLabel(finalResult.sourceLabel)) {
    const support = rankedEvidence.find((entry) => entry.value === finalResult.value) || null;
    const hasNonEdgeSupport = !!support && (
      support.nonEdgeHits >= 1
      || support.nonEdgeTopHits >= 1
      || support.nonEdgeSourceCount >= 1
    );
    const cellConfidences = Array.isArray(finalResult.cellConfidences)
      ? finalResult.cellConfidences.filter((value) => Number.isFinite(value))
      : [];
    const averageCellConfidence = cellConfidences.length
      ? (cellConfidences.reduce((sum, value) => sum + value, 0) / cellConfidences.length)
      : 0;
    const minCellConfidence = cellConfidences.length
      ? Math.min(...cellConfidences)
      : 0;
    const isClassifierResult = String(finalResult.method || '').startsWith('digit-classifier');

    if (isClassifierResult) {
      const fallbackEdgeMinAverageConfidence = Number.isFinite(classifierConfig.fallbackEdgeMinAverageConfidence)
        ? clamp(classifierConfig.fallbackEdgeMinAverageConfidence, 0, 100)
        : 65;
      const fallbackEdgeMinCellConfidence = Number.isFinite(classifierConfig.fallbackEdgeMinCellConfidence)
        ? clamp(classifierConfig.fallbackEdgeMinCellConfidence, 0, 100)
        : 35;
      const fallbackEdgeRequireNonEdgeSupport = classifierConfig.fallbackEdgeRequireNonEdgeSupport === true;
      const hasClassifierCellEvidence = (
        cellConfidences.length >= OCR_CONFIG.digitCellCount
        && averageCellConfidence >= fallbackEdgeMinAverageConfidence
        && minCellConfidence >= fallbackEdgeMinCellConfidence
      );
      const classifierEdgeAccepted = fallbackEdgeRequireNonEdgeSupport
        ? hasNonEdgeSupport
        : (hasNonEdgeSupport || hasClassifierCellEvidence);
      if (!classifierEdgeAccepted) {
        finalRejectReason = 'classifier-edge-gate-final-drop';
        finalRejectDetail = {
          stage: 'selection-finalize',
          method: finalResult.method || null,
          sourceLabel: finalResult.sourceLabel || null,
          value: finalResult.value || null,
          hasNonEdgeSupport,
          averageCellConfidence: Number(averageCellConfidence.toFixed(1)),
          minCellConfidence: Number(minCellConfidence.toFixed(1)),
          requiredAverageConfidence: fallbackEdgeMinAverageConfidence,
          requiredMinConfidence: fallbackEdgeMinCellConfidence,
          requireNonEdgeSupport: fallbackEdgeRequireNonEdgeSupport
        };
        finalResult = null;
      }
    } else {
      const hasStrongCellEvidence = (
        cellConfidences.length >= OCR_CONFIG.digitCellCount
        && averageCellConfidence >= 90
        && minCellConfidence >= 82
      );
      if (!hasNonEdgeSupport && !hasStrongCellEvidence) {
        finalRejectReason = 'edge-gate-final-drop';
        finalRejectDetail = {
          stage: 'selection-finalize',
          method: finalResult.method || null,
          sourceLabel: finalResult.sourceLabel || null,
          value: finalResult.value || null,
          hasNonEdgeSupport,
          averageCellConfidence: Number(averageCellConfidence.toFixed(1)),
          minCellConfidence: Number(minCellConfidence.toFixed(1))
        };
        finalResult = null;
      }
    }
  }

  if (finalRejectReason) {
    prependRejectSummary(rejectSummary, finalRejectReason, finalRejectDetail || {});
  }

  const stripReader = buildStripReaderSummary(stripReaderTrace, finalResult);

  pushSelectionLog({
    image: debugLabel,
    roiUsed,
    branchUsed,
    rejectSummary,
    finalRejectReason,
    finalRejectDetail,
    selected: finalResult ? {
      value: finalResult.value,
      score: Number((finalResult.score ?? 0).toFixed(3)),
      confidence: Number((finalResult.confidence ?? 0).toFixed(1)),
      branch: finalResult.branch || branchUsed,
      method: finalResult.method || null,
      sourceLabel: finalResult.sourceLabel || null,
      preprocessMode: finalResult.preprocessMode || null,
      angle: Number.isFinite(finalResult.angle) ? finalResult.angle : null,
      cellDigits: Array.isArray(finalResult.cellDigits) ? finalResult.cellDigits : null,
      cellConfidences: serializeCellConfidences(finalResult.cellConfidences)
    } : null,
    stripReader,
    topCandidates: buildSelectionSummary(rankedEvidence, 3),
    candidateTrace: Array.isArray(candidateTrace) ? candidateTrace : []
  });

  return finalResult;
};

const extractCandidateAngle = (label) => {
  if (!label || typeof label !== 'string') {
    return Number.NaN;
  }
  const tokens = label.split('-');
  for (const token of tokens) {
    const parsed = Number.parseInt(token, 10);
    if (!Number.isFinite(parsed)) {
      continue;
    }
    const normalized = ((parsed % 360) + 360) % 360;
    if (normalized % 90 === 0) {
      return normalized;
    }
  }
  return Number.NaN;
};

const resolveWinningDecodeCanvas = (decodeCanvasBySource, finalSelection) => {
  if (
    !(decodeCanvasBySource instanceof Map)
    || !finalSelection
    || typeof finalSelection.sourceLabel !== 'string'
  ) {
    return null;
  }
  const entries = decodeCanvasBySource.get(finalSelection.sourceLabel);
  if (!Array.isArray(entries) || !entries.length) {
    return null;
  }
  const targetAngle = Number.isFinite(finalSelection.angle) ? normalizeAngle(finalSelection.angle) : null;
  if (targetAngle === null) {
    return entries[0].canvas || null;
  }
  const exactMatch = entries.find((entry) => entry && entry.angle === targetAngle && entry.canvas);
  if (exactMatch) {
    return exactMatch.canvas;
  }
  const neutralMatch = entries.find((entry) => entry && entry.angle === null && entry.canvas);
  if (neutralMatch) {
    return neutralMatch.canvas;
  }
  return entries[0].canvas || null;
};

const resolveWinningDecodeCells = (decodeCanvasBySource, finalSelection) => {
  if (
    !(decodeCanvasBySource instanceof Map)
    || !finalSelection
    || typeof finalSelection.sourceLabel !== 'string'
  ) {
    return null;
  }
  const entries = decodeCanvasBySource.get(finalSelection.sourceLabel);
  if (!Array.isArray(entries) || !entries.length) {
    return null;
  }
  const targetAngle = Number.isFinite(finalSelection.angle) ? normalizeAngle(finalSelection.angle) : null;
  const exactMatch = entries.find((entry) => entry && entry.angle === targetAngle && Array.isArray(entry.cells));
  if (exactMatch) {
    return exactMatch.cells;
  }
  const neutralMatch = entries.find((entry) => entry && entry.angle === null && Array.isArray(entry.cells));
  if (neutralMatch) {
    return neutralMatch.cells;
  }
  const fallback = entries.find((entry) => entry && Array.isArray(entry.cells));
  return fallback ? fallback.cells : null;
};

const buildCellDebugCanvas = (cellCanvases) => {
  if (!Array.isArray(cellCanvases) || !cellCanvases.length) {
    return null;
  }
  const validCells = cellCanvases.filter((cell) => !!(cell && cell.width > 0 && cell.height > 0));
  if (!validCells.length) {
    return null;
  }
  const gap = 8;
  const pad = 8;
  const labelHeight = 18;
  const width = validCells.reduce((sum, cell) => sum + cell.width, 0) + gap * (validCells.length - 1) + pad * 2;
  const height = Math.max(...validCells.map((cell) => cell.height)) + pad * 2 + labelHeight;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#111827';
  ctx.fillRect(0, 0, width, height);
  ctx.font = '12px sans-serif';
  ctx.textBaseline = 'top';
  let x = pad;
  validCells.forEach((cell, index) => {
    ctx.fillStyle = '#e5e7eb';
    ctx.fillText(`cell ${index + 1}`, x, pad);
    ctx.drawImage(cell, x, pad + labelHeight);
    ctx.strokeStyle = '#f97316';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, pad + labelHeight, cell.width, cell.height);
    x += cell.width + gap;
  });
  return canvas;
};

const buildStripReaderDebugCanvas = (stripCanvas, stripReaderResult) => {
  if (!stripCanvas || stripCanvas.width <= 0 || stripCanvas.height <= 0) {
    return null;
  }
  const headerHeight = 42;
  const width = Math.max(stripCanvas.width, 360);
  const height = stripCanvas.height + headerHeight;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#111827';
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = '#e5e7eb';
  ctx.font = '13px sans-serif';
  ctx.textBaseline = 'top';
  const value = stripReaderResult && stripReaderResult.value ? stripReaderResult.value : 'n/a';
  const confidence = stripReaderResult && Number.isFinite(stripReaderResult.confidence)
    ? `${stripReaderResult.confidence.toFixed(1)}%`
    : 'n/a';
  const source = stripReaderResult && stripReaderResult.sourceLabel ? stripReaderResult.sourceLabel : 'unknown';
  ctx.fillText(`strip reader: ${value} (${confidence})`, 8, 6);
  ctx.fillText(`source: ${source}`, 8, 23);
  const x = Math.round((width - stripCanvas.width) / 2);
  ctx.drawImage(stripCanvas, x, headerHeight);
  return canvas;
};

const addWinningCandidateDebugStage = (
  debugSession,
  candidates,
  finalSelection,
  decodeCanvasBySource = null,
  stripReaderDebug = null
) => {
  if (!debugSession || !Array.isArray(candidates) || !finalSelection || !finalSelection.sourceLabel) {
    return;
  }
  const winningDecodeCanvas = resolveWinningDecodeCanvas(decodeCanvasBySource, finalSelection);
  if (winningDecodeCanvas) {
    addDebugStage(debugSession, '6. OCR input candidate', winningDecodeCanvas);
  }
  const winningDecodeCells = resolveWinningDecodeCells(decodeCanvasBySource, finalSelection);
  const cellDebugCanvas = buildCellDebugCanvas(winningDecodeCells);
  if (cellDebugCanvas) {
    addDebugStage(debugSession, '7. classifier cell crops', cellDebugCanvas);
  }
  if (stripReaderDebug && stripReaderDebug.canvas) {
    const stripReaderCanvas = buildStripReaderDebugCanvas(stripReaderDebug.canvas, stripReaderDebug.result);
    if (stripReaderCanvas) {
      addDebugStage(debugSession, '8. strip reader input', stripReaderCanvas);
    }
  }
  if (winningDecodeCanvas) {
    return;
  }
  const selectedCandidate = candidates.find((candidate) => (
    candidate
    && candidate.label
    && candidate.label === finalSelection.sourceLabel
    && candidate.canvas
  ));
  if (!selectedCandidate) {
    return;
  }
  const mode = typeof finalSelection.preprocessMode === 'string' ? finalSelection.preprocessMode : 'raw';
  const preview = mode !== 'raw'
    ? preprocessCanvas(selectedCandidate.canvas, mode)
    : selectedCandidate.canvas;
  addDebugStage(debugSession, '6. OCR input candidate', preview);
};

const isPreferredLengthReading = (reading) => {
  return !!(reading && reading.value && reading.value.length === OCR_CONFIG.preferredDigits);
};

const summarizeRejectMap = (rejectMap) => {
  return [...rejectMap.values()]
    .sort((a, b) => b.count - a.count)
    .map((entry) => ({
      reason: entry.reason,
      count: entry.count,
      samples: entry.samples
    }));
};

const prependRejectSummary = (rejectSummary, reason, detail = {}) => {
  if (!Array.isArray(rejectSummary) || !reason) {
    return;
  }
  const existingIndex = rejectSummary.findIndex((entry) => entry && entry.reason === reason);
  if (existingIndex >= 0) {
    const existing = rejectSummary[existingIndex];
    existing.count = Number.isFinite(existing.count) ? (existing.count + 1) : 1;
    if (!Array.isArray(existing.samples)) {
      existing.samples = [];
    }
    if (existing.samples.length < 3) {
      existing.samples.push(detail);
    }
    if (existingIndex > 0) {
      rejectSummary.splice(existingIndex, 1);
      rejectSummary.unshift(existing);
    }
    return;
  }
  rejectSummary.unshift({
    reason,
    count: 1,
    samples: [detail]
  });
};

const evaluateCandidateBranch = async ({
  candidates,
  setProgress,
  scanCanvas = null
}) => {
  const activeCandidates = Array.isArray(candidates)
    ? candidates.filter((candidate) => !!(candidate && candidate.canvas))
    : [];
  if (!activeCandidates.length && scanCanvas) {
    activeCandidates.push({ canvas: scanCanvas, label: 'raw-fallback-roi' });
  }
  if (scanCanvas) {
    activeCandidates.push({ canvas: scanCanvas, label: 'scan-roi' });
  }

  let bestResult = null;
  const valueEvidence = new Map();
  const decodeCanvasBySource = new Map();
  const candidateTrace = [];
  const roiDeterministic = OCR_CONFIG.roiDeterministic || {};
  const classifierConfig = OCR_CONFIG.digitClassifier || {};
  const stripReaderConfig = OCR_CONFIG.digitStripReader || {};
  const branchLabel = 'roi';
  const geometryConfig = OCR_CONFIG.geometry || {};
  const minCandidateWidth = Number.isFinite(geometryConfig.minCandidateWidth) ? geometryConfig.minCandidateWidth : 120;
  const minCandidateHeight = Number.isFinite(geometryConfig.minCandidateHeight) ? geometryConfig.minCandidateHeight : 28;
  const minCandidateAspect = Number.isFinite(geometryConfig.minCandidateAspect) ? geometryConfig.minCandidateAspect : 0.12;
  const maxCandidateAspect = Number.isFinite(geometryConfig.maxCandidateAspect) ? geometryConfig.maxCandidateAspect : 18;
  const rejectMap = new Map();
  let bestStripReaderRawConfidence = -1;
  let stripReaderConfidenceBestDebug = null;
  const stripReaderCandidates = [];
  const stripReaderDebugBySource = new Map();

  const recordReject = (reason, detail = {}) => {
    const key = reason || 'unknown';
    const existing = rejectMap.get(key) || {
      reason: key,
      count: 0,
      samples: []
    };
    existing.count += 1;
    if (existing.samples.length < 3) {
      existing.samples.push(detail);
    }
    rejectMap.set(key, existing);
  };

  const hasValidCandidateGeometry = (candidate, stage) => {
    if (!candidate || !candidate.canvas) {
      recordReject('candidate-missing', {
        stage,
        sourceLabel: candidate && candidate.label ? candidate.label : null
      });
      return false;
    }
    const width = candidate.canvas.width;
    const height = candidate.canvas.height;
    const aspect = width / Math.max(1, height);
    if (width < minCandidateWidth || height < minCandidateHeight) {
      recordReject('candidate-too-small', {
        stage,
        sourceLabel: candidate.label,
        width,
        height
      });
      return false;
    }
    if (aspect < minCandidateAspect || aspect > maxCandidateAspect) {
      recordReject('candidate-bad-aspect', {
        stage,
        sourceLabel: candidate.label,
        width,
        height,
        aspect: Number(aspect.toFixed(3))
      });
      return false;
    }
    return true;
  };

  const applyReadingMetadata = (reading, candidate, method) => {
    if (!reading) {
      return null;
    }
    const candidateAngle = candidate && candidate.label ? extractCandidateAngle(candidate.label) : Number.NaN;
    const orientationOffset = Number.isFinite(reading.orientation) ? normalizeAngle(reading.orientation) : 0;
    const resolvedAngle = Number.isFinite(candidateAngle)
      ? normalizeAngle(candidateAngle + orientationOffset)
      : null;
    return {
      ...reading,
      branch: branchLabel,
      method,
      sourceLabel: candidate && candidate.label ? candidate.label : null,
      angle: Number.isFinite(resolvedAngle) ? resolvedAngle : null
    };
  };

  const recordCandidateReadings = (reading, sourceLabel) => {
    if (!reading) {
      return;
    }
    if (Array.isArray(reading.topCandidates) && reading.topCandidates.length) {
      reading.topCandidates.forEach((entry, index) => {
        recordSelectionEvidence(valueEvidence, entry, {
          sourceLabel,
          isTopPick: index === 0
        });
      });
    } else {
      recordSelectionEvidence(valueEvidence, reading, {
        sourceLabel,
        isTopPick: true
      });
    }
  };

  const rankClassifierCandidates = (rawCandidates) => {
    if (!Array.isArray(rawCandidates) || !rawCandidates.length) {
      return [];
    }
    const fallbackTargetAspect = Number.isFinite(classifierConfig.fallbackTargetAspect)
      ? Math.max(0.4, classifierConfig.fallbackTargetAspect)
      : 2.6;
    const minStripAspect = Number.isFinite(roiDeterministic.minStripAspect) ? roiDeterministic.minStripAspect : 1.45;
    const maxStripAspect = Number.isFinite(roiDeterministic.maxStripAspect) ? roiDeterministic.maxStripAspect : 8.2;

    return rawCandidates
      .filter((candidate) => hasValidCandidateGeometry(candidate, 'classifier-primary'))
      .map((candidate) => {
        const width = candidate.canvas.width;
        const height = candidate.canvas.height;
        const aspect = width / Math.max(1, height);
        const isEdge = isEdgeSourceLabel(candidate.label);
        const angle = extractCandidateAngle(candidate.label);
        const inStripRange = aspect >= minStripAspect && aspect <= maxStripAspect;
        const aspectCloseness = 1 - Math.min(1, Math.abs(aspect - fallbackTargetAspect) / fallbackTargetAspect);
        const expectedHeight = width / fallbackTargetAspect;
        const heightCloseness = 1 - Math.min(1, Math.abs(height - expectedHeight) / Math.max(expectedHeight, 1));

        let fallbackScore = 0;
        fallbackScore += inStripRange ? 0.75 : -0.4;
        fallbackScore += aspectCloseness * 0.5;
        fallbackScore += heightCloseness * 0.2;
        fallbackScore += isEdge ? 0.35 : -0.2;
        if (Number.isFinite(angle) && (angle === 90 || angle === 270)) {
          fallbackScore += 0.1;
        }
        if (candidate.label === 'scan-roi') {
          fallbackScore -= 0.05;
        }

        return {
          ...candidate,
          fallbackScore,
          fallbackAspect: aspect,
          fallbackInStripRange: inStripRange
        };
      })
      .sort((a, b) => (
        b.fallbackScore - a.fallbackScore
        || ((isEdgeSourceLabel(b.label) ? 1 : 0) - (isEdgeSourceLabel(a.label) ? 1 : 0))
        || (b.canvas.width - a.canvas.width)
      ));
  };

  const recordDecodeReject = (candidate, detail = {}, stage = 'classifier-primary') => {
    const reason = detail && detail.reason ? detail.reason : 'classifier-reject';
    const payload = {
      stage,
      sourceLabel: candidate && candidate.label ? candidate.label : null
    };
    if (detail && typeof detail === 'object') {
      Object.keys(detail).forEach((key) => {
        if (key !== 'reason') {
          payload[key] = detail[key];
        }
      });
    }
    recordReject(reason, payload);
  };

  const recordDecodeCanvas = (sourceLabel, angle, canvas, cells = null) => {
    if (!sourceLabel || !canvas) {
      return;
    }
    const normalizedAngle = Number.isFinite(angle) ? normalizeAngle(angle) : null;
    const entries = decodeCanvasBySource.get(sourceLabel) || [];
    const existingIndex = entries.findIndex((entry) => entry && entry.angle === normalizedAngle);
    const nextEntry = {
      angle: normalizedAngle,
      canvas,
      cells: Array.isArray(cells) ? cells : null
    };
    if (existingIndex >= 0) {
      entries[existingIndex] = nextEntry;
    } else {
      entries.push(nextEntry);
    }
    decodeCanvasBySource.set(sourceLabel, entries);
  };

  const serializeStripReaderProbe = (probe, candidate, stage) => {
    const sourceLabel = candidate && candidate.label ? candidate.label : null;
    const width = candidate && candidate.canvas ? candidate.canvas.width : null;
    const height = candidate && candidate.canvas ? candidate.canvas.height : null;
    if (!probe || !probe.ok) {
      return {
        ok: false,
        reason: probe && probe.reason ? probe.reason : 'strip-reader-miss',
        stage,
        sourceLabel,
        width,
        height
      };
    }
    const rawConfidence = Number.isFinite(probe.confidence) ? probe.confidence : 0;
    return {
      ok: true,
      stage,
      sourceLabel,
      width,
      height,
      method: 'digit-strip-reader-shadow',
      value: probe.value || null,
      confidence: serializeStripConfidence(rawConfidence),
      rawConfidence: Number(rawConfidence.toFixed(4)),
      digits: Array.isArray(probe.digits) ? [...probe.digits] : null,
      digitConfidences: serializeStripConfidences(probe.digitConfidences),
      topKByPosition: Array.isArray(probe.topKByPosition) ? probe.topKByPosition : [],
      model: probe.model || null,
      device: probe.device || null
    };
  };

  const runStripReaderShadow = async (candidate, stage) => {
    if (!stripReaderConfig.enabled || !stripReaderConfig.endpoint || !candidate || !candidate.canvas) {
      return null;
    }
    const probe = await predictDigitStrip(candidate.canvas, stripReaderConfig);
    const serialized = serializeStripReaderProbe(probe, candidate, stage);
    if (serialized) {
      stripReaderCandidates.push(serialized);
    }
    if (serialized && serialized.ok) {
      const rawConfidence = Number.isFinite(serialized.rawConfidence) ? serialized.rawConfidence : 0;
      const debugEntry = {
        canvas: candidate.canvas,
        result: serialized
      };
      if (serialized.sourceLabel && !stripReaderDebugBySource.has(serialized.sourceLabel)) {
        stripReaderDebugBySource.set(serialized.sourceLabel, debugEntry);
      }
      if (rawConfidence > bestStripReaderRawConfidence) {
        bestStripReaderRawConfidence = rawConfidence;
        stripReaderConfidenceBestDebug = debugEntry;
      }
    }
    return serialized;
  };

  const buildStripReaderTrace = () => ({
    candidates: stripReaderCandidates,
    debugBySource: stripReaderDebugBySource,
    confidenceBestDebug: stripReaderConfidenceBestDebug
  });

  if (!classifierConfig.enabled || !classifierConfig.endpoint) {
    recordReject('classifier-disabled', {
      stage: 'classifier-primary'
    });
    return {
      bestResult: null,
      evidenceMap: valueEvidence,
      rejectSummary: summarizeRejectMap(rejectMap),
      decodeCanvasBySource,
      candidateTrace,
      stripReaderTrace: buildStripReaderTrace()
    };
  }

  const rankedCandidates = rankClassifierCandidates(activeCandidates);
  const maxPrimaryCandidates = Number.isFinite(classifierConfig.maxPrimaryCandidates)
    ? Math.max(1, Math.min(20, Math.round(classifierConfig.maxPrimaryCandidates)))
    : (
      Number.isFinite(OCR_CONFIG.fallbackCandidates)
        ? Math.max(1, Math.min(20, Math.round(OCR_CONFIG.fallbackCandidates)))
        : 6
    );
  const rankedEdgeCandidates = rankedCandidates.filter((candidate) => isEdgeSourceLabel(candidate.label));
  const rankedBaseCandidates = rankedCandidates.filter((candidate) => !isEdgeSourceLabel(candidate.label));
  let selectedCandidates;
  if (rankedEdgeCandidates.length && rankedBaseCandidates.length && maxPrimaryCandidates > 1) {
    const selectedSet = new Set();
    const reserveBaseSlots = Math.min(
      rankedBaseCandidates.length,
      Math.min(2, Math.max(1, maxPrimaryCandidates - 1))
    );
    const primaryCandidates = [
      ...rankedEdgeCandidates.slice(0, Math.max(1, maxPrimaryCandidates - reserveBaseSlots)),
      ...rankedBaseCandidates.slice(0, reserveBaseSlots)
    ];

    primaryCandidates.forEach((candidate) => {
      if (!selectedSet.has(candidate.label)) {
        selectedSet.add(candidate.label);
      }
    });
    selectedCandidates = [...selectedSet].map((label) => (
      rankedCandidates.find((candidate) => candidate.label === label)
    )).filter(Boolean);

    if (selectedCandidates.length < maxPrimaryCandidates) {
      rankedCandidates.forEach((candidate) => {
        if (selectedCandidates.length >= maxPrimaryCandidates || selectedSet.has(candidate.label)) {
          return;
        }
        selectedSet.add(candidate.label);
        selectedCandidates.push(candidate);
      });
    }
  } else {
    selectedCandidates = rankedEdgeCandidates.length
      ? rankedEdgeCandidates.slice(0, maxPrimaryCandidates)
      : rankedBaseCandidates.slice(0, maxPrimaryCandidates);
  }
  if (classifierConfig.forceInitialPreviewCandidate === true) {
    const initialCandidate = activeCandidates.find((candidate) => (
      hasValidCandidateGeometry(candidate, 'classifier-primary-force-initial')
    ));
    selectedCandidates = initialCandidate ? [initialCandidate] : [];
  }
  const primaryIncludesBaseCandidates = selectedCandidates.some((candidate) => !isEdgeSourceLabel(candidate.label));
  if (!selectedCandidates.length) {
    recordReject('classifier-no-candidate', {
      stage: 'classifier-primary'
    });
    return {
      bestResult: null,
      evidenceMap: valueEvidence,
      rejectSummary: summarizeRejectMap(rejectMap),
      decodeCanvasBySource,
      candidateTrace,
      stripReaderTrace: buildStripReaderTrace()
    };
  }

  const runCandidatePass = async (candidates, stageLabel) => {
    const nonEdgeAvailable = candidates.some((candidate) => !isEdgeSourceLabel(candidate.label));
    let pass = 0;
    const expectedPasses = candidates.length;

    for (const candidate of candidates) {
      pass += 1;
      if (setProgress) {
        setProgress(`Classifying digits (${pass}/${expectedPasses})`);
      }

      const stripReaderProbe = await runStripReaderShadow(candidate, stageLabel);

      if (isEdgeSourceLabel(candidate.label)) {
        const fallbackScore = Number.isFinite(candidate.fallbackScore)
          ? Number(candidate.fallbackScore.toFixed(3))
          : null;
        const fallbackAspect = Number.isFinite(candidate.fallbackAspect)
          ? Number(candidate.fallbackAspect.toFixed(3))
          : (
            candidate.canvas
              ? Number((candidate.canvas.width / Math.max(1, candidate.canvas.height)).toFixed(3))
              : null
          );
        recordReject('classifier-edge-candidate-selected', {
          stage: stageLabel,
          sourceLabel: candidate.label,
          score: fallbackScore,
          aspect: fallbackAspect,
          nonEdgeAlternative: nonEdgeAvailable
        });
      }

      const candidateRejects = [];
      const reading = await readDigitsByCells(candidate.canvas, null, {
        roiMode: true,
        onReject: (detail) => {
          candidateRejects.push(detail);
          recordDecodeReject(candidate, detail, stageLabel);
        }
      });
      if (!reading) {
        candidateTrace.push({
          stage: stageLabel,
          sourceLabel: candidate.label,
          width: candidate.canvas.width,
          height: candidate.canvas.height,
          fallbackScore: Number.isFinite(candidate.fallbackScore)
            ? Number(candidate.fallbackScore.toFixed(3))
            : null,
          fallbackAspect: Number.isFinite(candidate.fallbackAspect)
            ? Number(candidate.fallbackAspect.toFixed(3))
            : null,
          stripReader: stripReaderProbe,
          result: null,
          rejects: candidateRejects
        });
        if (candidateRejects.some((detail) => isClassifierAvailabilityReject(detail))) {
          break;
        }
        continue;
      }
      if (!isPreferredLengthReading(reading)) {
        candidateTrace.push({
          stage: stageLabel,
          sourceLabel: candidate.label,
          width: candidate.canvas.width,
          height: candidate.canvas.height,
          fallbackScore: Number.isFinite(candidate.fallbackScore)
            ? Number(candidate.fallbackScore.toFixed(3))
            : null,
          fallbackAspect: Number.isFinite(candidate.fallbackAspect)
            ? Number(candidate.fallbackAspect.toFixed(3))
            : null,
          stripReader: stripReaderProbe,
          result: {
            value: reading.value || null,
            confidence: Number.isFinite(reading.confidence) ? Number(reading.confidence.toFixed(1)) : null,
            score: Number.isFinite(reading.score) ? Number(reading.score.toFixed(3)) : null,
            cellDigits: Array.isArray(reading.cellDigits) ? [...reading.cellDigits] : null,
            cellConfidences: serializeCellConfidences(reading.cellConfidences)
          },
          rejects: candidateRejects
        });
        recordReject('classifier-non4-reading', {
          stage: stageLabel,
          sourceLabel: candidate.label,
          value: reading.value || null
        });
        continue;
      }

      const classifierReading = applyReadingMetadata({
        ...reading,
        preprocessMode: 'raw'
      }, candidate, 'digit-classifier-primary');
      if (!classifierReading) {
        continue;
      }
      if (
        !isEdgeSourceLabel(classifierReading.sourceLabel)
        && String(classifierReading.method || '').startsWith('digit-classifier')
        && isRepeatedDigitReading(classifierReading.value)
      ) {
        classifierReading.score = Math.max(0, (classifierReading.score ?? 0) - 0.22);
        recordReject('classifier-repeated-digit-penalty', {
          stage: stageLabel,
          sourceLabel: classifierReading.sourceLabel,
          value: classifierReading.value,
          penalty: 0.22
        });
      }
      if (classifierReading.sourceLabel && classifierReading.decodedStripCanvas) {
        recordDecodeCanvas(
          classifierReading.sourceLabel,
          classifierReading.angle,
          classifierReading.decodedStripCanvas,
          classifierReading.decodedCellCanvases
        );
      }
      const classifierReadingForSelection = { ...classifierReading };
      delete classifierReadingForSelection.decodedStripCanvas;
      delete classifierReadingForSelection.decodedCellCanvases;
      candidateTrace.push({
        stage: stageLabel,
        sourceLabel: candidate.label,
        width: candidate.canvas.width,
        height: candidate.canvas.height,
        fallbackScore: Number.isFinite(candidate.fallbackScore)
          ? Number(candidate.fallbackScore.toFixed(3))
          : null,
        fallbackAspect: Number.isFinite(candidate.fallbackAspect)
          ? Number(candidate.fallbackAspect.toFixed(3))
          : null,
        result: {
          value: classifierReadingForSelection.value || null,
          confidence: Number.isFinite(classifierReadingForSelection.confidence)
            ? Number(classifierReadingForSelection.confidence.toFixed(1))
            : null,
          score: Number.isFinite(classifierReadingForSelection.score)
            ? Number(classifierReadingForSelection.score.toFixed(3))
            : null,
          angle: Number.isFinite(classifierReadingForSelection.angle)
            ? classifierReadingForSelection.angle
            : null,
          method: classifierReadingForSelection.method || null,
          cellDigits: Array.isArray(classifierReadingForSelection.cellDigits)
            ? [...classifierReadingForSelection.cellDigits]
            : null,
          cellConfidences: serializeCellConfidences(classifierReadingForSelection.cellConfidences)
        },
        stripReader: stripReaderProbe,
        rejects: candidateRejects
      });

      const rankedEvidenceBeforeCurrentReading = rankSelectionEvidence(valueEvidence);
      const preserveAgreedEdgeResult = !!(
        stageLabel === 'classifier-fallback-base'
        && bestResult
        && isEdgeSourceLabel(bestResult.sourceLabel)
        && rankedEvidenceBeforeCurrentReading.some((entry) => (
          entry.value === bestResult.value
          && entry.topHits >= 2
          && entry.edgeTopHits >= 2
          && entry.nonEdgeHits === 0
        ))
      );
      if (
        !bestResult
        || (
          classifierReadingForSelection.score > bestResult.score
          && (
            !preserveAgreedEdgeResult
            || classifierReadingForSelection.score > (bestResult.score + 0.03)
          )
        )
        || (
          classifierReadingForSelection.score === bestResult.score
          && (classifierReadingForSelection.confidence ?? 0) > (bestResult.confidence ?? 0)
        )
      ) {
        bestResult = classifierReadingForSelection;
      }
      recordCandidateReadings(classifierReadingForSelection, `${candidate.label}:classifier`);

    }
    return false;
  };

  const edgeWonEarly = await runCandidatePass(selectedCandidates, 'classifier-primary');
  const rankedEvidenceAfterPrimary = rankSelectionEvidence(valueEvidence);
  const primaryTopEvidence = rankedEvidenceAfterPrimary[0] || null;
  const shouldReopenBaseFallback = !!(
    rankedEdgeCandidates.length > 0
    && primaryTopEvidence
    && primaryTopEvidence.nonEdgeSourceCount === 0
    && primaryTopEvidence.topHits < 2
  );
  if (
    !edgeWonEarly
    && rankedBaseCandidates.length
    && !primaryIncludesBaseCandidates
    && (
      !bestResult
      || shouldReopenBaseFallback
    )
  ) {
    selectedCandidates = rankedBaseCandidates.slice(0, Math.min(2, maxPrimaryCandidates));
    await runCandidatePass(selectedCandidates, 'classifier-fallback-base');
  }

  return {
    bestResult,
    evidenceMap: valueEvidence,
    rejectSummary: summarizeRejectMap(rejectMap),
    decodeCanvasBySource,
    candidateTrace,
    stripReaderTrace: buildStripReaderTrace()
  };
};

const runMeterOcr = async (file, setProgress) => {
  const image = await loadImageBitmap(file);
  const baseCanvas = drawImageToCanvas(image, OCR_CONFIG.maxDimension);
  const debugLabel = file && file.name ? file.name : `manual-${Date.now()}`;
  const debugSession = startDebugSession(debugLabel);

  try {
    const neuralRoiConfig = OCR_CONFIG.neuralRoi || {};
    if (!neuralRoiConfig.enabled || !neuralRoiConfig.endpoint) {
      const message = 'Neural ROI is disabled or misconfigured. Enter the measurement manually.';
      if (setProgress) {
        setProgress(message);
      }
      throw new Error(message);
    }

    if (setProgress) {
      setProgress('Requesting neural ROI...');
    }
    const roiProbe = await detectNeuralRoi(file, neuralRoiConfig);
    if (!roiProbe.ok) {
      addNeuralRoiMissStage(debugSession, baseCanvas, roiProbe);
      const reason = formatNeuralRoiMissReason(roiProbe);
      const message = `Neural ROI failed (${reason}). Enter the measurement manually.`;
      if (setProgress) {
        setProgress(message);
      }
      throw new Error(message);
    }

    const roiRect = resolveNeuralRoiRect(baseCanvas, roiProbe, neuralRoiConfig);
    addNeuralRoiDebugStages(debugSession, baseCanvas, roiRect, roiProbe);
    const roiCrop = cropCanvas(baseCanvas, roiRect);
    addDebugStage(debugSession, '0b. neural roi crop', roiCrop);

    const roiCandidates = roiCrop
      ? buildDigitCandidates(roiCrop, debugSession, addDebugStage).map((candidate) => ({
        ...candidate,
        label: `${candidate.label}-roi`
      }))
      : [];

    if (!roiCandidates.length) {
      const message = 'Neural ROI crop did not produce OCR candidates. Enter the measurement manually.';
      if (setProgress) {
        setProgress(message);
      }
      throw new Error(message);
    }

    const roiBranch = await evaluateCandidateBranch({
      candidates: roiCandidates,
      setProgress,
      scanCanvas: roiCrop
    });

    const finalSelection = finalizeSelection({
      debugLabel,
      roiUsed: true,
      bestResult: roiBranch.bestResult,
      evidenceMap: roiBranch.evidenceMap,
      branchUsed: isPreferredLengthReading(roiBranch.bestResult) ? 'roi-accepted' : 'roi-uncertain',
      rejectSummary: roiBranch.rejectSummary || [],
      candidateTrace: roiBranch.candidateTrace || [],
      stripReaderTrace: roiBranch.stripReaderTrace || null
    });
    const stripReaderDebug = resolveStripReaderDebug(roiBranch.stripReaderTrace || null, finalSelection);
    addWinningCandidateDebugStage(
      debugSession,
      roiCandidates,
      finalSelection,
      roiBranch.decodeCanvasBySource,
      stripReaderDebug
    );

    if (setProgress) {
      if (finalSelection && finalSelection.value) {
        setProgress(`Neural ROI + OCR complete (${finalSelection.value}).`);
      } else {
        setProgress('Neural ROI found, OCR uncertain. Enter the measurement manually.');
      }
    }

    return finalSelection;
  } finally {
    commitDebugSession(debugSession);
  }
};

export { runMeterOcr, setOcrDebugHooks };
