import { OCR_CONFIG } from './config.js';
import { setOcrDebugHooks, startDebugSession, addDebugStage, commitDebugSession } from './debug-hooks.js';
import {
  loadImageBitmap,
  drawImageToCanvas,
  preprocessCanvas,
  scaleCanvas,
  cropCanvas,
  splitIntoCells,
  normalizeRectToCanvas,
  drawOverlayCanvas,
  normalizeAngle
} from './canvas-utils.js';
import { buildDigitCandidates } from './alignment.js';
import { getWorker, selectBestReading, readDigitsByCells } from './recognition.js';
import { predictDigitCells } from './digit-classifier.js';
import { detectNeuralRoi } from './neural-roi.js';

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

const finalizeSelection = ({ debugLabel, roiUsed, bestResult, evidenceMap, branchUsed, rejectSummary = [] }) => {
  const rankedEvidence = rankSelectionEvidence(evidenceMap);
  const evidenceBest = rankedEvidence[0] || null;
  const roiDeterministic = OCR_CONFIG.roiDeterministic || {};
  const classifierConfig = OCR_CONFIG.digitClassifier || {};
  const minWordPassHits = Number.isFinite(roiDeterministic.minWordPassHits)
    ? Math.max(1, Math.round(roiDeterministic.minWordPassHits))
    : 2;
  let finalResult = bestResult;
  let finalRejectReason = null;
  let finalRejectDetail = null;

  if (evidenceBest) {
    const shouldPromoteEvidence = (
      !finalResult
      || evidenceBest.score >= (finalResult.score ?? -1) - 0.03
      || evidenceBest.topHits >= 2
    );

    if (shouldPromoteEvidence) {
      const confidenceFromBest = finalResult && finalResult.value === evidenceBest.value
        ? (finalResult.confidence ?? 0)
        : 0;
      const carryMetadata = finalResult && finalResult.value === evidenceBest.value
        ? finalResult
        : null;
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

  if (finalResult && finalResult.method === 'word-pass') {
    const support = rankedEvidence.find((entry) => entry.value === finalResult.value) || null;
    const confirmed = !!support && (
      support.hits >= minWordPassHits
      || support.topHits >= minWordPassHits
    );
    if (!confirmed) {
      finalRejectReason = 'word-pass-unconfirmed-finalize';
      finalRejectDetail = {
        stage: 'selection-finalize',
        method: finalResult.method || null,
        sourceLabel: finalResult.sourceLabel || null,
        value: finalResult.value || null,
        requiredHits: minWordPassHits,
        supportHits: support ? support.hits : 0,
        supportTopHits: support ? support.topHits : 0
      };
      finalResult = null;
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
    const isClassifierFallback = String(finalResult.method || '').startsWith('digit-classifier-fallback');

    if (isClassifierFallback) {
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
    topCandidates: buildSelectionSummary(rankedEvidence, 3)
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

const addWinningCandidateDebugStage = (debugSession, candidates, finalSelection) => {
  if (!debugSession || !Array.isArray(candidates) || !finalSelection || !finalSelection.sourceLabel) {
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
  worker,
  setProgress,
  useWordPass = true,
  allowSparseScan = false,
  scanCanvas = null
}) => {
  const activeCandidates = Array.isArray(candidates) && candidates.length
    ? candidates
    : [{ canvas: scanCanvas, label: 'raw-fallback-roi' }];
  let bestResult = null;
  const valueEvidence = new Map();
  const roiDeterministic = OCR_CONFIG.roiDeterministic || {};
  const configuredModes = Array.isArray(roiDeterministic.wordPassModes)
    ? roiDeterministic.wordPassModes
    : [];
  const modes = (configuredModes.length ? configuredModes : ['raw'])
    .filter((mode) => mode === 'soft' || mode === 'binary' || mode === 'raw');
  if (!modes.length) {
    modes.push('raw');
  }
  let pass = 0;
  const expectedPasses = Math.max(1, activeCandidates.length * modes.length);
  const branchLabel = 'roi';
  const geometryConfig = OCR_CONFIG.geometry || {};
  const minCandidateWidth = Number.isFinite(geometryConfig.minCandidateWidth) ? geometryConfig.minCandidateWidth : 120;
  const minCandidateHeight = Number.isFinite(geometryConfig.minCandidateHeight) ? geometryConfig.minCandidateHeight : 28;
  const minCandidateAspect = Number.isFinite(geometryConfig.minCandidateAspect) ? geometryConfig.minCandidateAspect : 0.12;
  const maxCandidateAspect = Number.isFinite(geometryConfig.maxCandidateAspect) ? geometryConfig.maxCandidateAspect : 18;
  const minCellWidth = Number.isFinite(geometryConfig.minCellWidth) ? geometryConfig.minCellWidth : 20;
  const minCellHeight = Number.isFinite(geometryConfig.minCellHeight) ? geometryConfig.minCellHeight : 24;
  const rejectMap = new Map();

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

  const hasValidCellGeometry = (cellCanvases, stage, extra = {}) => {
    for (let i = 0; i < cellCanvases.length; i += 1) {
      const cell = cellCanvases[i];
      if (!cell || cell.width < minCellWidth || cell.height < minCellHeight) {
        recordReject('cell-too-small', {
          stage,
          index: i,
          sourceLabel: extra.sourceLabel || null,
          width: cell ? cell.width : 0,
          height: cell ? cell.height : 0
        });
        return false;
      }
    }
    return true;
  };

  const pickSingleDigit = (data) => {
    const symbolDigits = (data && Array.isArray(data.symbols) ? data.symbols : [])
      .map((item) => ({
        digit: (item && item.text ? String(item.text) : '').replace(/\D/g, '').slice(0, 1),
        confidence: Number.isFinite(item && item.confidence) ? item.confidence : (Number.isFinite(data && data.confidence) ? data.confidence : 0)
      }))
      .filter((item) => item.digit);
    const bestSymbol = symbolDigits.sort((a, b) => b.confidence - a.confidence)[0];
    if (bestSymbol) {
      return bestSymbol;
    }
    const textDigits = (data && data.text ? String(data.text) : '').replace(/\D/g, '');
    if (textDigits) {
      return {
        digit: textDigits[0],
        confidence: Number.isFinite(data && data.confidence) ? data.confidence : 0
      };
    }
    return null;
  };

  const refineSingleCellWithTesseract = async ({ cellCanvases, index, sourceLabel }) => {
    if (!Array.isArray(cellCanvases) || !cellCanvases[index]) {
      return null;
    }
    const modes = ['binary', 'soft', 'raw'];
    let best = null;
    for (const mode of modes) {
      const processed = mode === 'raw'
        ? cellCanvases[index]
        : preprocessCanvas(cellCanvases[index], mode);
      const scaled = scaleCanvas(processed, OCR_CONFIG.minDigitWidth);
      const { data } = await worker.recognize(scaled);
      const picked = pickSingleDigit(data);
      if (!picked) {
        continue;
      }
      const confidence = Number.isFinite(picked.confidence)
        ? clamp(picked.confidence, 0, 100)
        : 0;
      if (!best || confidence > best.confidence) {
        best = {
          digit: picked.digit,
          confidence,
          mode
        };
      }
    }
    if (!best) {
      recordReject('classifier-single-cell-refine-no-digit', {
        stage: 'classifier-fallback',
        sourceLabel,
        index
      });
      return null;
    }
    return best;
  };

  const verifyEdgeWordPassCandidate = async ({ candidate, mode, reading }) => {
    if (!reading || !candidate || !candidate.label || mode !== 'raw' || !candidate.label.includes('-edge')) {
      return {
        reading,
        method: 'word-pass'
      };
    }

    const verified = await readDigitsByCells(worker, candidate.canvas, null, {
      roiMode: true,
      onReject: (detail) => {
        recordReject('edge-word-pass-cell-reject', {
          stage: 'word-pass',
          sourceLabel: candidate.label,
          mode,
          reason: detail && detail.reason ? detail.reason : null
        });
      }
    });

    if (!verified || !isPreferredLengthReading(verified)) {
      recordReject('edge-word-pass-unverified', {
        stage: 'word-pass',
        sourceLabel: candidate.label,
        mode,
        value: reading.value || null
      });
      return {
        reading: null,
        method: 'word-pass'
      };
    }

    if (verified.value !== reading.value) {
      const wordDigits = String(reading.value || '').replace(/\D/g, '');
      const cellDigits = String(verified.value || '').replace(/\D/g, '');
      const wordUniqueDigits = new Set(wordDigits.split('').filter(Boolean)).size;
      const cellUniqueDigits = new Set(cellDigits.split('').filter(Boolean)).size;
      if (cellUniqueDigits === 1 && wordUniqueDigits > 1) {
        recordReject('edge-word-pass-cell-collapse', {
          stage: 'word-pass',
          sourceLabel: candidate.label,
          mode,
          wordPassValue: reading.value || null,
          cellValue: verified.value || null
        });
        return {
          reading: {
            ...reading,
            preprocessMode: mode
          },
          method: 'word-pass'
        };
      }
      recordReject('edge-word-pass-cell-mismatch', {
        stage: 'word-pass',
        sourceLabel: candidate.label,
        mode,
        wordPassValue: reading.value || null,
        cellValue: verified.value || null
      });
      return {
        reading: {
          ...verified,
          preprocessMode: mode
        },
        method: 'cell-verify'
      };
    }

    return {
      reading: {
        ...reading,
        preprocessMode: mode,
        cellDigits: Array.isArray(verified.cellDigits) ? verified.cellDigits : (reading.cellDigits || null),
        cellConfidences: Array.isArray(verified.cellConfidences) ? verified.cellConfidences : (reading.cellConfidences || null)
      },
      method: 'word-pass'
    };
  };

  const rankClassifierFallbackCandidates = (rawCandidates) => {
    if (!Array.isArray(rawCandidates) || !rawCandidates.length) {
      return [];
    }
    const classifierConfig = OCR_CONFIG.digitClassifier || {};
    const fallbackPreferNonEdge = classifierConfig.fallbackPreferNonEdge !== false;
    const fallbackTargetAspect = Number.isFinite(classifierConfig.fallbackTargetAspect)
      ? Math.max(0.4, classifierConfig.fallbackTargetAspect)
      : 2.6;
    const minStripAspect = Number.isFinite(roiDeterministic.minStripAspect) ? roiDeterministic.minStripAspect : 1.45;
    const maxStripAspect = Number.isFinite(roiDeterministic.maxStripAspect) ? roiDeterministic.maxStripAspect : 8.2;

    return rawCandidates
      .filter((candidate) => hasValidCandidateGeometry(candidate, 'classifier-fallback'))
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
        if (fallbackPreferNonEdge) {
          fallbackScore += isEdge ? -0.35 : 0.35;
        }
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
        || ((isEdgeSourceLabel(a.label) ? 1 : 0) - (isEdgeSourceLabel(b.label) ? 1 : 0))
        || (b.canvas.width - a.canvas.width)
      ));
  };

  const tryClassifierFallback = async () => {
    if (bestResult) {
      return null;
    }

    const classifierConfig = OCR_CONFIG.digitClassifier || {};
    if (!classifierConfig.enabled) {
      return null;
    }

    const fallbackOnNoDigitsOnly = classifierConfig.fallbackOnNoDigitsOnly !== false;
    const noDigitsRejects = rejectMap.get('ocr-no-digits');
    if (fallbackOnNoDigitsOnly && !(noDigitsRejects && noDigitsRejects.count > 0)) {
      return null;
    }

    const fallbackCandidates = [...activeCandidates];
    if (scanCanvas) {
      fallbackCandidates.push({ canvas: scanCanvas, label: 'scan-roi' });
    }

    const rankedFallbackCandidates = rankClassifierFallbackCandidates(fallbackCandidates);
    const fallbackCandidate = rankedFallbackCandidates[0] || null;
    if (!fallbackCandidate) {
      recordReject('classifier-no-candidate', {
        stage: 'classifier-fallback'
      });
      return null;
    }
    if (isEdgeSourceLabel(fallbackCandidate.label)) {
      recordReject('classifier-fallback-edge-selected', {
        stage: 'classifier-fallback',
        sourceLabel: fallbackCandidate.label,
        score: Number(fallbackCandidate.fallbackScore.toFixed(3)),
        aspect: Number(fallbackCandidate.fallbackAspect.toFixed(3)),
        nonEdgeAlternative: rankedFallbackCandidates.some((candidate) => !isEdgeSourceLabel(candidate.label))
      });
    }

    const overlap = Number.isFinite(roiDeterministic.cellOverlap) ? roiDeterministic.cellOverlap : 0.03;
    const cellCanvases = splitIntoCells(fallbackCandidate.canvas, OCR_CONFIG.digitCellCount, overlap);
    if (!hasValidCellGeometry(cellCanvases, 'classifier-fallback', { sourceLabel: fallbackCandidate.label })) {
      return null;
    }

    if (setProgress) {
      setProgress('Fallback: classifier check...');
    }
    const classifierProbe = await predictDigitCells(cellCanvases, classifierConfig);
    if (!classifierProbe.ok) {
      if (classifierProbe.reason !== 'disabled') {
        recordReject('classifier-unavailable', {
          stage: 'classifier-fallback',
          sourceLabel: fallbackCandidate.label,
          reason: classifierProbe.reason
        });
      }
      return null;
    }

    const minAcceptedCells = Number.isFinite(classifierConfig.fallbackMinAcceptedCells)
      ? Math.max(1, Math.min(OCR_CONFIG.digitCellCount, Math.round(classifierConfig.fallbackMinAcceptedCells)))
      : OCR_CONFIG.digitCellCount;
    const digits = classifierProbe.predictions.map((item) => (item && item.accepted ? item.digit : ''));
    const rawCellConfidences = classifierProbe.predictions.map((item) => {
      if (!item || !Number.isFinite(item.confidence)) {
        return 0;
      }
      return clamp(item.confidence, 0, 1);
    });
    const cellConfidences = classifierProbe.predictions.map((item) => {
      if (!item || !item.accepted || !Number.isFinite(item.confidence)) {
        return 0;
      }
      return clamp(item.confidence * 100, 0, 100);
    });
    let acceptedCount = digits.filter(Boolean).length;

    const singleCellRefineEnabled = classifierConfig.singleCellRefine !== false;
    const lowConfidenceThresholdRaw = Number.isFinite(classifierConfig.singleCellLowConfidence)
      ? classifierConfig.singleCellLowConfidence
      : (
        Number.isFinite(classifierConfig.minCellConfidence)
          ? classifierConfig.minCellConfidence + 0.2
          : 0.55
      );
    const lowConfidenceThreshold = clamp(lowConfidenceThresholdRaw, 0, 1);
    const singleCellRefineMinConfidence = Number.isFinite(classifierConfig.singleCellRefineMinConfidence)
      ? clamp(classifierConfig.singleCellRefineMinConfidence, 0, 100)
      : 42;
    const singleCellRefineSwitchMargin = Number.isFinite(classifierConfig.singleCellRefineSwitchMargin)
      ? Math.max(0, classifierConfig.singleCellRefineSwitchMargin)
      : 4;
    const lowConfidenceIndices = classifierProbe.predictions
      .map((item, index) => {
        const accepted = !!(item && item.accepted && item.digit);
        const confidence = rawCellConfidences[index];
        return (!accepted || confidence < lowConfidenceThreshold) ? index : -1;
      })
      .filter((index) => index >= 0);
    let refinedCellIndex = null;

    if (
      singleCellRefineEnabled
      && lowConfidenceIndices.length === 1
      && acceptedCount >= Math.max(0, minAcceptedCells - 1)
    ) {
      const targetIndex = lowConfidenceIndices[0];
      if (setProgress) {
        setProgress('Fallback: refining one low-confidence section...');
      }
      const refined = await refineSingleCellWithTesseract({
        cellCanvases,
        index: targetIndex,
        sourceLabel: fallbackCandidate.label
      });
      if (refined && refined.digit) {
        const previousDigit = digits[targetIndex] || '';
        const previousConfidence = cellConfidences[targetIndex] || 0;
        const shouldApply = !previousDigit
          ? refined.confidence >= singleCellRefineMinConfidence
          : (
            refined.digit === previousDigit
            || refined.confidence >= (previousConfidence + singleCellRefineSwitchMargin)
          );
        if (shouldApply) {
          digits[targetIndex] = refined.digit;
          cellConfidences[targetIndex] = Math.max(previousConfidence, refined.confidence);
          acceptedCount = digits.filter(Boolean).length;
          refinedCellIndex = targetIndex;
        } else {
          recordReject('classifier-single-cell-refine-skipped', {
            stage: 'classifier-fallback',
            sourceLabel: fallbackCandidate.label,
            index: targetIndex,
            classifierDigit: previousDigit || null,
            classifierConfidence: Number(previousConfidence.toFixed(1)),
            refinedDigit: refined.digit,
            refinedConfidence: Number(refined.confidence.toFixed(1))
          });
        }
      }
    }

    if (acceptedCount < minAcceptedCells) {
      recordReject('classifier-insufficient-cell-digits', {
        stage: 'classifier-fallback',
        sourceLabel: fallbackCandidate.label,
        accepted: acceptedCount,
        required: minAcceptedCells
      });
      return null;
    }

    const value = digits.join('');
    if (!value || value.length !== OCR_CONFIG.preferredDigits) {
      recordReject('classifier-non4-reading', {
        stage: 'classifier-fallback',
        sourceLabel: fallbackCandidate.label,
        value
      });
      return null;
    }

    const confidenceSum = cellConfidences.reduce((sum, score) => sum + score, 0);
    const averageConfidence = confidenceSum / Math.max(acceptedCount, 1);
    const normalizedConfidence = clamp(averageConfidence / 100, 0, 1);
    const score = clamp(0.42 + normalizedConfidence * 0.46, 0, 0.94);

    const fallbackReading = applyReadingMetadata({
      value,
      confidence: averageConfidence,
      areaRatio: 0.28,
      score,
      decoder: refinedCellIndex === null ? 'digit-classifier' : 'digit-classifier-single-cell-refine',
      cellDigits: digits,
      cellConfidences,
      refinedCellIndex
    }, fallbackCandidate, 'digit-classifier-fallback');
    if (!fallbackReading) {
      return null;
    }
    recordCandidateReadings(fallbackReading, `${fallbackCandidate.label}:classifier`);
    return fallbackReading;
  };

  if (useWordPass) {
    await worker.setParameters({
      tessedit_pageseg_mode: Tesseract.PSM.SINGLE_WORD,
      tessedit_char_whitelist: '0123456789',
      classify_bln_numeric_mode: 1
    });
    for (const candidate of activeCandidates) {
      if (!hasValidCandidateGeometry(candidate, 'word-pass')) {
        continue;
      }
      for (const mode of modes) {
        pass += 1;
        if (setProgress) {
          setProgress(`Analyzing meter (${pass}/${expectedPasses})`);
        }
        const processed = mode === 'raw'
          ? candidate.canvas
          : preprocessCanvas(candidate.canvas, mode);
        const { data } = await worker.recognize(processed);
        const candidateRawBest = selectBestReading(data, processed);
        let candidateBest = candidateRawBest;
        let candidateMethod = 'word-pass';
        if (!candidateRawBest) {
          recordReject('ocr-no-digits', {
            stage: 'word-pass',
            sourceLabel: candidate.label,
            mode
          });
        } else if (!isPreferredLengthReading(candidateRawBest)) {
          recordReject('ocr-non4-reading', {
            stage: 'word-pass',
            sourceLabel: candidate.label,
            mode,
            value: candidateRawBest.value || null
          });
          candidateBest = null;
        }
        if (candidateBest) {
          const verified = await verifyEdgeWordPassCandidate({
            candidate,
            mode,
            reading: {
              ...candidateBest,
              preprocessMode: mode
            }
          });
          candidateBest = verified.reading;
          candidateMethod = verified.method;
        }
        candidateBest = applyReadingMetadata(candidateBest, candidate, candidateMethod);
        if (candidateBest && (!bestResult || candidateBest.score > bestResult.score)) {
          bestResult = candidateBest;
        }
        if (candidateBest) {
          recordCandidateReadings(candidateBest, candidate.label);
        }
      }
    }
  }

  if (!bestResult && allowSparseScan && scanCanvas) {
    if (setProgress) {
      setProgress('Scanning full image...');
    }
    await worker.setParameters({
      tessedit_pageseg_mode: Tesseract.PSM.SPARSE_TEXT,
      tessedit_char_whitelist: '0123456789'
    });
    const softened = preprocessCanvas(scanCanvas, 'soft');
    const { data } = await worker.recognize(softened);
    const sparseRawCandidate = selectBestReading(data, softened);
    let fullCandidate = sparseRawCandidate;
    if (!sparseRawCandidate) {
      recordReject('ocr-no-digits', {
        stage: 'sparse-scan',
        sourceLabel: 'scan-roi',
        mode: 'soft'
      });
    } else if (!isPreferredLengthReading(sparseRawCandidate)) {
      recordReject('ocr-non4-reading', {
        stage: 'sparse-scan',
        sourceLabel: 'scan-roi',
        mode: 'soft',
        value: sparseRawCandidate.value || null
      });
      fullCandidate = null;
    }
    fullCandidate = applyReadingMetadata(fullCandidate, { label: 'scan-roi' }, 'sparse-scan');
    if (fullCandidate) {
      bestResult = fullCandidate;
      recordCandidateReadings(fullCandidate, 'scan-roi');
    }
  }

  if (!bestResult) {
    const classifierFallback = await tryClassifierFallback();
    if (classifierFallback) {
      bestResult = classifierFallback;
    }
  }

  return {
    bestResult,
    evidenceMap: valueEvidence,
    rejectSummary: summarizeRejectMap(rejectMap)
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

    const worker = await getWorker();
    if (!roiCandidates.length) {
      const message = 'Neural ROI crop did not produce OCR candidates. Enter the measurement manually.';
      if (setProgress) {
        setProgress(message);
      }
      throw new Error(message);
    }

    const roiBranch = await evaluateCandidateBranch({
      candidates: roiCandidates,
      worker,
      setProgress,
      useWordPass: true,
      allowSparseScan: true,
      scanCanvas: roiCrop
    });

    const finalSelection = finalizeSelection({
      debugLabel,
      roiUsed: true,
      bestResult: roiBranch.bestResult,
      evidenceMap: roiBranch.evidenceMap,
      branchUsed: isPreferredLengthReading(roiBranch.bestResult) ? 'roi-accepted' : 'roi-uncertain',
      rejectSummary: roiBranch.rejectSummary || []
    });
    addWinningCandidateDebugStage(debugSession, roiCandidates, finalSelection);

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
