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
  const classifierConfig = OCR_CONFIG.digitClassifier || {};
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
  const roiDeterministic = OCR_CONFIG.roiDeterministic || {};
  const classifierConfig = OCR_CONFIG.digitClassifier || {};
  const branchLabel = 'roi';
  const geometryConfig = OCR_CONFIG.geometry || {};
  const minCandidateWidth = Number.isFinite(geometryConfig.minCandidateWidth) ? geometryConfig.minCandidateWidth : 120;
  const minCandidateHeight = Number.isFinite(geometryConfig.minCandidateHeight) ? geometryConfig.minCandidateHeight : 28;
  const minCandidateAspect = Number.isFinite(geometryConfig.minCandidateAspect) ? geometryConfig.minCandidateAspect : 0.12;
  const maxCandidateAspect = Number.isFinite(geometryConfig.maxCandidateAspect) ? geometryConfig.maxCandidateAspect : 18;
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

  const rankClassifierCandidates = (rawCandidates) => {
    if (!Array.isArray(rawCandidates) || !rawCandidates.length) {
      return [];
    }
    const fallbackPreferNonEdge = classifierConfig.fallbackPreferNonEdge !== false;
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

  if (!classifierConfig.enabled || !classifierConfig.endpoint) {
    recordReject('classifier-disabled', {
      stage: 'classifier-primary'
    });
    return {
      bestResult: null,
      evidenceMap: valueEvidence,
      rejectSummary: summarizeRejectMap(rejectMap)
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
  const selectedCandidates = rankedCandidates.slice(0, maxPrimaryCandidates);
  if (!selectedCandidates.length) {
    recordReject('classifier-no-candidate', {
      stage: 'classifier-primary'
    });
    return {
      bestResult: null,
      evidenceMap: valueEvidence,
      rejectSummary: summarizeRejectMap(rejectMap)
    };
  }

  const nonEdgeAvailable = selectedCandidates.some((candidate) => !isEdgeSourceLabel(candidate.label));
  let pass = 0;
  const expectedPasses = selectedCandidates.length;

  for (const candidate of selectedCandidates) {
    pass += 1;
    if (setProgress) {
      setProgress(`Classifying digits (${pass}/${expectedPasses})`);
    }

    if (isEdgeSourceLabel(candidate.label)) {
      recordReject('classifier-edge-candidate-selected', {
        stage: 'classifier-primary',
        sourceLabel: candidate.label,
        score: Number(candidate.fallbackScore.toFixed(3)),
        aspect: Number(candidate.fallbackAspect.toFixed(3)),
        nonEdgeAlternative: nonEdgeAvailable
      });
    }

    const reading = await readDigitsByCells(candidate.canvas, null, {
      roiMode: true,
      onReject: (detail) => recordDecodeReject(candidate, detail, 'classifier-primary')
    });
    if (!reading) {
      continue;
    }
    if (!isPreferredLengthReading(reading)) {
      recordReject('classifier-non4-reading', {
        stage: 'classifier-primary',
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
      !bestResult
      || classifierReading.score > bestResult.score
      || (
        classifierReading.score === bestResult.score
        && (classifierReading.confidence ?? 0) > (bestResult.confidence ?? 0)
      )
    ) {
      bestResult = classifierReading;
    }
    recordCandidateReadings(classifierReading, `${candidate.label}:classifier`);

    if (
      bestResult
      && bestResult.score >= OCR_CONFIG.earlyStopScore
      && !isEdgeSourceLabel(bestResult.sourceLabel)
    ) {
      break;
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
