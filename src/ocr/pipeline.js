import { OCR_CONFIG } from './config.js';
import { setOcrDebugHooks, startDebugSession, addDebugStage, commitDebugSession } from './debug-hooks.js';
import {
  loadImageBitmap,
  drawImageToCanvas,
  preprocessCanvas,
  cropCanvas,
  normalizeRectToCanvas,
  drawOverlayCanvas
} from './canvas-utils.js';
import { buildDigitCandidates } from './alignment.js';
import { getWorker, selectBestReading, readDigitsByCells } from './recognition.js';
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

const recordSelectionEvidence = (
  evidenceMap,
  reading,
  { sourceLabel = '', isTopPick = false, isRefined = false } = {}
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
    refinedHits: 0,
    totalScore: 0,
    bestScore: -1,
    bestConfidence: 0,
    sources: new Set()
  };

  existing.hits += 1;
  existing.totalScore += score;
  if (isTopPick) {
    existing.topHits += 1;
  }
  if (isRefined) {
    existing.refinedHits += 1;
  }
  if (sourceLabel) {
    existing.sources.add(sourceLabel);
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
      const refinedBoost = clamp(entry.refinedHits * 0.05, 0, 0.15);
      const sourceSpreadBoost = clamp((entry.sources.size - 1) * 0.02, 0, 0.08);
      const preferredLengthBoost = entry.value.length === OCR_CONFIG.preferredDigits ? 0.05 : -0.08;
      const leadingZeroPenalty = (
        entry.value.length === OCR_CONFIG.preferredDigits
        && entry.value.startsWith('0')
      ) ? 0.07 : 0;
      const score = clamp(
        entry.bestScore * 0.58
          + averageScore * 0.27
          + consensusBoost
          + refinedBoost
          + sourceSpreadBoost
          + preferredLengthBoost
          - leadingZeroPenalty,
        0,
        0.99
      );

      return {
        ...entry,
        averageScore,
        score,
        sourceCount: entry.sources.size
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
    refinedHits: entry.refinedHits,
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
  let finalResult = bestResult;

  if (evidenceBest) {
    const shouldPromoteEvidence = (
      !finalResult
      || evidenceBest.score >= (finalResult.score ?? -1) - 0.03
      || evidenceBest.topHits >= 2
      || evidenceBest.refinedHits >= 1
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
        angle: carryMetadata && Number.isFinite(carryMetadata.angle) ? carryMetadata.angle : null,
        cellDigits: carryMetadata && Array.isArray(carryMetadata.cellDigits) ? carryMetadata.cellDigits : null,
        cellConfidences: carryMetadata ? serializeCellConfidences(carryMetadata.cellConfidences) : null
      };
    }
  }

  pushSelectionLog({
    image: debugLabel,
    roiUsed,
    branchUsed,
    rejectSummary,
    selected: finalResult ? {
      value: finalResult.value,
      score: Number((finalResult.score ?? 0).toFixed(3)),
      confidence: Number((finalResult.confidence ?? 0).toFixed(1)),
      branch: finalResult.branch || branchUsed,
      method: finalResult.method || null,
      sourceLabel: finalResult.sourceLabel || null,
      angle: Number.isFinite(finalResult.angle) ? finalResult.angle : null,
      cellDigits: Array.isArray(finalResult.cellDigits) ? finalResult.cellDigits : null,
      cellConfidences: serializeCellConfidences(finalResult.cellConfidences)
    } : null,
    topCandidates: buildSelectionSummary(rankedEvidence, 3)
  });

  return finalResult;
};

const scoreFallbackPriority = (entry, roiUsed) => {
  const label = entry.label || '';
  let priority = entry.score;
  if (label.includes('aligned-')) {
    priority += 0.18;
  }
  if (label.includes('strip-main')) {
    priority += 0.14;
  }
  if (label.includes('strip-tight')) {
    priority += 0.08;
  }
  if (label.includes('-edge')) {
    priority += 0.06;
  }
  if (label.includes('fallback')) {
    priority -= 0.08;
  }
  if (roiUsed && label.endsWith('-roi')) {
    priority += 0.05;
  }
  return priority;
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

const mergeRejectSummaries = (...summaries) => {
  const merged = new Map();
  summaries
    .filter((list) => Array.isArray(list))
    .forEach((list) => {
      list.forEach((entry) => {
        if (!entry || !entry.reason) {
          return;
        }
        const existing = merged.get(entry.reason) || {
          reason: entry.reason,
          count: 0,
          samples: []
        };
        existing.count += Number.isFinite(entry.count) ? entry.count : 1;
        if (Array.isArray(entry.samples)) {
          entry.samples.forEach((sample) => {
            if (existing.samples.length < 3) {
              existing.samples.push(sample);
            }
          });
        }
        merged.set(entry.reason, existing);
      });
    });
  return [...merged.values()].sort((a, b) => b.count - a.count);
};

const evaluateCandidateBranch = async ({
  candidates,
  worker,
  setProgress,
  roiMode = false,
  useWordPass = true,
  allowSparseScan = false,
  scanCanvas = null
}) => {
  const activeCandidates = Array.isArray(candidates) && candidates.length
    ? candidates
    : [{ canvas: scanCanvas, label: roiMode ? 'raw-fallback-roi' : 'raw-fallback-full' }];
  let bestResult = null;
  const valueEvidence = new Map();
  const topPickHits = new Map();
  const candidateScores = new Map();
  const modes = ['binary', 'soft'];
  let pass = 0;
  const expectedPasses = Math.max(1, activeCandidates.length * modes.length);
  const branchLabel = roiMode ? 'roi' : 'fallback';
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
    return {
      ...reading,
      branch: branchLabel,
      method,
      sourceLabel: candidate && candidate.label ? candidate.label : null,
      angle: candidate && candidate.label ? extractCandidateAngle(candidate.label) : null
    };
  };

  const recordCandidateReadings = (reading, sourceLabel, isRefined = false) => {
    if (!reading) {
      return;
    }
    if (Array.isArray(reading.topCandidates) && reading.topCandidates.length) {
      reading.topCandidates.forEach((entry, index) => {
        recordSelectionEvidence(valueEvidence, entry, {
          sourceLabel,
          isTopPick: index === 0,
          isRefined
        });
      });
    } else {
      recordSelectionEvidence(valueEvidence, reading, {
        sourceLabel,
        isTopPick: true,
        isRefined
      });
    }
    if (reading.value) {
      topPickHits.set(reading.value, (topPickHits.get(reading.value) || 0) + 1);
    }
  };

  activeCandidates.forEach((candidate) => {
    candidateScores.set(candidate.label, {
      score: -1,
      canvas: candidate.canvas,
      angle: extractCandidateAngle(candidate.label),
      label: candidate.label
    });
  });

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
        const processed = preprocessCanvas(candidate.canvas, mode);
        const { data } = await worker.recognize(processed);
        let candidateBest = selectBestReading(data, processed);
        if (roiMode && candidateBest && !isPreferredLengthReading(candidateBest)) {
          candidateBest = null;
        }
        candidateBest = applyReadingMetadata(candidateBest, candidate, 'word-pass');
        if (candidateBest && (!bestResult || candidateBest.score > bestResult.score)) {
          bestResult = candidateBest;
        }
        if (candidateBest) {
          recordCandidateReadings(candidateBest, candidate.label, false);
          const existing = candidateScores.get(candidate.label);
          if (!existing || candidateBest.score > existing.score) {
            candidateScores.set(candidate.label, {
              ...existing,
              score: candidateBest.score,
              canvas: candidate.canvas,
              label: candidate.label
            });
          }
        }
        const confirmationHits = bestResult ? (topPickHits.get(bestResult.value) || 0) : 0;
        if (
          bestResult
          && bestResult.score >= OCR_CONFIG.earlyStopScore
          && isPreferredLengthReading(bestResult)
          && confirmationHits >= 2
        ) {
          return {
            bestResult,
            evidenceMap: valueEvidence,
            rejectSummary: summarizeRejectMap(rejectMap)
          };
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
    let fullCandidate = selectBestReading(data, softened);
    if (roiMode && fullCandidate && !isPreferredLengthReading(fullCandidate)) {
      fullCandidate = null;
    }
    fullCandidate = applyReadingMetadata(fullCandidate, { label: roiMode ? 'scan-roi' : 'scan-full' }, 'sparse-scan');
    if (fullCandidate) {
      bestResult = fullCandidate;
      recordCandidateReadings(fullCandidate, roiMode ? 'scan-roi' : 'scan-full', false);
    }
  }

  const angleScores = new Map();
  candidateScores.forEach((entry) => {
    if (entry.score < 0 || !Number.isFinite(entry.angle)) {
      return;
    }
    const previous = angleScores.get(entry.angle) ?? -1;
    angleScores.set(entry.angle, Math.max(previous, entry.score));
  });
  const rankedAngles = [...angleScores.entries()].sort((a, b) => b[1] - a[1]);
  const bestAngleScore = rankedAngles.length ? rankedAngles[0][1] : null;
  const hasAngleScores = rankedAngles.length > 0;
  const allowedAngles = new Set(
    rankedAngles
      .filter(([, score]) => bestAngleScore === null || score >= bestAngleScore - 0.06)
      .map(([angle]) => angle)
  );

  const fallbackLimit = roiMode
    ? activeCandidates.length
    : OCR_CONFIG.fallbackCandidates;
  const fallbackPool = [...candidateScores.values()]
    .filter((entry) => !allowedAngles.size || allowedAngles.has(entry.angle))
    .filter((entry) => !hasAngleScores || entry.score >= -0.5)
    .map((entry) => ({
      ...entry,
      fallbackPriority: scoreFallbackPriority(entry, roiMode)
    }))
    .sort((a, b) => b.fallbackPriority - a.fallbackPriority)
    .slice(0, fallbackLimit);

  if (fallbackPool.length) {
    await worker.setParameters({
      tessedit_pageseg_mode: Tesseract.PSM.SINGLE_CHAR,
      classify_bln_numeric_mode: 1
    });
    for (const candidate of fallbackPool) {
      if (!hasValidCandidateGeometry(candidate, 'cell-refine')) {
        continue;
      }
      if (setProgress) {
        setProgress(roiMode ? 'Refining ROI digits...' : 'Refining digits...');
      }
      const processed = preprocessCanvas(candidate.canvas, 'binary');
      const refined = applyReadingMetadata(
        await readDigitsByCells(worker, processed, setProgress, {
          roiMode,
          onReject: (detail) => recordReject(
            detail && detail.reason ? detail.reason : 'cell-read-reject',
            {
              stage: 'cell-refine',
              sourceLabel: candidate.label,
              ...(detail || {})
            }
          )
        }),
        candidate,
        'cell-refine'
      );
      if (refined && (!bestResult || refined.score > bestResult.score)) {
        bestResult = refined;
      }
      if (refined) {
        recordCandidateReadings(refined, candidate.label, true);
      }
    }
    await worker.setParameters({
      tessedit_pageseg_mode: Tesseract.PSM.SPARSE_TEXT,
      classify_bln_numeric_mode: 1
    });
  } else {
    recordReject('no-fallback-candidates', {
      stage: 'cell-refine',
      roiMode
    });
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
    let roiCrop = null;
    let roiUsed = false;
    let roiProbe = { ok: false, reason: 'disabled' };
    if (neuralRoiConfig.enabled) {
      if (setProgress) {
        setProgress('Requesting neural ROI...');
      }
      roiProbe = await detectNeuralRoi(file, neuralRoiConfig);
      if (roiProbe.ok) {
        const roiRect = resolveNeuralRoiRect(baseCanvas, roiProbe, neuralRoiConfig);
        addNeuralRoiDebugStages(debugSession, baseCanvas, roiRect, roiProbe);
        roiCrop = cropCanvas(baseCanvas, roiRect);
        addDebugStage(debugSession, '0b. neural roi crop', roiCrop);
        roiUsed = true;
      } else {
        addNeuralRoiMissStage(debugSession, baseCanvas, roiProbe);
        if (setProgress) {
          setProgress(`Neural ROI miss (${formatNeuralRoiMissReason(roiProbe)}), using fallback.`);
        }
      }
    }

    const roiCandidates = roiCrop
      ? buildDigitCandidates(roiCrop, debugSession, addDebugStage, { roiMode: true }).map((candidate) => ({
        ...candidate,
        label: `${candidate.label}-roi`
      }))
      : [];
    const fullCandidates = buildDigitCandidates(baseCanvas, roiCrop ? null : debugSession, addDebugStage).map((candidate) => ({
      ...candidate,
      label: `${candidate.label}-full`
    }));

    const worker = await getWorker();
    let roiBranch = null;
    if (roiCandidates.length) {
      roiBranch = await evaluateCandidateBranch({
        candidates: roiCandidates,
        worker,
        setProgress,
        roiMode: true,
        useWordPass: false,
        allowSparseScan: false,
        scanCanvas: roiCrop
      });
      const roiAccepted = isPreferredLengthReading(roiBranch.bestResult);
      if (roiAccepted) {
        if (setProgress) {
          setProgress(`Neural ROI + OCR complete (${roiBranch.bestResult.value}).`);
        }
        return finalizeSelection({
          debugLabel,
          roiUsed: true,
          bestResult: roiBranch.bestResult,
          evidenceMap: roiBranch.evidenceMap,
          branchUsed: 'roi-accepted',
          rejectSummary: roiBranch.rejectSummary || []
        });
      }
      if (neuralRoiConfig.skipFullFallbackWhenDetected !== false) {
        if (setProgress) {
          if (roiBranch.bestResult && roiBranch.bestResult.value) {
            setProgress(`Neural ROI branch complete (${roiBranch.bestResult.value}); full fallback skipped.`);
          } else {
            setProgress('Neural ROI branch complete; full fallback skipped.');
          }
        }
        return finalizeSelection({
          debugLabel,
          roiUsed: true,
          bestResult: roiBranch.bestResult,
          evidenceMap: roiBranch.evidenceMap,
          branchUsed: 'roi-skipped-fallback',
          rejectSummary: roiBranch.rejectSummary || []
        });
      }
      if (setProgress) {
        setProgress('Neural ROI uncertain, running full fallback...');
      }
    }

    const fullBranch = await evaluateCandidateBranch({
      candidates: fullCandidates,
      worker,
      setProgress,
      roiMode: false,
      useWordPass: true,
      allowSparseScan: true,
      scanCanvas: baseCanvas
    });
    const bestResult = fullBranch.bestResult || (roiBranch ? roiBranch.bestResult : null);
    const evidenceMap = (
      fullBranch.evidenceMap && fullBranch.evidenceMap.size
        ? fullBranch.evidenceMap
        : (roiBranch ? roiBranch.evidenceMap : new Map())
    );

    if (setProgress) {
      if (bestResult && bestResult.value) {
        setProgress(`Fallback OCR complete (${bestResult.value}).`);
      } else if (roiUsed) {
        setProgress('Neural ROI found, OCR uncertain.');
      } else if (neuralRoiConfig.enabled) {
        setProgress(`Fallback OCR complete (neural miss: ${formatNeuralRoiMissReason(roiProbe)}).`);
      }
    }

    return finalizeSelection({
      debugLabel,
      roiUsed,
      bestResult,
      evidenceMap,
      branchUsed: roiUsed ? 'fallback-after-roi' : 'fallback-only',
      rejectSummary: mergeRejectSummaries(
        roiBranch ? roiBranch.rejectSummary : [],
        fullBranch ? fullBranch.rejectSummary : []
      )
    });
  } finally {
    commitDebugSession(debugSession);
  }
};

export { runMeterOcr, setOcrDebugHooks };
