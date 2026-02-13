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
  console.info('[OCR] selection', payload);
};

const finalizeSelection = ({ debugLabel, roiUsed, bestResult, evidenceMap }) => {
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
      finalResult = {
        value: evidenceBest.value,
        confidence: Math.max(evidenceBest.bestConfidence, confidenceFromBest),
        areaRatio: finalResult && finalResult.value === evidenceBest.value
          ? (finalResult.areaRatio ?? 0.28)
          : 0.28,
        score: evidenceBest.score
      };
    }
  }

  pushSelectionLog({
    image: debugLabel,
    roiUsed,
    selected: finalResult ? {
      value: finalResult.value,
      score: Number((finalResult.score ?? 0).toFixed(3)),
      confidence: Number((finalResult.confidence ?? 0).toFixed(1))
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

const runMeterOcr = async (file, setProgress) => {
  const image = await loadImageBitmap(file);
  const baseCanvas = drawImageToCanvas(image, OCR_CONFIG.maxDimension);
  const debugLabel = file && file.name ? file.name : `manual-${Date.now()}`;
  const debugSession = startDebugSession(debugLabel);

  try {
    const neuralRoiConfig = OCR_CONFIG.neuralRoi || {};
    let roiCrop = null;
    let roiUsed = false;
    if (neuralRoiConfig.enabled) {
      if (setProgress) {
        setProgress('Requesting neural ROI...');
      }
      const roiDetection = await detectNeuralRoi(file, neuralRoiConfig);
      if (roiDetection) {
        const roiRect = resolveNeuralRoiRect(baseCanvas, roiDetection, neuralRoiConfig);
        addNeuralRoiDebugStages(debugSession, baseCanvas, roiRect, roiDetection);
        roiCrop = cropCanvas(baseCanvas, roiRect);
        addDebugStage(debugSession, '0b. neural roi crop', roiCrop);
        roiUsed = true;
      }
    }

    const candidateBuckets = [];
    if (roiCrop) {
      const roiCandidates = buildDigitCandidates(roiCrop, debugSession, addDebugStage);
      candidateBuckets.push(...roiCandidates.map((candidate) => ({
        ...candidate,
        label: `${candidate.label}-roi`
      })));
    }
    if (!roiCrop || neuralRoiConfig.includeFullFallbackCandidates) {
      const fullCandidates = buildDigitCandidates(baseCanvas, roiCrop ? null : debugSession, addDebugStage);
      candidateBuckets.push(...fullCandidates.map((candidate) => ({
        ...candidate,
        label: `${candidate.label}-full`
      })));
    }

    const candidates = candidateBuckets.length
      ? candidateBuckets
      : [{ canvas: baseCanvas, label: 'raw-fallback-full' }];
    const modes = ['binary', 'soft'];
    let bestResult = null;
    const valueEvidence = new Map();
    const topPickHits = new Map();
    const candidateScores = new Map();
    let pass = 0;
    const worker = await getWorker();
    if (worker.setParameters) {
      await worker.setParameters({
        tessedit_pageseg_mode: Tesseract.PSM.SINGLE_WORD,
        tessedit_char_whitelist: '0123456789',
        classify_bln_numeric_mode: 1
      });
    }

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

    for (const candidate of candidates) {
      if (!candidateScores.has(candidate.label)) {
        const [angleToken] = candidate.label.split('-');
        candidateScores.set(candidate.label, {
          score: -1,
          canvas: candidate.canvas,
          angle: Number.parseInt(angleToken, 10),
          label: candidate.label
        });
      }
      for (const mode of modes) {
        pass += 1;
        if (setProgress) {
          setProgress(`Analyzing meter (${pass}/${candidates.length * modes.length})`);
        }
        const processed = preprocessCanvas(candidate.canvas, mode);
        const { data } = await worker.recognize(processed);
        const candidateBest = selectBestReading(data, processed);
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
          && bestResult.value.length === OCR_CONFIG.preferredDigits
          && confirmationHits >= 2
        ) {
          return finalizeSelection({
            debugLabel,
            roiUsed,
            bestResult,
            evidenceMap: valueEvidence
          });
        }
      }
    }

    if (!bestResult) {
      if (setProgress) {
        setProgress('Scanning full image...');
      }
      if (worker.setParameters) {
        await worker.setParameters({
          tessedit_pageseg_mode: Tesseract.PSM.SPARSE_TEXT,
          tessedit_char_whitelist: '0123456789'
        });
      }
      const scanCanvas = roiCrop && !neuralRoiConfig.includeFullFallbackCandidates ? roiCrop : baseCanvas;
      const softened = preprocessCanvas(scanCanvas, 'soft');
      const { data } = await worker.recognize(softened);
      const fullCandidate = selectBestReading(data, softened);
      if (fullCandidate) {
        bestResult = fullCandidate;
        recordCandidateReadings(fullCandidate, 'scan-full', false);
      }
    }

    if (bestResult && bestResult.score >= OCR_CONFIG.fallbackScoreThreshold && bestResult.value.length === OCR_CONFIG.preferredDigits) {
      return finalizeSelection({
        debugLabel,
        roiUsed,
        bestResult,
        evidenceMap: valueEvidence
      });
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

    const fallbackPool = [...candidateScores.values()]
      .filter((entry) => !allowedAngles.size || allowedAngles.has(entry.angle))
      .filter((entry) => !hasAngleScores || entry.score >= -0.5)
      .map((entry) => ({
        ...entry,
        fallbackPriority: scoreFallbackPriority(entry, roiUsed)
      }))
      .sort((a, b) => b.fallbackPriority - a.fallbackPriority)
      .slice(0, OCR_CONFIG.fallbackCandidates);

    if (fallbackPool.length) {
      await worker.setParameters({
        tessedit_pageseg_mode: Tesseract.PSM.SINGLE_CHAR,
        classify_bln_numeric_mode: 1
      });
      for (const candidate of fallbackPool) {
        if (setProgress) {
          setProgress('Refining digits...');
        }
        const processed = preprocessCanvas(candidate.canvas, 'binary');
        const refined = await readDigitsByCells(worker, processed, setProgress);
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
    }

    if (setProgress && roiUsed) {
      if (bestResult && bestResult.value) {
        setProgress(`Neural ROI + OCR complete (${bestResult.value}).`);
      } else {
        setProgress('Neural ROI found, OCR uncertain.');
      }
    }

    return finalizeSelection({
      debugLabel,
      roiUsed,
      bestResult,
      evidenceMap: valueEvidence
    });
  } finally {
    commitDebugSession(debugSession);
  }
};

export { runMeterOcr, setOcrDebugHooks };
