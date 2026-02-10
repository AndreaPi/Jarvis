import { OCR_CONFIG } from './config.js';
import { setOcrDebugHooks, startDebugSession, addDebugStage, commitDebugSession } from './debug-hooks.js';
import { loadImageBitmap, drawImageToCanvas, preprocessCanvas } from './canvas-utils.js';
import { buildDigitCandidates } from './alignment.js';
import { getWorker, selectBestReading, readDigitsByCells } from './recognition.js';

const runMeterOcr = async (file, setProgress) => {
  const image = await loadImageBitmap(file);
  const baseCanvas = drawImageToCanvas(image, OCR_CONFIG.maxDimension);
  const debugLabel = file && file.name ? file.name : `manual-${Date.now()}`;
  const debugSession = startDebugSession(debugLabel);

  try {
    const candidates = buildDigitCandidates(baseCanvas, debugSession, addDebugStage);
    const modes = ['binary', 'soft'];
    let bestResult = null;
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

    for (const candidate of candidates) {
      if (!candidateScores.has(candidate.label)) {
        const [angleToken] = candidate.label.split('-');
        candidateScores.set(candidate.label, {
          score: -1,
          canvas: candidate.canvas,
          angle: Number.parseInt(angleToken, 10)
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
          const existing = candidateScores.get(candidate.label);
          if (!existing || candidateBest.score > existing.score) {
            candidateScores.set(candidate.label, { ...existing, score: candidateBest.score, canvas: candidate.canvas });
          }
        }
        if (bestResult && bestResult.score >= OCR_CONFIG.earlyStopScore && bestResult.value.length === OCR_CONFIG.preferredDigits) {
          return bestResult;
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
      const softened = preprocessCanvas(baseCanvas, 'soft');
      const { data } = await worker.recognize(softened);
      const fullCandidate = selectBestReading(data, softened);
      if (fullCandidate) {
        bestResult = fullCandidate;
      }
    }

    if (bestResult && bestResult.score >= OCR_CONFIG.fallbackScoreThreshold && bestResult.value.length === OCR_CONFIG.preferredDigits) {
      return bestResult;
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
      .sort((a, b) => b.score - a.score)
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
      }
      await worker.setParameters({
        tessedit_pageseg_mode: Tesseract.PSM.SPARSE_TEXT,
        classify_bln_numeric_mode: 1
      });
    }

    return bestResult;
  } finally {
    commitDebugSession(debugSession);
  }
};

export { runMeterOcr, setOcrDebugHooks };
