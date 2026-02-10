import { OCR_CONFIG } from './config.js';
import {
  clamp,
  preprocessCanvas,
  scaleCanvas,
  splitIntoCells,
  cropCanvas,
  tightenCropByInk
} from './canvas-utils.js';

let ocrWorker = null;

const getWorker = async () => {
  if (ocrWorker) {
    return ocrWorker;
  }

  const worker = await Tesseract.createWorker('eng');
  if (worker.loadLanguage && worker.initialize) {
    await worker.loadLanguage('eng');
    await worker.initialize('eng');
  }
  if (worker.setParameters) {
    await worker.setParameters({
      tessedit_char_whitelist: '0123456789',
      tessedit_pageseg_mode: Tesseract.PSM.SINGLE_WORD,
      classify_bln_numeric_mode: 1
    });
  }

  ocrWorker = worker;
  return ocrWorker;
};

const buildCandidateScores = (data, canvas) => {
  const candidates = [];

  const pushCandidate = (value, confidence, areaRatio) => {
    if (!value) {
      return;
    }
    candidates.push({
      value,
      confidence: confidence ?? data.confidence ?? 0,
      areaRatio: areaRatio ?? 0
    });
  };

  const collectDigitSequence = (items) => {
    if (!items) {
      return null;
    }

    const canvasArea = Math.max(1, canvas.width * canvas.height);
    const digits = items
      .filter((item) => item.text && /^\d$/.test(item.text))
      .map((item) => {
        const box = item.bbox || item;
        const x0 = Number.isFinite(box.x0) ? box.x0 : 0;
        const y0 = Number.isFinite(box.y0) ? box.y0 : 0;
        const x1 = Number.isFinite(box.x1) ? box.x1 : x0 + 1;
        const y1 = Number.isFinite(box.y1) ? box.y1 : y0 + 1;
        const width = Math.max(1, x1 - x0);
        const height = Math.max(1, y1 - y0);
        return {
          digit: item.text,
          x0,
          x1: x0 + width,
          centerY: y0 + height / 2,
          width,
          height,
          areaRatio: (width * height) / canvasArea,
          confidence: item.confidence ?? data.confidence ?? 0
        };
      })
      .sort((a, b) => a.x0 - b.x0);

    if (digits.length < OCR_CONFIG.preferredDigits) {
      return null;
    }

    const isNeighbor = (left, right) => {
      const gap = right.x0 - left.x1;
      const maxGap = Math.max(left.width, right.width) * 2.4;
      const yDelta = Math.abs(right.centerY - left.centerY);
      const maxYDelta = Math.max(left.height, right.height) * 1.2;
      return gap <= maxGap && yDelta <= maxYDelta;
    };

    let bestSequence = null;
    for (let i = 0; i <= digits.length - OCR_CONFIG.preferredDigits; i += 1) {
      const sequence = digits.slice(i, i + OCR_CONFIG.preferredDigits);
      const contiguous = sequence.every((digit, index) => index === 0 || isNeighbor(sequence[index - 1], digit));
      if (!contiguous) {
        continue;
      }

      const confidence = sequence.reduce((sum, item) => sum + item.confidence, 0) / sequence.length;
      const areaRatio = sequence.reduce((sum, item) => sum + item.areaRatio, 0) / sequence.length;
      const candidate = {
        value: sequence.map((item) => item.digit).join(''),
        confidence,
        areaRatio
      };

      if (
        !bestSequence
        || candidate.areaRatio > bestSequence.areaRatio
        || (candidate.areaRatio === bestSequence.areaRatio && candidate.confidence > bestSequence.confidence)
      ) {
        bestSequence = candidate;
      }
    }

    return bestSequence;
  };

  const textMatches = (data.text || '').match(/\d+/g);
  if (textMatches) {
    textMatches.forEach((chunk) => {
      if (chunk.length >= OCR_CONFIG.minDigits) {
        pushCandidate(chunk, data.confidence, 0.15);
        if (chunk.length > OCR_CONFIG.preferredDigits) {
          pushCandidate(chunk.slice(0, OCR_CONFIG.preferredDigits), data.confidence, 0.15);
          pushCandidate(chunk.slice(-OCR_CONFIG.preferredDigits), data.confidence, 0.15);
        }
      }
    });
  }

  if (data.words) {
    const wordSequence = collectDigitSequence(data.words);
    if (wordSequence) {
      pushCandidate(wordSequence.value, wordSequence.confidence, wordSequence.areaRatio);
    }
    data.words.forEach((word) => {
      const digits = (word.text || '').replace(/\D/g, '');
      if (digits.length >= OCR_CONFIG.minDigits) {
        const box = word.bbox || {};
        const area = (box.x1 - box.x0 || 0) * (box.y1 - box.y0 || 0);
        const ratio = area / (canvas.width * canvas.height);
        pushCandidate(digits, word.confidence, ratio);
        if (digits.length > OCR_CONFIG.preferredDigits) {
          pushCandidate(digits.slice(0, OCR_CONFIG.preferredDigits), word.confidence, ratio);
          pushCandidate(digits.slice(-OCR_CONFIG.preferredDigits), word.confidence, ratio);
        }
      }
    });
  }

  if (data.symbols) {
    const symbolSequence = collectDigitSequence(data.symbols);
    if (symbolSequence) {
      pushCandidate(symbolSequence.value, symbolSequence.confidence, symbolSequence.areaRatio);
    }
  }

  return candidates;
};

const scoreCandidate = (candidate) => {
  const length = candidate.value.length;
  const lengthScore = length === OCR_CONFIG.preferredDigits ? 1 : length === 5 ? 0.75 : length === 3 ? 0.45 : 0.2;
  const confidenceScore = clamp(candidate.confidence / 100, 0, 1);
  const areaScore = clamp(candidate.areaRatio * 4, 0, 1);
  return confidenceScore * 0.6 + lengthScore * 0.3 + areaScore * 0.1;
};

const selectBestReading = (data, canvas) => {
  const candidates = buildCandidateScores(data, canvas);
  const preferred = candidates.filter((candidate) => candidate.value.length === OCR_CONFIG.preferredDigits);
  const shortlist = preferred.length ? preferred : candidates;
  let best = null;
  shortlist.forEach((candidate) => {
    const score = scoreCandidate(candidate);
    if (!best || score > best.score) {
      best = { ...candidate, score };
    }
  });
  return best;
};

const readDigitsByCells = async (worker, source, setProgress) => {
  const pickDigit = (data) => {
    const symbolDigits = (data.symbols || [])
      .map((item) => ({
        digit: (item.text || '').replace(/\D/g, '').slice(0, 1),
        confidence: Number.isFinite(item.confidence) ? item.confidence : data.confidence ?? 0
      }))
      .filter((item) => item.digit);
    const bestSymbol = symbolDigits.sort((a, b) => b.confidence - a.confidence)[0];
    if (bestSymbol) {
      return bestSymbol;
    }
    const textDigits = (data.text || '').replace(/\D/g, '');
    if (textDigits) {
      return {
        digit: textDigits[0],
        confidence: Number.isFinite(data.confidence) ? data.confidence : 0
      };
    }
    return null;
  };

  const decodeCells = async (cellCanvases) => {
    const digits = [];
    let confidenceTotal = 0;
    let found = 0;

    for (let i = 0; i < cellCanvases.length; i += 1) {
      if (setProgress) {
        setProgress(`Refining digits (${i + 1}/${cellCanvases.length})`);
      }
      let best = null;
      for (const mode of ['binary', 'soft']) {
        let cell = preprocessCanvas(cellCanvases[i], mode);
        cell = scaleCanvas(cell, OCR_CONFIG.minDigitWidth);
        const { data } = await worker.recognize(cell);
        const picked = pickDigit(data);
        if (picked && (!best || picked.confidence > best.confidence)) {
          best = picked;
        }
      }

      if (best) {
        digits.push(best.digit);
        confidenceTotal += best.confidence;
        found += 1;
      } else {
        digits.push('0');
      }
    }

    return {
      value: digits.join(''),
      foundRatio: found / cellCanvases.length,
      averageConfidence: confidenceTotal / cellCanvases.length
    };
  };

  const cropToFocus = (canvas, rect) => {
    return cropCanvas(canvas, {
      x: canvas.width * rect.x,
      y: canvas.height * rect.y,
      width: canvas.width * rect.width,
      height: canvas.height * rect.height
    });
  };

  const focusRects = [
    { x: 0.0, y: 0.0, width: 0.62, height: 1.0 },
    { x: 0.02, y: 0.05, width: 0.6, height: 0.88 },
    { x: 0.05, y: 0.1, width: 0.58, height: 0.8 }
  ];
  const variants = [];
  focusRects.forEach((rect) => {
    const focused = cropToFocus(source, rect);
    variants.push(focused);
    variants.push(tightenCropByInk(focused, 0.08));
    variants.push(tightenCropByInk(focused, 0.18));
  });
  let bestReading = null;

  for (const variant of variants) {
    for (const overlap of [0.03, OCR_CONFIG.digitCellOverlap]) {
      const cellCanvases = splitIntoCells(variant, OCR_CONFIG.digitCellCount, overlap);
      const reading = await decodeCells(cellCanvases);
      if (
        !bestReading
        || reading.foundRatio > bestReading.foundRatio
        || (reading.foundRatio === bestReading.foundRatio && reading.averageConfidence > bestReading.averageConfidence)
      ) {
        bestReading = reading;
      }
    }
  }

  if (!bestReading) {
    return null;
  }

  const tunedConfidence = clamp(bestReading.averageConfidence + bestReading.foundRatio * 25, 0, 100);
  const bonus = bestReading.foundRatio === 1 ? 0.2 : 0.04;
  const score = scoreCandidate({ value: bestReading.value, confidence: tunedConfidence, areaRatio: 0.28 }) + bonus;

  return {
    value: bestReading.value,
    confidence: tunedConfidence,
    areaRatio: 0.28,
    score: clamp(score, 0, 0.99)
  };
};

export { getWorker, selectBestReading, readDigitsByCells };
