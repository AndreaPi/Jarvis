import { OCR_CONFIG } from './config.js';
import {
  clamp,
  preprocessCanvas,
  scaleCanvas,
  splitIntoCells,
  cropCanvas,
  tightenCropByInk,
  rotateCanvas,
  normalizeAngle
} from './canvas-utils.js';
import { predictDigitCells } from './digit-classifier.js';

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

const rankReadings = (data, canvas, limit = 3) => {
  const candidates = buildCandidateScores(data, canvas);
  if (!candidates.length) {
    return [];
  }

  const valueBuckets = new Map();
  candidates.forEach((candidate) => {
    const score = scoreCandidate(candidate);
    const value = candidate.value;
    if (!valueBuckets.has(value)) {
      valueBuckets.set(value, {
        value,
        confidence: candidate.confidence,
        areaRatio: candidate.areaRatio,
        bestScore: score,
        totalScore: score,
        hits: 1
      });
      return;
    }

    const existing = valueBuckets.get(value);
    existing.hits += 1;
    existing.totalScore += score;
    if (score > existing.bestScore) {
      existing.bestScore = score;
      existing.confidence = candidate.confidence;
      existing.areaRatio = candidate.areaRatio;
    }
  });

  const ranked = [...valueBuckets.values()]
    .map((entry) => {
      const averageScore = entry.totalScore / entry.hits;
      const consensusBoost = clamp((entry.hits - 1) * 0.08, 0, 0.24);
      const preferredLengthBoost = entry.value.length === OCR_CONFIG.preferredDigits ? 0.05 : -0.05;
      const leadingZeroPenalty = (
        entry.value.length === OCR_CONFIG.preferredDigits
        && entry.value.startsWith('0')
      ) ? 0.06 : 0;
      const score = clamp(
        entry.bestScore * 0.7 + averageScore * 0.25 + consensusBoost + preferredLengthBoost - leadingZeroPenalty,
        0,
        0.99
      );

      return {
        value: entry.value,
        confidence: entry.confidence,
        areaRatio: entry.areaRatio,
        score,
        bestScore: entry.bestScore,
        averageScore,
        hits: entry.hits
      };
    })
    .sort((a, b) => b.score - a.score || b.hits - a.hits || b.bestScore - a.bestScore);

  if (!Number.isFinite(limit) || limit <= 0) {
    return ranked;
  }
  return ranked.slice(0, limit);
};

const selectBestReading = (data, canvas) => {
  const ranked = rankReadings(data, canvas, 3);
  if (!ranked.length) {
    return null;
  }
  return {
    ...ranked[0],
    topCandidates: ranked
  };
};

const readDigitsByCells = async (worker, source, setProgress, options = {}) => {
  const geometry = OCR_CONFIG.geometry || {};
  const roiDeterministic = OCR_CONFIG.roiDeterministic || {};
  const minCandidateWidth = Number.isFinite(geometry.minCandidateWidth) ? geometry.minCandidateWidth : 120;
  const minCandidateHeight = Number.isFinite(geometry.minCandidateHeight) ? geometry.minCandidateHeight : 28;
  const minCandidateAspect = Number.isFinite(geometry.minCandidateAspect) ? geometry.minCandidateAspect : 0.12;
  const maxCandidateAspect = Number.isFinite(geometry.maxCandidateAspect) ? geometry.maxCandidateAspect : 18;
  const minCellWidth = Number.isFinite(geometry.minCellWidth) ? geometry.minCellWidth : 20;
  const minCellHeight = Number.isFinite(geometry.minCellHeight) ? geometry.minCellHeight : 24;

  const emitReject = (reason, detail = {}) => {
    if (typeof options.onReject === 'function') {
      options.onReject({ reason, ...detail });
    }
  };

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

  const hasValidCandidateGeometry = (canvas, context = {}) => {
    if (!canvas) {
      emitReject('candidate-missing', context);
      return false;
    }
    const width = canvas.width;
    const height = canvas.height;
    const aspect = width / Math.max(1, height);
    const mode = context && typeof context.mode === 'string' ? context.mode : '';
    const isRoiMode = mode.startsWith('roi-');
    if (width < minCandidateWidth || height < minCandidateHeight) {
      emitReject('candidate-too-small', { width, height, ...context });
      return false;
    }
    if (aspect < minCandidateAspect || aspect > maxCandidateAspect) {
      if (isRoiMode) {
        const relaxedMinAspect = minCandidateAspect * 0.8;
        const relaxedMaxAspect = maxCandidateAspect * 1.12;
        if (aspect >= relaxedMinAspect && aspect <= relaxedMaxAspect) {
          emitReject('candidate-bad-aspect-soft', {
            width,
            height,
            aspect: Number(aspect.toFixed(3)),
            minCandidateAspect,
            maxCandidateAspect,
            relaxedMinAspect: Number(relaxedMinAspect.toFixed(3)),
            relaxedMaxAspect: Number(relaxedMaxAspect.toFixed(3)),
            ...context
          });
          return true;
        }
      }
      emitReject('candidate-bad-aspect', { width, height, aspect: Number(aspect.toFixed(3)), ...context });
      return false;
    }
    return true;
  };

  const hasValidCellGeometry = (cellCanvases, context = {}) => {
    for (let i = 0; i < cellCanvases.length; i += 1) {
      const cell = cellCanvases[i];
      if (!cell || cell.width < minCellWidth || cell.height < minCellHeight) {
        emitReject('cell-too-small', {
          index: i,
          width: cell ? cell.width : 0,
          height: cell ? cell.height : 0,
          ...context
        });
        return false;
      }
    }
    return true;
  };

  const resizeCanvasWidth = (canvas, targetWidth) => {
    if (!canvas || !Number.isFinite(targetWidth) || targetWidth <= 0) {
      return canvas;
    }
    if (canvas.width === Math.round(targetWidth)) {
      return canvas;
    }
    const scale = targetWidth / Math.max(1, canvas.width);
    const resized = document.createElement('canvas');
    resized.width = Math.max(1, Math.round(canvas.width * scale));
    resized.height = Math.max(1, Math.round(canvas.height * scale));
    const ctx = resized.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(canvas, 0, 0, resized.width, resized.height);
    return resized;
  };

  const rotateCanvasExpanded = (canvas, angle) => {
    const normalized = normalizeAngle(angle);
    if (!Number.isFinite(normalized) || normalized === 0) {
      return canvas;
    }
    if (normalized % 90 === 0) {
      return rotateCanvas(canvas, normalized);
    }
    const radians = (normalized * Math.PI) / 180;
    const cos = Math.cos(radians);
    const sin = Math.sin(radians);
    const width = Math.max(
      1,
      Math.round(Math.abs(canvas.width * cos) + Math.abs(canvas.height * sin))
    );
    const height = Math.max(
      1,
      Math.round(Math.abs(canvas.width * sin) + Math.abs(canvas.height * cos))
    );
    const rotated = document.createElement('canvas');
    rotated.width = width;
    rotated.height = height;
    const ctx = rotated.getContext('2d');
    ctx.translate(width * 0.5, height * 0.5);
    ctx.rotate(radians);
    ctx.drawImage(canvas, -canvas.width * 0.5, -canvas.height * 0.5);
    return rotated;
  };

  const buildDeskewAngles = () => {
    const maxAngleRaw = Number.isFinite(roiDeterministic.deskewMaxAngle)
      ? roiDeterministic.deskewMaxAngle
      : 8;
    const stepRaw = Number.isFinite(roiDeterministic.deskewStep)
      ? roiDeterministic.deskewStep
      : 2;
    const maxAngle = Math.max(0, Math.min(20, Math.abs(maxAngleRaw)));
    const step = Math.max(1, Math.min(10, Math.abs(stepRaw)));
    const angles = [0];
    for (let delta = step; delta <= maxAngle; delta += step) {
      const rounded = Number(delta.toFixed(3));
      angles.push(rounded);
      angles.push(-rounded);
    }
    return angles;
  };

  const scoreDeskewCandidate = (sourceCanvas, tightenRatio, angle) => {
    const rotated = angle === 0 ? sourceCanvas : rotateCanvasExpanded(sourceCanvas, angle);
    const tightened = tightenCropByInk(rotated, tightenRatio);
    if (!tightened) {
      return null;
    }
    const aspect = tightened.width / Math.max(1, tightened.height);
    const areaRatio = (tightened.width * tightened.height) / Math.max(1, rotated.width * rotated.height);
    // Prefer wider strips; penalize rotations that shrink ink area below 14% of the bounding box.
    const score = aspect - Math.max(0, 0.14 - areaRatio) * 3.5;
    return {
      canvas: tightened,
      angle,
      score
    };
  };

  const normalizeRoiOrientation = (sourceCanvas, tightenRatio) => {
    const angles = buildDeskewAngles();
    let best = scoreDeskewCandidate(sourceCanvas, tightenRatio, 0);
    if (!best) {
      return {
        canvas: sourceCanvas,
        deskewAngle: 0
      };
    }
    angles.forEach((angle) => {
      if (angle === 0) {
        return;
      }
      const candidate = scoreDeskewCandidate(sourceCanvas, tightenRatio, angle);
      if (!candidate) {
        return;
      }
      if (candidate.score > best.score + 0.02) {
        best = candidate;
      }
    });
    return {
      canvas: best.canvas,
      deskewAngle: normalizeAngle(best.angle)
    };
  };

  const buildInkProjection = (canvas) => {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const { width, height } = canvas;
    const data = ctx.getImageData(0, 0, width, height).data;
    const columns = new Array(width).fill(0);
    const rows = new Array(height).fill(0);

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const idx = (y * width + x) * 4;
        const lum = data[idx] * 0.2126 + data[idx + 1] * 0.7152 + data[idx + 2] * 0.0722;
        const ink = 255 - lum;
        columns[x] += ink;
        rows[y] += ink;
      }
    }

    return { columns, rows };
  };

  const findMaxInkWindowStart = (values, windowSize) => {
    if (!Array.isArray(values) || !values.length) {
      return 0;
    }
    const size = Math.min(values.length, Math.max(1, Math.round(windowSize)));
    let windowSum = 0;
    for (let i = 0; i < size; i += 1) {
      windowSum += values[i];
    }
    let bestSum = windowSum;
    let bestStart = 0;
    for (let i = size; i < values.length; i += 1) {
      windowSum += values[i] - values[i - size];
      const start = i - size + 1;
      if (windowSum > bestSum) {
        bestSum = windowSum;
        bestStart = start;
      }
    }
    return bestStart;
  };

  const normalizeRoiStripCanvas = (canvas) => {
    const tightenRatio = Number.isFinite(roiDeterministic.tightenInk) ? roiDeterministic.tightenInk : 0.08;
    const minStripAspect = Number.isFinite(roiDeterministic.minStripAspect) ? roiDeterministic.minStripAspect : 1.8;
    const maxStripAspect = Number.isFinite(roiDeterministic.maxStripAspect) ? roiDeterministic.maxStripAspect : 6.5;
    const normalizeWidth = Number.isFinite(roiDeterministic.normalizeWidth) ? roiDeterministic.normalizeWidth : OCR_CONFIG.minScaleWidth;
    const orientationNormalized = normalizeRoiOrientation(canvas, tightenRatio);
    let normalized = orientationNormalized.canvas;
    if (!hasValidCandidateGeometry(normalized, { mode: 'roi-initial' })) {
      return null;
    }

    let aspect = normalized.width / Math.max(1, normalized.height);
    if (aspect < minStripAspect) {
      const targetHeight = Math.max(minCandidateHeight, Math.min(normalized.height, Math.round(normalized.width / minStripAspect)));
      if (targetHeight < normalized.height) {
        const { rows } = buildInkProjection(normalized);
        const startY = findMaxInkWindowStart(rows, targetHeight);
        normalized = cropCanvas(normalized, {
          x: 0,
          y: startY,
          width: normalized.width,
          height: targetHeight
        });
      }
    } else if (aspect > maxStripAspect) {
      const targetWidth = Math.max(minCandidateWidth, Math.min(normalized.width, Math.round(normalized.height * maxStripAspect)));
      if (targetWidth < normalized.width) {
        const { columns } = buildInkProjection(normalized);
        const startX = findMaxInkWindowStart(columns, targetWidth);
        normalized = cropCanvas(normalized, {
          x: startX,
          y: 0,
          width: targetWidth,
          height: normalized.height
        });
      }
    }

    normalized = resizeCanvasWidth(normalized, normalizeWidth);
    if (!hasValidCandidateGeometry(normalized, { mode: 'roi-normalized' })) {
      return null;
    }

    aspect = normalized.width / Math.max(1, normalized.height);
    if (aspect < minStripAspect || aspect > maxStripAspect) {
      const hardMinStripAspect = minStripAspect * 0.96;
      const hardMaxStripAspect = maxStripAspect * 1.06;
      const detail = {
        width: normalized.width,
        height: normalized.height,
        aspect: Number(aspect.toFixed(3)),
        minStripAspect,
        maxStripAspect,
        hardMinStripAspect: Number(hardMinStripAspect.toFixed(3)),
        hardMaxStripAspect: Number(hardMaxStripAspect.toFixed(3))
      };
      if (aspect < hardMinStripAspect || aspect > hardMaxStripAspect) {
        emitReject('roi-strip-aspect-out-of-range', detail);
        return null;
      }
      emitReject('roi-strip-aspect-soft', detail);
    }

    return {
      canvas: normalized,
      deskewAngle: Number.isFinite(orientationNormalized.deskewAngle)
        ? orientationNormalized.deskewAngle
        : 0
    };
  };

  const decodeCells = async (cellCanvases, metadata = {}, decodeOptions = {}) => {
    const requireAllCells = !!decodeOptions.requireAllCells;
    const minFound = requireAllCells ? cellCanvases.length : OCR_CONFIG.minDigits;

    const buildReading = (digits, cellConfidences, decoder, extra = {}) => {
      let confidenceTotal = 0;
      let found = 0;
      for (let i = 0; i < digits.length; i += 1) {
        if (!digits[i]) {
          continue;
        }
        found += 1;
        confidenceTotal += Number.isFinite(cellConfidences[i]) ? cellConfidences[i] : 0;
      }
      const value = digits.join('');
      if (found < minFound || !value) {
        return {
          ok: false,
          found,
          value
        };
      }
      return {
        ok: true,
        reading: {
          value,
          foundRatio: found / cellCanvases.length,
          averageConfidence: confidenceTotal / Math.max(found, 1),
          cellDigits: digits,
          cellConfidences,
          variantIndex: metadata.variantIndex,
          overlap: metadata.overlap,
          orientation: Number.isFinite(metadata.orientation) ? metadata.orientation : null,
          deskewAngle: Number.isFinite(metadata.deskewAngle) ? metadata.deskewAngle : null,
          decoder,
          ...extra
        }
      };
    };

    const digitClassifierConfig = OCR_CONFIG.digitClassifier || {};
    if (digitClassifierConfig.enabled) {
      if (setProgress) {
        setProgress('Refining digits (classifier)...');
      }
      const classifierProbe = await predictDigitCells(cellCanvases, digitClassifierConfig);
      if (classifierProbe.ok) {
        const digits = classifierProbe.predictions.map((item) => (item && item.accepted ? item.digit : ''));
        const cellConfidences = classifierProbe.predictions.map((item) => {
          if (!item || !item.accepted || !Number.isFinite(item.confidence)) {
            return 0;
          }
          return clamp(item.confidence * 100, 0, 100);
        });
        const classifierReading = buildReading(
          digits,
          cellConfidences,
          'digit-classifier',
          { classifierModel: classifierProbe.model || null }
        );
        if (classifierReading.ok) {
          return classifierReading.reading;
        }
        emitReject(
          requireAllCells ? 'classifier-missing-cell-digit' : 'classifier-insufficient-cell-digits',
          {
            found: classifierReading.found,
            required: minFound,
            ...metadata
          }
        );
      } else if (classifierProbe.reason !== 'disabled') {
        emitReject('classifier-unavailable', {
          reason: classifierProbe.reason,
          ...metadata
        });
      }
    }

    const digits = [];
    const cellConfidences = [];
    const cellDecodeModes = ['binary', 'soft', 'raw'];

    for (let i = 0; i < cellCanvases.length; i += 1) {
      if (setProgress) {
        setProgress(`Refining digits (${i + 1}/${cellCanvases.length})`);
      }
      let best = null;
      for (const mode of cellDecodeModes) {
        let cell = mode === 'raw' ? cellCanvases[i] : preprocessCanvas(cellCanvases[i], mode);
        cell = scaleCanvas(cell, OCR_CONFIG.minDigitWidth);
        const { data } = await worker.recognize(cell);
        const picked = pickDigit(data);
        if (picked && (!best || picked.confidence > best.confidence)) {
          best = picked;
        }
      }

      if (best) {
        digits.push(best.digit);
        cellConfidences.push(best.confidence);
      } else {
        digits.push('');
        cellConfidences.push(0);
      }
    }

    const tesseractReading = buildReading(digits, cellConfidences, 'tesseract');
    if (!tesseractReading.ok) {
      emitReject(requireAllCells ? 'missing-cell-digit' : 'insufficient-cell-digits', {
        found: tesseractReading.found,
        required: minFound,
        ...metadata
      });
      return null;
    }
    return tesseractReading.reading;
  };

  const finalizeReading = (reading) => {
    if (!reading) {
      return null;
    }
    if (options.roiMode && reading.value.length !== OCR_CONFIG.preferredDigits) {
      emitReject('roi-non4-output', { value: reading.value });
      return null;
    }

    const tunedConfidence = clamp(reading.averageConfidence + reading.foundRatio * 20, 0, 100);
    const bonus = reading.foundRatio === 1 ? 0.2 : 0.04;
    const score = scoreCandidate({ value: reading.value, confidence: tunedConfidence, areaRatio: 0.28 }) + bonus;

    return {
      value: reading.value,
      confidence: tunedConfidence,
      areaRatio: 0.28,
      score: clamp(score, 0, 0.99),
      cellDigits: reading.cellDigits || [],
      cellConfidences: reading.cellConfidences || [],
      decoder: reading.decoder || 'tesseract',
      classifierModel: reading.classifierModel || null,
      variantIndex: Number.isFinite(reading.variantIndex) ? reading.variantIndex : null,
      overlap: Number.isFinite(reading.overlap) ? reading.overlap : null,
      orientation: Number.isFinite(reading.orientation) ? reading.orientation : null,
      deskewAngle: Number.isFinite(reading.deskewAngle) ? reading.deskewAngle : null
    };
  };

  if (options.roiMode) {
    const overlap = Number.isFinite(roiDeterministic.cellOverlap) ? roiDeterministic.cellOverlap : 0.03;
    const requireAllCells = roiDeterministic.requireAllCells !== false;
    const roiNormalized = normalizeRoiStripCanvas(source);
    if (!roiNormalized || !roiNormalized.canvas) {
      return null;
    }
    const orientationVariants = [
      { orientation: 0, canvas: roiNormalized.canvas },
      { orientation: 180, canvas: rotateCanvas(roiNormalized.canvas, 180) }
    ];
    let best = null;

    for (let i = 0; i < orientationVariants.length; i += 1) {
      const variant = orientationVariants[i];
      const cellCanvases = splitIntoCells(variant.canvas, OCR_CONFIG.digitCellCount, overlap);
      if (!hasValidCellGeometry(cellCanvases, {
        mode: 'roi-deterministic',
        overlap,
        orientation: variant.orientation
      })) {
        continue;
      }
      const reading = await decodeCells(
        cellCanvases,
        {
          variantIndex: i,
          overlap,
          orientation: variant.orientation,
          deskewAngle: roiNormalized.deskewAngle
        },
        { requireAllCells }
      );
      const finalized = finalizeReading(reading);
      if (!finalized) {
        continue;
      }
      if (
        !best
        || finalized.score > best.score
        || (finalized.score === best.score && finalized.confidence > best.confidence)
      ) {
        best = finalized;
      }
    }

    return best;
  }

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

  for (let variantIndex = 0; variantIndex < variants.length; variantIndex += 1) {
    const variant = variants[variantIndex];
    if (!hasValidCandidateGeometry(variant, { mode: 'fallback', variantIndex })) {
      continue;
    }
    for (const overlap of [0.03, OCR_CONFIG.digitCellOverlap]) {
      const cellCanvases = splitIntoCells(variant, OCR_CONFIG.digitCellCount, overlap);
      if (!hasValidCellGeometry(cellCanvases, { mode: 'fallback', variantIndex, overlap })) {
        continue;
      }
      const reading = await decodeCells(
        cellCanvases,
        { variantIndex, overlap },
        { requireAllCells: false }
      );
      if (!reading) {
        continue;
      }
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
    emitReject('fallback-no-reading');
    return null;
  }

  return finalizeReading(bestReading);
};

export { getWorker, selectBestReading, readDigitsByCells };
