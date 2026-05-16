import { OCR_CONFIG } from './config.js';
import {
  clamp,
  splitIntoCells,
  cropCanvas,
  tightenCropByInk,
  rotateCanvas,
  normalizeAngle
} from './canvas-utils.js';
import { predictDigitCells } from './digit-classifier.js';

const readDigitsByCells = async (source, setProgress, options = {}) => {
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

  const smoothProjection = (values, radius) => {
    const safeRadius = Math.max(0, Math.round(radius));
    if (!Array.isArray(values) || !values.length || safeRadius <= 0) {
      return Array.isArray(values) ? [...values] : [];
    }
    return values.map((_, index) => {
      const start = Math.max(0, index - safeRadius);
      const end = Math.min(values.length - 1, index + safeRadius);
      let total = 0;
      for (let i = start; i <= end; i += 1) {
        total += values[i];
      }
      return total / Math.max(1, end - start + 1);
    });
  };

  const buildBandColumnProjection = (canvas, bandY, bandHeight) => {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const { width, height } = canvas;
    const startY = Math.max(0, Math.min(height - 1, Math.round(bandY)));
    const safeHeight = Math.max(1, Math.min(height - startY, Math.round(bandHeight)));
    const data = ctx.getImageData(0, startY, width, safeHeight).data;
    const columns = new Array(width).fill(0);

    for (let y = 0; y < safeHeight; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const idx = (y * width + x) * 4;
        const lum = data[idx] * 0.2126 + data[idx + 1] * 0.7152 + data[idx + 2] * 0.0722;
        columns[x] += 255 - lum;
      }
    }

    return columns;
  };

  const buildRegisterTightenedCanvases = (canvas) => {
    if (roiDeterministic.tightenRegister === false || !canvas) {
      return [];
    }
    const { width, height } = canvas;
    if (width < minCandidateWidth || height < minCandidateHeight) {
      return [];
    }

    const verticalBandRatio = Number.isFinite(roiDeterministic.registerVerticalBandRatio)
      ? clamp(roiDeterministic.registerVerticalBandRatio, 0.35, 1)
      : 0.68;
    const targetAspect = Number.isFinite(roiDeterministic.registerTargetAspect)
      ? clamp(roiDeterministic.registerTargetAspect, 2.2, 4.8)
      : 3.2;
    const minCropRatio = Number.isFinite(roiDeterministic.registerMinCropRatio)
      ? clamp(roiDeterministic.registerMinCropRatio, 0.25, 0.9)
      : 0.45;
    const maxCropRatio = Number.isFinite(roiDeterministic.registerMaxCropRatio)
      ? clamp(roiDeterministic.registerMaxCropRatio, minCropRatio, 0.96)
      : 0.82;
    const paddingRatio = Number.isFinite(roiDeterministic.registerPaddingRatio)
      ? clamp(roiDeterministic.registerPaddingRatio, 1, 1.18)
      : 1.02;

    const { rows } = buildInkProjection(canvas);
    const bandHeight = Math.max(
      minCellHeight,
      Math.min(height, Math.round(height * verticalBandRatio))
    );
    const bandY = findMaxInkWindowStart(rows, bandHeight);
    const rawColumns = buildBandColumnProjection(canvas, bandY, bandHeight);
    const columns = smoothProjection(rawColumns, Math.max(2, Math.round(width * 0.012)));
    const desiredWidth = clamp(
      Math.round(bandHeight * targetAspect),
      Math.round(width * minCropRatio),
      Math.round(width * maxCropRatio)
    );
    const configuredRatios = Array.isArray(roiDeterministic.registerCropRatios)
      ? roiDeterministic.registerCropRatios
        .filter((ratio) => Number.isFinite(ratio))
        .map((ratio) => clamp(ratio, minCropRatio, maxCropRatio))
      : [];
    const requestedWidths = [
      desiredWidth,
      ...configuredRatios.map((ratio) => Math.round(width * ratio))
    ]
      .map((cropWidth) => clamp(
        Math.round(cropWidth * paddingRatio),
        Math.round(width * minCropRatio),
        Math.round(width * maxCropRatio)
      ))
      .filter((cropWidth) => cropWidth >= minCandidateWidth && cropWidth < width * 0.94);

    const variants = [];
    const seenKeys = new Set();
    requestedWidths.forEach((cropWidth) => {
      const coreStart = findMaxInkWindowStart(columns, cropWidth);
      const x = clamp(
        coreStart,
        0,
        Math.max(0, width - cropWidth)
      );
      const cropRatio = cropWidth / Math.max(1, width);
      if (cropRatio < minCropRatio || cropRatio > maxCropRatio) {
        return;
      }
      const key = `${Math.round(x / 4)}:${Math.round(cropWidth / 4)}`;
      if (seenKeys.has(key)) {
        return;
      }
      seenKeys.add(key);

      const tightened = cropCanvas(canvas, {
        x,
        y: 0,
        width: cropWidth,
        height
      });
      if (!tightened || tightened.width >= width * 0.94) {
        return;
      }

      emitReject('roi-register-tighten-candidate', {
        originalWidth: width,
        cropX: x,
        cropWidth: tightened.width,
        cropRatio: Number(cropRatio.toFixed(3)),
        bandY,
        bandHeight,
        targetAspect
      });
      variants.push({
        canvas: tightened,
        cropMode: `tight-register-${cropRatio.toFixed(2)}`,
        cropX: x,
        cropWidth: tightened.width,
        cropRatio
      });
    });

    return variants;
  };

  const getGeometryScoreAdjustment = (reading) => {
    if (!reading) {
      return 0;
    }
    const cropRatio = Number.isFinite(reading.cropRatio) ? reading.cropRatio : 1;
    const cropMode = typeof reading.cropMode === 'string' ? reading.cropMode : '';
    if (!cropMode.startsWith('tight-register')) {
      return cropRatio > 0.92 ? -0.05 : 0;
    }
    if (cropRatio >= 0.56 && cropRatio <= 0.76) {
      return 0.08;
    }
    if (cropRatio > 0.76 && cropRatio <= 0.84) {
      return 0.04;
    }
    if (cropRatio >= 0.48 && cropRatio < 0.56) {
      return 0.02;
    }
    return -0.02;
  };

  const normalizeRoiStripCanvas = (canvas) => {
    const tightenRatio = Number.isFinite(roiDeterministic.tightenInk) ? roiDeterministic.tightenInk : 0.08;
    const minStripAspect = Number.isFinite(roiDeterministic.minStripAspect) ? roiDeterministic.minStripAspect : 1.8;
    const maxStripAspect = Number.isFinite(roiDeterministic.maxStripAspect) ? roiDeterministic.maxStripAspect : 6.5;
    const normalizeWidth = Number.isFinite(roiDeterministic.normalizeWidth) ? roiDeterministic.normalizeWidth : OCR_CONFIG.minScaleWidth;
    const orientationNormalized = normalizeRoiOrientation(canvas, tightenRatio);
    let normalized = orientationNormalized.canvas;
    let majorAxisRotation = 0;
    if (normalized && normalized.height > normalized.width) {
      normalized = rotateCanvas(normalized, 90);
      majorAxisRotation = 90;
    }
    if (!hasValidCandidateGeometry(normalized, { mode: 'roi-initial' })) {
      return null;
    }

    let aspect = normalized.width / Math.max(1, normalized.height);
    if (aspect < minStripAspect) {
      const targetHeight = Math.max(
        minCandidateHeight,
        Math.min(normalized.height, Math.round(normalized.width / minStripAspect))
      );
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
      const targetWidth = Math.max(
        minCandidateWidth,
        Math.min(normalized.width, Math.round(normalized.height * maxStripAspect))
      );
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
    if (normalized && normalized.height > normalized.width) {
      normalized = rotateCanvas(normalized, 90);
      majorAxisRotation = normalizeAngle(majorAxisRotation + 90);
    }
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
        : 0,
      majorAxisRotation
    };
  };

  const decodeCells = async (cellCanvases, metadata = {}, decodeOptions = {}) => {
    const requireAllCells = !!decodeOptions.requireAllCells;
    const minFound = requireAllCells ? cellCanvases.length : OCR_CONFIG.minDigits;
    const classifier = OCR_CONFIG.digitClassifier || {};

    if (!classifier.enabled || !classifier.endpoint) {
      emitReject('classifier-disabled', metadata);
      return null;
    }

    if (setProgress) {
      setProgress('Classifying digit sections...');
    }
    const classifierProbe = await predictDigitCells(cellCanvases, classifier, { ignoreCooldown: true });
    if (!classifierProbe.ok) {
      const rejectReason = classifierProbe.reason === 'disabled'
        ? 'classifier-disabled'
        : 'classifier-unavailable';
      emitReject(rejectReason, {
        classifierReason: classifierProbe.reason || null,
        ...metadata
      });
      return null;
    }

    const digits = classifierProbe.predictions.map((item) => (item && item.accepted ? item.digit : ''));
    const cellConfidences = classifierProbe.predictions.map((item) => {
      if (!item || !item.accepted || !Number.isFinite(item.confidence)) {
        return 0;
      }
      return clamp(item.confidence * 100, 0, 100);
    });

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
      emitReject(
        requireAllCells ? 'classifier-missing-cell-digit' : 'classifier-insufficient-cell-digits',
        {
          found,
          required: minFound,
          ...metadata
        }
      );
      return null;
    }

    return {
      value,
      foundRatio: found / cellCanvases.length,
      averageConfidence: confidenceTotal / Math.max(found, 1),
      cellDigits: digits,
      cellConfidences,
      variantIndex: metadata.variantIndex,
      overlap: metadata.overlap,
      orientation: Number.isFinite(metadata.orientation) ? metadata.orientation : null,
      deskewAngle: Number.isFinite(metadata.deskewAngle) ? metadata.deskewAngle : null,
      cropMode: metadata.cropMode || null,
      cropX: Number.isFinite(metadata.cropX) ? metadata.cropX : null,
      cropWidth: Number.isFinite(metadata.cropWidth) ? metadata.cropWidth : null,
      cropRatio: Number.isFinite(metadata.cropRatio) ? metadata.cropRatio : null,
      decoder: 'digit-classifier',
      classifierModel: classifierProbe.model || null
    };
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
    const normalizedConfidence = clamp(tunedConfidence / 100, 0, 1);
    const baseScore = clamp(0.42 + normalizedConfidence * 0.46 + (reading.foundRatio === 1 ? 0.08 : 0.02), 0, 0.99);
    const geometryScoreAdjustment = getGeometryScoreAdjustment(reading);

    return {
      value: reading.value,
      confidence: tunedConfidence,
      areaRatio: 0.28,
      score: baseScore,
      baseScore,
      geometryScoreAdjustment,
      cellDigits: reading.cellDigits || [],
      cellConfidences: reading.cellConfidences || [],
      decoder: reading.decoder || 'digit-classifier',
      classifierModel: reading.classifierModel || null,
      variantIndex: Number.isFinite(reading.variantIndex) ? reading.variantIndex : null,
      overlap: Number.isFinite(reading.overlap) ? reading.overlap : null,
      orientation: Number.isFinite(reading.orientation) ? reading.orientation : null,
      deskewAngle: Number.isFinite(reading.deskewAngle) ? reading.deskewAngle : null,
      cropMode: reading.cropMode || null,
      cropX: Number.isFinite(reading.cropX) ? reading.cropX : null,
      cropWidth: Number.isFinite(reading.cropWidth) ? reading.cropWidth : null,
      cropRatio: Number.isFinite(reading.cropRatio) ? reading.cropRatio : null
    };
  };

  if (!options.roiMode) {
    emitReject('classifier-non-roi-mode');
    return null;
  }

  const overlap = Number.isFinite(roiDeterministic.cellOverlap) ? roiDeterministic.cellOverlap : 0.03;
  const requireAllCells = roiDeterministic.requireAllCells !== false;
  const roiNormalized = normalizeRoiStripCanvas(source);
  if (!roiNormalized || !roiNormalized.canvas) {
    return null;
  }
  const baseOrientation = Number.isFinite(roiNormalized.majorAxisRotation)
    ? normalizeAngle(roiNormalized.majorAxisRotation)
    : 0;
  const tryOppositeOrientation = roiDeterministic.tryOppositeOrientation === true;
  const selectRegisterVariants = roiDeterministic.selectRegisterVariants === true;
  const baseVariants = [
    {
      canvas: roiNormalized.canvas,
      cropMode: 'full-strip',
      cropX: 0,
      cropWidth: roiNormalized.canvas.width,
      cropRatio: 1
    }
  ];
  baseVariants.push(...buildRegisterTightenedCanvases(roiNormalized.canvas));
  const orientationVariants = baseVariants.map((variant) => ({
    ...variant,
    orientation: baseOrientation
  }));
  if (tryOppositeOrientation) {
    baseVariants.forEach((variant) => {
      orientationVariants.push({
        ...variant,
        orientation: normalizeAngle(baseOrientation + 180),
        canvas: rotateCanvas(variant.canvas, 180)
      });
    });
  }
  let best = null;
  const finalizedVariants = [];

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
        deskewAngle: roiNormalized.deskewAngle,
        cropMode: variant.cropMode,
        cropX: variant.cropX,
        cropWidth: variant.cropWidth,
        cropRatio: variant.cropRatio
      },
      { requireAllCells }
    );
    const finalized = finalizeReading(reading);
    if (!finalized) {
      continue;
    }
    const finalizedWithDecodeCanvas = {
      ...finalized,
      decodedStripCanvas: variant.canvas,
      decodedCellCanvases: cellCanvases
    };
    finalizedVariants.push(finalizedWithDecodeCanvas);
    const eligibleForPrimarySelection = selectRegisterVariants || variant.cropMode === 'full-strip';
    if (!eligibleForPrimarySelection) {
      continue;
    }
    if (
      !best
      || finalizedWithDecodeCanvas.score > best.score
      || (
        finalizedWithDecodeCanvas.score === best.score
        && finalizedWithDecodeCanvas.confidence > best.confidence
      )
    ) {
      best = finalizedWithDecodeCanvas;
    }
  }

  if (best && finalizedVariants.length) {
    best.diagnosticVariants = finalizedVariants
      .map((candidate) => ({
        value: candidate.value,
        score: Number.isFinite(candidate.score) ? candidate.score : null,
        baseScore: Number.isFinite(candidate.baseScore) ? candidate.baseScore : null,
        geometryScoreAdjustment: Number.isFinite(candidate.geometryScoreAdjustment)
          ? candidate.geometryScoreAdjustment
          : null,
        confidence: Number.isFinite(candidate.confidence) ? candidate.confidence : null,
        cellDigits: Array.isArray(candidate.cellDigits) ? [...candidate.cellDigits] : [],
        cellConfidences: Array.isArray(candidate.cellConfidences) ? [...candidate.cellConfidences] : [],
        cropMode: candidate.cropMode || null,
        cropRatio: Number.isFinite(candidate.cropRatio) ? candidate.cropRatio : null,
        orientation: Number.isFinite(candidate.orientation) ? candidate.orientation : null
      }))
      .sort((a, b) => (b.score ?? -1) - (a.score ?? -1))
      .slice(0, 6);
  }

  return best;
};

export { readDigitsByCells };
