import { OCR_CONFIG } from './config.js';
import {
  rotateCanvas,
  scaleCanvas,
  cropCanvas,
  findDigitWindowByEdges,
  preprocessCanvas
} from './canvas-utils.js';

const buildNeuralRoiCandidates = (source, debugSession, addDebugStageFn = () => {}) => {
  const roiDeterministic = OCR_CONFIG.roiDeterministic || {};
  const normalizeWidth = Number.isFinite(roiDeterministic.normalizeWidth)
    ? roiDeterministic.normalizeWidth
    : OCR_CONFIG.minScaleWidth;
  const useEdgeCandidates = roiDeterministic.useEdgeCandidates !== false;
  const debugWordMode = Array.isArray(roiDeterministic.wordPassModes) && roiDeterministic.wordPassModes.length
    ? roiDeterministic.wordPassModes.find((mode) => mode === 'soft' || mode === 'binary' || mode === 'raw') || 'raw'
    : 'raw';

  const configuredAngles = Array.isArray(roiDeterministic.primaryAngles) && roiDeterministic.primaryAngles.length
    ? roiDeterministic.primaryAngles
    : [90, 270];
  const angles = configuredAngles
    .map((angle) => Number.parseInt(angle, 10))
    .filter((angle, index, values) => Number.isFinite(angle) && values.indexOf(angle) === index);
  const candidates = [];
  let debugStripSource = null;
  const baseCandidates = [];
  const edgeContextPaddingX = Number.isFinite(roiDeterministic.edgeContextPaddingX)
    ? Math.max(0, Math.min(0.5, roiDeterministic.edgeContextPaddingX))
    : 0.18;
  const edgeContextPaddingY = Number.isFinite(roiDeterministic.edgeContextPaddingY)
    ? Math.max(0, Math.min(0.4, roiDeterministic.edgeContextPaddingY))
    : 0.08;
  const edgeContextShiftRatios = Array.isArray(roiDeterministic.edgeContextShiftRatios)
    ? roiDeterministic.edgeContextShiftRatios
      .filter((ratio) => Number.isFinite(ratio))
      .map((ratio) => Math.max(-0.35, Math.min(0.35, ratio)))
    : [-0.08, 0, 0.08];
  const edgeContextMaxVariantsPerAngle = Number.isFinite(roiDeterministic.edgeContextMaxVariantsPerAngle)
    ? Math.max(0, Math.min(12, Math.round(roiDeterministic.edgeContextMaxVariantsPerAngle)))
    : 3;

  const pushCandidate = (canvas, label) => {
    if (!canvas) {
      return;
    }
    const normalized = scaleCanvas(canvas, normalizeWidth);
    if (!normalized || normalized.width < 24 || normalized.height < 16) {
      return;
    }
    candidates.push({ canvas: normalized, label });
    if (!debugStripSource) {
      debugStripSource = canvas;
    }
  };

  const normalizeCropRect = (canvas, rect) => {
    if (!canvas || !rect) {
      return null;
    }
    const x = Math.max(0, Math.min(canvas.width - 1, Math.round(rect.x)));
    const y = Math.max(0, Math.min(canvas.height - 1, Math.round(rect.y)));
    const width = Math.max(1, Math.min(canvas.width - x, Math.round(rect.width)));
    const height = Math.max(1, Math.min(canvas.height - y, Math.round(rect.height)));
    return { x, y, width, height };
  };

  const pushEdgeContextCandidates = (rotated, edgeRect, angle) => {
    if (!rotated || !edgeRect || edgeContextMaxVariantsPerAngle <= 0) {
      return;
    }
    const paddedWidth = Math.min(
      rotated.width,
      edgeRect.width * (1 + edgeContextPaddingX * 2)
    );
    const paddedHeight = Math.min(
      rotated.height,
      edgeRect.height * (1 + edgeContextPaddingY * 2)
    );
    if (paddedWidth <= edgeRect.width + 1 && paddedHeight <= edgeRect.height + 1) {
      return;
    }

    const centerX = edgeRect.x + edgeRect.width * 0.5;
    const centerY = edgeRect.y + edgeRect.height * 0.5;
    const seen = new Set();
    let emitted = 0;
    edgeContextShiftRatios.forEach((shiftRatio) => {
      if (emitted >= edgeContextMaxVariantsPerAngle) {
        return;
      }
      const rect = normalizeCropRect(rotated, {
        x: centerX - paddedWidth * 0.5 + edgeRect.width * shiftRatio,
        y: centerY - paddedHeight * 0.5,
        width: paddedWidth,
        height: paddedHeight
      });
      if (!rect) {
        return;
      }
      const key = `${Math.round(rect.x / 3)}:${Math.round(rect.y / 3)}:${Math.round(rect.width / 3)}:${Math.round(rect.height / 3)}`;
      if (seen.has(key)) {
        return;
      }
      seen.add(key);
      emitted += 1;
      const shiftPercent = Math.round(Math.abs(shiftRatio) * 100);
      const suffix = shiftRatio === 0
        ? 'center'
        : (shiftRatio > 0 ? `right${shiftPercent}` : `left${shiftPercent}`);
      pushCandidate(cropCanvas(rotated, rect), `roi-${angle}-edge-context-${suffix}`);
    });
  };

  angles.forEach((angle) => {
    const rotated = angle === 0 ? source : rotateCanvas(source, angle);

    if (useEdgeCandidates) {
      const edgeRect = findDigitWindowByEdges(rotated);
      if (edgeRect) {
        const edgeCrop = cropCanvas(rotated, edgeRect);
        pushCandidate(edgeCrop, `roi-${angle}-edge`);
        pushEdgeContextCandidates(rotated, edgeRect, angle);
      }
    }

    const normalized = scaleCanvas(rotated, normalizeWidth);
    if (normalized && normalized.width >= 24 && normalized.height >= 16) {
      baseCandidates.push({ canvas: normalized, label: `roi-${angle}-base` });
      if (!debugStripSource) {
        debugStripSource = rotated;
      }
    }
  });

  if (baseCandidates.length) {
    candidates.push(...baseCandidates);
  }

  if (!candidates.length) {
    const fallback = scaleCanvas(source, normalizeWidth);
    candidates.push({ canvas: fallback, label: 'roi-base-fallback' });
    debugStripSource = source;
  }

  if (debugSession) {
    const stripPreview = preprocessCanvas(debugStripSource || source, 'soft');
    addDebugStageFn(debugSession, '5. detected strip crop', stripPreview);
    const ocrPreview = debugWordMode === 'raw'
      ? candidates[0].canvas
      : preprocessCanvas(candidates[0].canvas, debugWordMode);
    addDebugStageFn(debugSession, '6a. OCR input candidate (initial preview)', ocrPreview);
  }

  return candidates;
};

const buildDigitCandidates = (source, debugSession = null, addDebugStageFn = () => {}) => {
  return buildNeuralRoiCandidates(source, debugSession, addDebugStageFn);
};

export { buildDigitCandidates };
