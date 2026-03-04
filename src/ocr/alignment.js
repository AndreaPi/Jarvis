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

  const angles = source.height > source.width
    ? [90, 270, 0, 180]
    : [0, 180, 90, 270];
  const candidates = [];
  let debugStripSource = null;

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

  angles.forEach((angle) => {
    const rotated = angle === 0 ? source : rotateCanvas(source, angle);

    if (useEdgeCandidates) {
      const edgeRect = findDigitWindowByEdges(rotated);
      if (edgeRect) {
        const edgeCrop = cropCanvas(rotated, edgeRect);
        pushCandidate(edgeCrop, `roi-${angle}-edge`);
      }
    }

    pushCandidate(rotated, `roi-${angle}-base`);
  });

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
    addDebugStageFn(debugSession, '6. OCR input candidate', ocrPreview);
  }

  return candidates;
};

const buildDigitCandidates = (source, debugSession = null, addDebugStageFn = () => {}) => {
  return buildNeuralRoiCandidates(source, debugSession, addDebugStageFn);
};

export { buildDigitCandidates };
