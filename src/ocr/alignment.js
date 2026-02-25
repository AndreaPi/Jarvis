import { OCR_CONFIG } from './config.js';
import {
  rotateCanvas,
  scaleCanvas,
  tightenCropByInk,
  preprocessCanvas
} from './canvas-utils.js';

const buildNeuralRoiCandidates = (source, debugSession, addDebugStageFn = () => {}) => {
  const roiDeterministic = OCR_CONFIG.roiDeterministic || {};
  const tightenInk = Number.isFinite(roiDeterministic.tightenInk)
    ? roiDeterministic.tightenInk
    : 0.08;
  const normalizeWidth = Number.isFinite(roiDeterministic.normalizeWidth)
    ? roiDeterministic.normalizeWidth
    : OCR_CONFIG.minScaleWidth;

  const refinedCrop = tightenCropByInk(source, tightenInk);
  const softPreview = preprocessCanvas(refinedCrop, 'soft');

  const angles = [0, 90, 180, 270];
  const candidates = [];

  angles.forEach((angle) => {
    const rotated = angle === 0 ? refinedCrop : rotateCanvas(refinedCrop, angle);
    const normalized = scaleCanvas(rotated, normalizeWidth);
    if (normalized && normalized.width >= 24 && normalized.height >= 16) {
      candidates.push({ canvas: normalized, label: `roi-${angle}-tight` });
    }
  });

  if (!candidates.length) {
    const fallback = scaleCanvas(source, normalizeWidth);
    candidates.push({ canvas: fallback, label: 'roi-raw-fallback' });
  }

  if (debugSession) {
    addDebugStageFn(debugSession, '5. detected strip crop', softPreview);
    addDebugStageFn(debugSession, '6. OCR input candidate', preprocessCanvas(candidates[0].canvas, 'binary'));
  }

  return candidates;
};

const buildDigitCandidates = (source, debugSession = null, addDebugStageFn = () => {}) => {
  return buildNeuralRoiCandidates(source, debugSession, addDebugStageFn);
};

export { buildDigitCandidates };
