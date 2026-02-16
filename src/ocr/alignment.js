import { OCR_CONFIG, ALIGNMENT_CONFIG, DEBUG_CONFIG } from './config.js';
import {
  getRectFromNormalized,
  drawOverlayCanvas,
  cropCenterSquare,
  cropCanvas,
  rotateCanvas,
  scaleCanvas,
  tightenCropByInk,
  hasInk,
  adjustRectAroundCenter,
  findDigitWindowByEdges
} from './canvas-utils.js';
import { detectMeterFace, scoreCanonicalRotation } from './face-detection.js';
import { detectDigitStripInBand, buildStripTopKOverlay } from './strip-detection.js';

const buildAlignedDigitCandidates = (source, debugSession, addDebugStageFn = () => {}) => {
  const meterCrop = cropCenterSquare(source, OCR_CONFIG.meterCropScale);
  const detection = detectMeterFace(meterCrop);
  const faceCanvas = cropCanvas(meterCrop, detection.faceRect);
  const rotations = [0, 90, 180, 270];
  const rotationCandidates = rotations.map((angle) => {
    const canvas = rotateCanvas(faceCanvas, angle);
    const canonicalScore = scoreCanonicalRotation(canvas);
    return { angle, canvas, canonicalScore };
  });

  let best = rotationCandidates[0];
  rotationCandidates.forEach((candidate) => {
    if (!best || candidate.canonicalScore > best.canonicalScore) {
      best = candidate;
    }
  });

  const canonicalCircle = {
    cx: best.canvas.width * 0.5,
    cy: best.canvas.height * 0.5,
    radius: detection.circle.radius
  };
  const stripSearch = detectDigitStripInBand(best.canvas, canonicalCircle);
  const selectedStrip = stripSearch.selected;
  const stripOrientation = selectedStrip ? selectedStrip.orientation : 'horizontal';
  const windows = selectedStrip
    ? ALIGNMENT_CONFIG.stripWindows.map((settings) => ({
      name: settings.name,
      rect: adjustRectAroundCenter(best.canvas, selectedStrip.rect, settings)
    }))
    : ALIGNMENT_CONFIG.fallbackWindows.map((rect) => ({
      name: rect.name,
      rect: getRectFromNormalized(best.canvas, rect)
    }));

  const candidates = [];
  const geometry = OCR_CONFIG.geometry || {};
  const minCandidateWidth = Number.isFinite(geometry.minCandidateWidth) ? geometry.minCandidateWidth : 120;
  const minCandidateHeight = Number.isFinite(geometry.minCandidateHeight) ? geometry.minCandidateHeight : 28;
  const minStripAspect = Number.isFinite(geometry.minStripAspect) ? geometry.minStripAspect : 1.15;
  const maxStripAspect = Number.isFinite(geometry.maxStripAspect) ? geometry.maxStripAspect : 12;
  const isStripLike = (canvas) => {
    if (!canvas) {
      return false;
    }
    const width = canvas.width;
    const height = canvas.height;
    if (width < minCandidateWidth || height < minCandidateHeight) {
      return false;
    }
    const aspect = width / Math.max(1, height);
    return aspect >= minStripAspect && aspect <= maxStripAspect;
  };

  windows.forEach((window) => {
    const baseCrop = cropCanvas(best.canvas, window.rect);
    const variantAngles = stripOrientation === 'vertical' ? [90, 270] : [0];

    variantAngles.forEach((variantAngle) => {
      let digitCanvas = variantAngle === 0 ? baseCrop : rotateCanvas(baseCrop, variantAngle);
      digitCanvas = tightenCropByInk(digitCanvas, 0.08);
      digitCanvas = scaleCanvas(digitCanvas, OCR_CONFIG.minScaleWidth);
      if (!hasInk(digitCanvas) && window.name !== 'strip-main' && window.name !== 'fallback-main') {
        return;
      }
      if (!isStripLike(digitCanvas)) {
        return;
      }
      candidates.push({
        canvas: digitCanvas,
        label: `aligned-${best.angle}-${window.name}${variantAngle ? `-r${variantAngle}` : ''}`
      });
    });
  });

  if (debugSession) {
    const meterOverlay = drawOverlayCanvas(meterCrop, [
      {
        type: 'circle',
        cx: detection.circle.cx,
        cy: detection.circle.cy,
        radius: detection.circle.radius,
        label: 'detected face'
      },
      {
        x: detection.faceRect.x,
        y: detection.faceRect.y,
        width: detection.faceRect.width,
        height: detection.faceRect.height,
        label: 'face crop'
      }
    ]);
    addDebugStageFn(debugSession, '1. face detection', meterOverlay);

    const shapes = [];
    shapes.push({
      x: stripSearch.searchBand.x,
      y: stripSearch.searchBand.y,
      width: stripSearch.searchBand.width,
      height: stripSearch.searchBand.height,
      label: 'search-band',
      color: '#f59e0b'
    });

    if (selectedStrip) {
      shapes.push({
        x: selectedStrip.rect.x,
        y: selectedStrip.rect.y,
        width: selectedStrip.rect.width,
        height: selectedStrip.rect.height,
        label: `detected-strip (${selectedStrip.orientation})`,
        color: '#22c55e'
      });
    }

    windows.forEach((item, index) => {
      shapes.push({
        x: item.rect.x,
        y: item.rect.y,
        width: item.rect.width,
        height: item.rect.height,
        label: item.name,
        color: DEBUG_CONFIG.colors[index % DEBUG_CONFIG.colors.length]
      });
    });

    const alignedOverlay = drawOverlayCanvas(best.canvas, shapes);
    addDebugStageFn(
      debugSession,
      `2. canonical (${best.angle}deg, ${selectedStrip ? 'strip' : 'fallback'})`,
      alignedOverlay
    );

    const topKOverlay = buildStripTopKOverlay(best.canvas, stripSearch.searchBand, stripSearch.topCandidates, selectedStrip?.rect);
    addDebugStageFn(debugSession, '3. strip score top-k', topKOverlay);
    addDebugStageFn(debugSession, '4. strip binary/edge map', stripSearch.decisionMaps.combinedCanvas);

    if (selectedStrip) {
      const stripCanvas = cropCanvas(best.canvas, selectedStrip.rect);
      addDebugStageFn(debugSession, '5. detected strip crop', stripCanvas);
    } else if (windows[0]) {
      const fallbackCanvas = cropCanvas(best.canvas, windows[0].rect);
      addDebugStageFn(debugSession, '5. fallback ROI crop', fallbackCanvas);
    }

    if (candidates[0]) {
      addDebugStageFn(debugSession, '6. OCR input candidate', candidates[0].canvas);
    }
  }

  return candidates;
};

const buildNeuralRoiCandidates = (source, debugSession, addDebugStageFn = () => {}) => {
  const uniqueAngles = [0, 90, 180, 270];
  const candidates = [];
  const isUsable = (canvas) => canvas && canvas.width >= 24 && canvas.height >= 16;

  uniqueAngles.forEach((angle) => {
    const rotated = angle === 0 ? source : rotateCanvas(source, angle);
    const rawCanvas = scaleCanvas(rotated, OCR_CONFIG.minScaleWidth);
    if (isUsable(rawCanvas)) {
      candidates.push({ canvas: rawCanvas, label: `roi-${angle}-raw` });
    }
  });

  if (!candidates.length) {
    let fallback = scaleCanvas(source, OCR_CONFIG.minScaleWidth);
    candidates.push({ canvas: fallback, label: 'roi-raw-fallback' });
  }

  if (debugSession) {
    addDebugStageFn(debugSession, '5. detected strip crop', source);
    addDebugStageFn(debugSession, '6. OCR input candidate', candidates[0].canvas);
  }

  return candidates;
};

const buildDigitCandidates = (source, debugSession = null, addDebugStageFn = () => {}, options = {}) => {
  if (options && options.roiMode) {
    return buildNeuralRoiCandidates(source, debugSession, addDebugStageFn);
  }

  const alignedCandidates = buildAlignedDigitCandidates(source, debugSession, addDebugStageFn);
  if (alignedCandidates.length) {
    return alignedCandidates;
  }

  // Last-resort compact fallback: only one strip-like window per rotation.
  const meterCrop = cropCenterSquare(source, OCR_CONFIG.meterCropScale);
  const rotations = [0, 90, 180, 270];
  const fallbackCandidates = [];
  const fallbackRectNorm = ALIGNMENT_CONFIG.fallbackWindows[0] || { x: 0.08, y: 0.14, width: 0.48, height: 0.24 };
  const geometry = OCR_CONFIG.geometry || {};
  const minCandidateWidth = Number.isFinite(geometry.minCandidateWidth) ? geometry.minCandidateWidth : 120;
  const minCandidateHeight = Number.isFinite(geometry.minCandidateHeight) ? geometry.minCandidateHeight : 28;
  const minStripAspect = Number.isFinite(geometry.minStripAspect) ? geometry.minStripAspect : 1.15;
  const maxStripAspect = Number.isFinite(geometry.maxStripAspect) ? geometry.maxStripAspect : 12;
  const isStripLike = (canvas) => {
    if (!canvas) {
      return false;
    }
    const width = canvas.width;
    const height = canvas.height;
    if (width < minCandidateWidth || height < minCandidateHeight) {
      return false;
    }
    const aspect = width / Math.max(1, height);
    return aspect >= minStripAspect && aspect <= maxStripAspect;
  };

  rotations.forEach((angle) => {
    const rotated = rotateCanvas(meterCrop, angle);
    const fallbackRect = getRectFromNormalized(rotated, fallbackRectNorm);
    let fallbackCanvas = cropCanvas(rotated, fallbackRect);
    fallbackCanvas = tightenCropByInk(fallbackCanvas, 0.12);
    fallbackCanvas = scaleCanvas(fallbackCanvas, OCR_CONFIG.minScaleWidth);
    if (hasInk(fallbackCanvas) && isStripLike(fallbackCanvas)) {
      fallbackCandidates.push({ canvas: fallbackCanvas, label: `${angle}-fallback-compact` });
    }
  });

  if (fallbackCandidates.length) {
    return fallbackCandidates;
  }

  const edgeRect = findDigitWindowByEdges(meterCrop);
  if (edgeRect) {
    let edgeCanvas = cropCanvas(meterCrop, edgeRect);
    edgeCanvas = tightenCropByInk(edgeCanvas, 0.2);
    edgeCanvas = scaleCanvas(edgeCanvas, OCR_CONFIG.minScaleWidth);
    if (hasInk(edgeCanvas) && isStripLike(edgeCanvas)) {
      return [{ canvas: edgeCanvas, label: 'fallback-edge-compact' }];
    }
  }

  return [];
};

export { buildDigitCandidates };
