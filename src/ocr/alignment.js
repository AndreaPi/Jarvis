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
  const uniqueAngles = [0, 180, 90, 270];
  const candidates = [];

  uniqueAngles.forEach((angle) => {
    const rotated = angle === 0 ? source : rotateCanvas(source, angle);

    let primaryCanvas = tightenCropByInk(rotated, 0.12);
    primaryCanvas = scaleCanvas(primaryCanvas, OCR_CONFIG.minScaleWidth);
    if (hasInk(primaryCanvas)) {
      candidates.push({ canvas: primaryCanvas, label: `roi-${angle}-primary` });
    }

    const edgeRect = findDigitWindowByEdges(rotated);
    if (edgeRect) {
      let edgeCanvas = cropCanvas(rotated, edgeRect);
      edgeCanvas = tightenCropByInk(edgeCanvas, 0.16);
      edgeCanvas = scaleCanvas(edgeCanvas, OCR_CONFIG.minScaleWidth);
      if (hasInk(edgeCanvas)) {
        candidates.push({ canvas: edgeCanvas, label: `roi-${angle}-edge` });
      }
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
  const meterCrop = cropCenterSquare(source, OCR_CONFIG.meterCropScale);
  const rotations = [0, 90, 180, 270];
  const candidates = [...alignedCandidates];
  const gridCrops = [
    { name: 'grid-a', x: 0.0, y: 0.12, width: 0.6, height: 0.26 },
    { name: 'grid-b', x: 0.02, y: 0.2, width: 0.6, height: 0.26 },
    { name: 'grid-c', x: 0.05, y: 0.28, width: 0.6, height: 0.26 },
    { name: 'grid-d', x: 0.0, y: 0.18, width: 0.55, height: 0.3 },
    { name: 'grid-e', x: 0.06, y: 0.16, width: 0.64, height: 0.28 },
    { name: 'grid-f', x: 0.1, y: 0.14, width: 0.7, height: 0.24 }
  ];

  rotations.forEach((angle) => {
    const rotated = rotateCanvas(meterCrop, angle);
    const beforeCount = candidates.length;

    const edgeRect = findDigitWindowByEdges(rotated);
    if (edgeRect) {
      let edgeCanvas = cropCanvas(rotated, edgeRect);
      edgeCanvas = tightenCropByInk(edgeCanvas, 0.25);
      edgeCanvas = scaleCanvas(edgeCanvas, OCR_CONFIG.minScaleWidth);
      if (hasInk(edgeCanvas)) {
        candidates.push({ canvas: edgeCanvas, label: `${angle}-edge` });
      }
    }

    gridCrops.forEach((crop) => {
      const rect = {
        x: rotated.width * crop.x,
        y: rotated.height * crop.y,
        width: rotated.width * crop.width,
        height: rotated.height * crop.height
      };
      let gridCanvas = cropCanvas(rotated, rect);
      gridCanvas = scaleCanvas(gridCanvas, OCR_CONFIG.minScaleWidth);
      if (hasInk(gridCanvas)) {
        candidates.push({ canvas: gridCanvas, label: `${angle}-${crop.name}` });
      }
    });

    OCR_CONFIG.digitCrops.forEach((crop) => {
      const rect = {
        x: rotated.width * crop.x,
        y: rotated.height * crop.y,
        width: rotated.width * crop.width,
        height: rotated.height * crop.height
      };
      let digitCanvas = cropCanvas(rotated, rect);
      digitCanvas = tightenCropByInk(digitCanvas, 0.2);
      digitCanvas = scaleCanvas(digitCanvas, OCR_CONFIG.minScaleWidth);
      if (hasInk(digitCanvas)) {
        candidates.push({ canvas: digitCanvas, label: `${angle}-${crop.name}` });
      }
    });

    if (candidates.length === beforeCount) {
      const fallbackRect = {
        x: rotated.width * 0.02,
        y: rotated.height * 0.18,
        width: rotated.width * 0.7,
        height: rotated.height * 0.32
      };
      let fallbackCanvas = cropCanvas(rotated, fallbackRect);
      fallbackCanvas = scaleCanvas(fallbackCanvas, OCR_CONFIG.minScaleWidth);
      candidates.push({ canvas: fallbackCanvas, label: `${angle}-fallback` });
    }
  });

  return candidates;
};

export { buildDigitCandidates };
