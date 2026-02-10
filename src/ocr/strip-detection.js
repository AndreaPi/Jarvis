import { ALIGNMENT_CONFIG } from './config.js';
import {
  drawOverlayCanvas,
  normalizeRectToCanvas,
  preprocessCanvas,
  clamp,
  analyzeRegion
} from './canvas-utils.js';

const getStrictStripSearchBand = (canvas, circle) => {
  const config = ALIGNMENT_CONFIG.stripSearchBand;
  const bandCenterX = circle.cx + circle.radius * config.offsetXFromCenterRadius;
  const bandCenterY = circle.cy + circle.radius * config.offsetYFromCenterRadius;
  const bandWidth = circle.radius * config.widthFromRadius;
  const bandHeight = circle.radius * config.heightFromRadius;
  return normalizeRectToCanvas(canvas, {
    x: bandCenterX - bandWidth * 0.5,
    y: bandCenterY - bandHeight * 0.5,
    width: bandWidth,
    height: bandHeight
  });
};

const getDigitStripCandidatesInBand = (canvas, bandRect) => {
  const templates = [];
  const xOffsets = [0.0, 0.06, 0.12, 0.18, 0.24];
  const yOffsets = [0.0, 0.08, 0.16, 0.24];
  const widths = [0.5, 0.62, 0.74, 0.86];
  const heights = [0.18, 0.24, 0.3, 0.36];

  xOffsets.forEach((x) => {
    yOffsets.forEach((y) => {
      widths.forEach((width) => {
        heights.forEach((height) => {
          if (x + width <= 0.98 && y + height <= 0.98) {
            templates.push({
              orientation: 'horizontal',
              rect: normalizeRectToCanvas(canvas, {
                x: bandRect.x + x * bandRect.width,
                y: bandRect.y + y * bandRect.height,
                width: width * bandRect.width,
                height: height * bandRect.height
              })
            });
          }
        });
      });
    });
  });

  return templates;
};

const buildStripDecisionMaps = (faceCanvas) => {
  const ctx = faceCanvas.getContext('2d', { willReadFrequently: true });
  const { width, height } = faceCanvas;
  const sourceImageData = ctx.getImageData(0, 0, width, height);
  const sourceData = sourceImageData.data;

  const binaryCanvas = preprocessCanvas(faceCanvas, 'binary');
  const binaryCtx = binaryCanvas.getContext('2d', { willReadFrequently: true });
  const binaryData = binaryCtx.getImageData(0, 0, width, height).data;

  const edgeCanvas = document.createElement('canvas');
  edgeCanvas.width = width;
  edgeCanvas.height = height;
  const edgeCtx = edgeCanvas.getContext('2d');
  const edgeImageData = edgeCtx.createImageData(width, height);
  const edgeData = edgeImageData.data;

  const luminanceAt = (x, y) => {
    const safeX = clamp(x, 0, width - 1);
    const safeY = clamp(y, 0, height - 1);
    const idx = (safeY * width + safeX) * 4;
    return sourceData[idx] * 0.2126 + sourceData[idx + 1] * 0.7152 + sourceData[idx + 2] * 0.0722;
  };

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const gx = Math.abs(luminanceAt(x + 1, y) - luminanceAt(x - 1, y));
      const gy = Math.abs(luminanceAt(x, y + 1) - luminanceAt(x, y - 1));
      const edge = clamp(gx * 0.9 + gy * 0.35, 0, 255) | 0;
      const idx = (y * width + x) * 4;
      edgeData[idx] = edge;
      edgeData[idx + 1] = edge;
      edgeData[idx + 2] = edge;
      edgeData[idx + 3] = 255;
    }
  }
  edgeCtx.putImageData(edgeImageData, 0, 0);

  const combinedCanvas = document.createElement('canvas');
  combinedCanvas.width = width * 2 + 8;
  combinedCanvas.height = height;
  const combinedCtx = combinedCanvas.getContext('2d');
  combinedCtx.fillStyle = '#0f172a';
  combinedCtx.fillRect(0, 0, combinedCanvas.width, combinedCanvas.height);
  combinedCtx.drawImage(binaryCanvas, 0, 0);
  combinedCtx.drawImage(edgeCanvas, width + 8, 0);

  return {
    sourceData,
    binaryData,
    edgeData,
    width,
    height,
    binaryCanvas,
    edgeCanvas,
    combinedCanvas
  };
};

const measureVerticalStrokePeriodicity = (edgeData, width, height, rect) => {
  const x0 = clamp(Math.floor(rect.x), 0, width - 1);
  const y0 = clamp(Math.floor(rect.y + rect.height * 0.08), 0, height - 1);
  const x1 = clamp(Math.ceil(rect.x + rect.width), x0 + 1, width);
  const y1 = clamp(Math.ceil(rect.y + rect.height * 0.92), y0 + 1, height);
  const profile = [];
  const rows = Math.max(1, y1 - y0);

  for (let x = x0; x < x1; x += 1) {
    let sum = 0;
    for (let y = y0; y < y1; y += 1) {
      const idx = (y * width + x) * 4;
      sum += edgeData[idx] / 255;
    }
    profile.push(sum / rows);
  }

  if (profile.length < 8) {
    return 0;
  }

  const smooth = profile.map((value, index) => {
    const left = profile[Math.max(0, index - 1)];
    const right = profile[Math.min(profile.length - 1, index + 1)];
    return (left + value + right) / 3;
  });
  const mean = smooth.reduce((sum, value) => sum + value, 0) / smooth.length;
  const variance = smooth.reduce((sum, value) => sum + (value - mean) ** 2, 0) / smooth.length;
  const std = Math.sqrt(variance);
  const peakThreshold = mean + std * 0.55;
  const highThreshold = mean + std * 0.24;

  let peaks = 0;
  for (let i = 1; i < smooth.length - 1; i += 1) {
    if (smooth[i] >= peakThreshold && smooth[i] >= smooth[i - 1] && smooth[i] >= smooth[i + 1]) {
      peaks += 1;
    }
  }

  let transitions = 0;
  let previousHigh = smooth[0] >= highThreshold;
  for (let i = 1; i < smooth.length; i += 1) {
    const currentHigh = smooth[i] >= highThreshold;
    if (currentHigh !== previousHigh) {
      transitions += 1;
      previousHigh = currentHigh;
    }
  }

  const peakScore = clamp(peaks / Math.max(2, smooth.length * 0.11), 0, 1);
  const transitionScore = clamp(transitions / Math.max(2, smooth.length * 0.26), 0, 1);
  return clamp(peakScore * 0.65 + transitionScore * 0.35, 0, 1);
};

const evaluateStripGates = (stats, rect, circle, periodicity) => {
  const gates = ALIGNMENT_CONFIG.stripGates;
  const aspect = rect.width / Math.max(1, rect.height);
  const centerX = rect.x + rect.width * 0.5;
  const centerY = rect.y + rect.height * 0.5;
  const distanceRatio = Math.hypot(centerX - circle.cx, centerY - circle.cy) / Math.max(1, circle.radius);
  const checks = {
    red: stats.redRatio <= gates.maxRedRatio,
    aspect: aspect >= gates.minAspect && aspect <= gates.maxAspect,
    distance: distanceRatio >= gates.minDistanceFromCenterRadius && distanceRatio <= gates.maxDistanceFromCenterRadius,
    periodicity: periodicity >= gates.minVerticalPeriodicity
  };

  return {
    checks,
    passed: Object.values(checks).every(Boolean),
    aspect,
    distanceRatio
  };
};

const scoreDigitStripRect = (stats, gateEval, periodicity) => {
  const gates = ALIGNMENT_CONFIG.stripGates;
  const aspectScore = 1 - Math.min(1, Math.abs(gateEval.aspect - 2.2) / 1.8);
  const distanceScore = 1 - Math.min(
    1,
    Math.abs(gateEval.distanceRatio - gates.expectedDistanceFromCenterRadius) / gates.distanceTolerance
  );
  const edgeScore = stats.edgeXRatio * 2.7 + stats.edgeYRatio * 0.9;
  const darkScore = 1 - Math.min(1, Math.abs(stats.darkRatio - 0.22) / 0.2);
  const brightnessScore = 1 - Math.min(1, Math.abs(stats.meanLum - 155) / 110);

  const gatePenalty = (
    (gateEval.checks.red ? 0 : -1.9)
    + (gateEval.checks.aspect ? 0 : -1.3)
    + (gateEval.checks.distance ? 0 : -1.4)
    + (gateEval.checks.periodicity ? 0 : -2.1)
  );

  return (
    edgeScore * 1.8
    + darkScore * 1.15
    + brightnessScore * 0.95
    + aspectScore * 1.35
    + distanceScore * 1.25
    + periodicity * 2.3
    - stats.redRatio * 4.8
    + gatePenalty
  );
};

const detectDigitStripInBand = (faceCanvas, circle) => {
  const decisionMaps = buildStripDecisionMaps(faceCanvas);
  const searchBand = getStrictStripSearchBand(faceCanvas, circle);
  const templates = getDigitStripCandidatesInBand(faceCanvas, searchBand);

  const scored = templates.map((template) => {
    const rect = normalizeRectToCanvas(faceCanvas, template.rect);
    const stats = analyzeRegion(decisionMaps.sourceData, decisionMaps.width, decisionMaps.height, rect);
    const periodicity = measureVerticalStrokePeriodicity(decisionMaps.edgeData, decisionMaps.width, decisionMaps.height, rect);
    const gateEval = evaluateStripGates(stats, rect, circle, periodicity);
    const score = scoreDigitStripRect(stats, gateEval, periodicity);
    return {
      rect,
      score,
      orientation: template.orientation,
      stats,
      periodicity,
      gateEval
    };
  }).sort((a, b) => b.score - a.score);

  const topCandidates = scored.slice(0, ALIGNMENT_CONFIG.stripDebugTopK);
  const bestAccepted = scored.find((candidate) => (
    candidate.gateEval.passed && candidate.score >= ALIGNMENT_CONFIG.stripGates.minAcceptedScore
  )) || null;

  return {
    searchBand,
    topCandidates,
    selected: bestAccepted,
    usedFallback: !bestAccepted,
    decisionMaps
  };
};

const buildStripTopKOverlay = (canvas, searchBand, topCandidates, selectedRect) => {
  const shapes = [
    {
      x: searchBand.x,
      y: searchBand.y,
      width: searchBand.width,
      height: searchBand.height,
      label: 'search-band',
      color: '#f59e0b'
    }
  ];

  topCandidates.forEach((candidate, index) => {
    const passed = candidate.gateEval.passed && candidate.score >= ALIGNMENT_CONFIG.stripGates.minAcceptedScore;
    const gateSuffix = [
      candidate.gateEval.checks.red ? 'r' : '!r',
      candidate.gateEval.checks.aspect ? 'a' : '!a',
      candidate.gateEval.checks.distance ? 'd' : '!d',
      candidate.gateEval.checks.periodicity ? 'p' : '!p'
    ].join('');

    shapes.push({
      x: candidate.rect.x,
      y: candidate.rect.y,
      width: candidate.rect.width,
      height: candidate.rect.height,
      label: `#${index + 1} ${candidate.score.toFixed(2)} ${gateSuffix}`,
      color: passed ? '#22c55e' : '#ef4444'
    });
  });

  if (selectedRect) {
    shapes.push({
      x: selectedRect.x,
      y: selectedRect.y,
      width: selectedRect.width,
      height: selectedRect.height,
      label: 'accepted-strip',
      color: '#16a34a'
    });
  }

  return drawOverlayCanvas(canvas, shapes);
};

export { detectDigitStripInBand, buildStripTopKOverlay };
