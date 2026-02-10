import { ALIGNMENT_CONFIG } from './config.js';
import { getLuminanceAt, analyzeRegion, getRectFromNormalized } from './canvas-utils.js';

const detectMeterFace = (meterCrop) => {
  const ctx = meterCrop.getContext('2d', { willReadFrequently: true });
  const { width, height } = meterCrop;
  const data = ctx.getImageData(0, 0, width, height).data;
  const minDim = Math.min(width, height);
  let best = null;

  for (
    let yOffset = -ALIGNMENT_CONFIG.centerOffsetLimit;
    yOffset <= ALIGNMENT_CONFIG.centerOffsetLimit;
    yOffset += ALIGNMENT_CONFIG.centerOffsetStep
  ) {
    for (
      let xOffset = -ALIGNMENT_CONFIG.centerOffsetLimit;
      xOffset <= ALIGNMENT_CONFIG.centerOffsetLimit;
      xOffset += ALIGNMENT_CONFIG.centerOffsetStep
    ) {
      const cx = width * (0.5 + xOffset);
      const cy = height * (0.5 + yOffset);
      for (
        let radiusRatio = ALIGNMENT_CONFIG.radiusMinRatio;
        radiusRatio <= ALIGNMENT_CONFIG.radiusMaxRatio;
        radiusRatio += ALIGNMENT_CONFIG.radiusStepRatio
      ) {
        const radius = minDim * radiusRatio;
        const padded = radius * ALIGNMENT_CONFIG.facePadding;
        if (cx - padded < 0 || cy - padded < 0 || cx + padded >= width || cy + padded >= height) {
          continue;
        }

        let insideLum = 0;
        let ringLum = 0;
        let outsideLum = 0;
        const steps = 24;
        for (let i = 0; i < steps; i += 1) {
          const angle = (i / steps) * Math.PI * 2;
          const cos = Math.cos(angle);
          const sin = Math.sin(angle);
          insideLum += getLuminanceAt(data, width, height, cx + cos * radius * 0.58, cy + sin * radius * 0.58);
          ringLum += getLuminanceAt(data, width, height, cx + cos * radius * 1.0, cy + sin * radius * 1.0);
          outsideLum += getLuminanceAt(data, width, height, cx + cos * radius * 1.26, cy + sin * radius * 1.26);
        }
        insideLum /= steps;
        ringLum /= steps;
        outsideLum /= steps;
        const centerPenalty = (Math.abs(xOffset) + Math.abs(yOffset)) * 20;
        const score = insideLum * 0.7 + (255 - ringLum) * 0.95 - outsideLum * 0.25 - centerPenalty;

        if (!best || score > best.score) {
          best = { cx, cy, radius, score };
        }
      }
    }
  }

  if (!best) {
    return {
      faceRect: { x: 0, y: 0, width, height },
      circle: { cx: width / 2, cy: height / 2, radius: minDim * 0.35 }
    };
  }

  const paddedRadius = best.radius * ALIGNMENT_CONFIG.facePadding;
  return {
    faceRect: {
      x: best.cx - paddedRadius,
      y: best.cy - paddedRadius,
      width: paddedRadius * 2,
      height: paddedRadius * 2
    },
    circle: {
      cx: best.cx,
      cy: best.cy,
      radius: best.radius
    }
  };
};

const scoreCanonicalRotation = (faceCanvas) => {
  const ctx = faceCanvas.getContext('2d', { willReadFrequently: true });
  const { width, height } = faceCanvas;
  const data = ctx.getImageData(0, 0, width, height).data;
  const blackWindow = analyzeRegion(
    data,
    width,
    height,
    getRectFromNormalized(faceCanvas, { x: 0.06, y: 0.11, width: 0.52, height: 0.34 })
  );
  const redDialArea = analyzeRegion(
    data,
    width,
    height,
    getRectFromNormalized(faceCanvas, { x: 0.45, y: 0.18, width: 0.5, height: 0.58 })
  );

  return (
    blackWindow.darkRatio * 1.7
    + blackWindow.edgeXRatio * 2.4
    + redDialArea.redRatio * 1.8
    - blackWindow.redRatio * 2.5
    - redDialArea.darkRatio * 0.4
  );
};

export { detectMeterFace, scoreCanonicalRotation };
