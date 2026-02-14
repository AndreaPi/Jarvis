import { clamp } from './canvas-utils.js';

const toFiniteNumber = (value) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const normalizeRect = (rect) => {
  if (!rect || typeof rect !== 'object') {
    return null;
  }
  const x = toFiniteNumber(rect.x);
  const y = toFiniteNumber(rect.y);
  const width = toFiniteNumber(rect.width);
  const height = toFiniteNumber(rect.height);
  if (x === null || y === null || width === null || height === null) {
    return null;
  }
  if (width <= 0 || height <= 0) {
    return null;
  }
  const normalizedX = clamp(x, 0, 0.999);
  const normalizedY = clamp(y, 0, 0.999);
  const maxWidth = Math.max(0.001, 1 - normalizedX);
  const maxHeight = Math.max(0.001, 1 - normalizedY);
  const normalizedWidth = clamp(width, 0.001, maxWidth);
  const normalizedHeight = clamp(height, 0.001, maxHeight);
  return {
    x: normalizedX,
    y: normalizedY,
    width: normalizedWidth,
    height: normalizedHeight
  };
};

const createProbeMiss = (reason, extra = {}) => ({
  ok: false,
  reason,
  ...extra
});

const evaluateRectSanity = (rect, sanityConfig) => {
  if (!sanityConfig || sanityConfig.enabled === false) {
    return { ok: true };
  }

  const centerX = rect.x + rect.width * 0.5;
  const centerY = rect.y + rect.height * 0.5;
  const area = rect.width * rect.height;
  const aspect = rect.width / Math.max(rect.height, 1e-6);

  const minCenterX = Number.isFinite(sanityConfig.minCenterX) ? sanityConfig.minCenterX : 0;
  const maxCenterX = Number.isFinite(sanityConfig.maxCenterX) ? sanityConfig.maxCenterX : 1;
  if (centerX < minCenterX || centerX > maxCenterX) {
    return createProbeMiss('invalid-geometry', {
      geometry: { metric: 'centerX', value: centerX, min: minCenterX, max: maxCenterX }
    });
  }

  const minCenterY = Number.isFinite(sanityConfig.minCenterY) ? sanityConfig.minCenterY : 0;
  const maxCenterY = Number.isFinite(sanityConfig.maxCenterY) ? sanityConfig.maxCenterY : 1;
  if (centerY < minCenterY || centerY > maxCenterY) {
    return createProbeMiss('invalid-geometry', {
      geometry: { metric: 'centerY', value: centerY, min: minCenterY, max: maxCenterY }
    });
  }

  const minArea = Number.isFinite(sanityConfig.minArea) ? sanityConfig.minArea : 0;
  const maxArea = Number.isFinite(sanityConfig.maxArea) ? sanityConfig.maxArea : 1;
  if (area < minArea || area > maxArea) {
    return createProbeMiss('invalid-geometry', {
      geometry: { metric: 'area', value: area, min: minArea, max: maxArea }
    });
  }

  const minAspect = Number.isFinite(sanityConfig.minAspect) ? sanityConfig.minAspect : 0;
  const maxAspect = Number.isFinite(sanityConfig.maxAspect) ? sanityConfig.maxAspect : Number.POSITIVE_INFINITY;
  if (aspect < minAspect || aspect > maxAspect) {
    return createProbeMiss('invalid-geometry', {
      geometry: { metric: 'aspect', value: aspect, min: minAspect, max: maxAspect }
    });
  }

  return {
    ok: true,
    geometry: { centerX, centerY, area, aspect }
  };
};

const detectNeuralRoi = async (file, neuralRoiConfig) => {
  if (!neuralRoiConfig || !neuralRoiConfig.enabled || !neuralRoiConfig.endpoint) {
    return createProbeMiss('disabled');
  }
  if (!file || typeof fetch !== 'function' || typeof FormData === 'undefined') {
    return createProbeMiss('unsupported-environment');
  }

  const formData = new FormData();
  formData.append('image', file, file.name || 'meter.jpg');

  const timeoutMs = Number.isFinite(neuralRoiConfig.timeoutMs) ? neuralRoiConfig.timeoutMs : 3500;
  const abortController = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const timeoutId = abortController ? setTimeout(() => abortController.abort(), timeoutMs) : null;
  try {
    const response = await fetch(neuralRoiConfig.endpoint, {
      method: 'POST',
      body: formData,
      signal: abortController ? abortController.signal : undefined
    });
    if (!response.ok) {
      return createProbeMiss('http-error', { status: response.status });
    }

    let payload = null;
    try {
      payload = await response.json();
    } catch (error) {
      return createProbeMiss('invalid-json');
    }
    if (!payload || !payload.ok || !payload.bbox_norm) {
      return createProbeMiss('no-detection', {
        confidence: toFiniteNumber(payload && payload.confidence),
        model: payload && payload.model ? payload.model : null
      });
    }

    const rect = normalizeRect(payload.bbox_norm);
    if (!rect) {
      return createProbeMiss('invalid-bbox');
    }

    const confidence = toFiniteNumber(payload.confidence);
    if (confidence === null) {
      return createProbeMiss('invalid-confidence');
    }
    const minConfidence = Number.isFinite(neuralRoiConfig.minConfidence) ? neuralRoiConfig.minConfidence : 0;
    if (confidence < minConfidence) {
      return createProbeMiss('low-confidence', {
        confidence,
        minConfidence
      });
    }

    const sanity = evaluateRectSanity(rect, neuralRoiConfig.sanity);
    if (!sanity.ok) {
      return createProbeMiss(sanity.reason, {
        confidence,
        geometry: sanity.geometry
      });
    }

    return {
      ok: true,
      rect,
      confidence,
      model: payload.model || 'unknown',
      geometry: sanity.geometry || null
    };
  } catch (error) {
    if (error && error.name === 'AbortError') {
      return createProbeMiss('timeout');
    }
    return createProbeMiss('network-error');
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  }
};

export { detectNeuralRoi };
