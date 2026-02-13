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

const detectNeuralRoi = async (file, neuralRoiConfig) => {
  if (!neuralRoiConfig || !neuralRoiConfig.enabled || !neuralRoiConfig.endpoint) {
    return null;
  }
  if (!file || typeof fetch !== 'function' || typeof FormData === 'undefined') {
    return null;
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
      return null;
    }

    const payload = await response.json();
    if (!payload || !payload.ok || !payload.bbox_norm) {
      return null;
    }

    const rect = normalizeRect(payload.bbox_norm);
    if (!rect) {
      return null;
    }

    const confidence = toFiniteNumber(payload.confidence);
    if (confidence === null) {
      return null;
    }
    const minConfidence = Number.isFinite(neuralRoiConfig.minConfidence) ? neuralRoiConfig.minConfidence : 0;
    if (confidence < minConfidence) {
      return null;
    }

    return {
      rect,
      confidence,
      model: payload.model || 'unknown'
    };
  } catch (error) {
    return null;
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  }
};

export { detectNeuralRoi };
