const toFiniteNumber = (value) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const createProbeMiss = (reason, extra = {}) => ({
  ok: false,
  reason,
  ...extra
});

const runtimeState = {
  consecutiveFailures: 0,
  disabledUntilTs: 0
};

const stripRuntimeState = {
  consecutiveFailures: 0,
  disabledUntilTs: 0
};

const canvasToBlob = (canvas) => {
  return new Promise((resolve, reject) => {
    if (!canvas || typeof canvas.toBlob !== 'function') {
      reject(new Error('canvas-to-blob-unavailable'));
      return;
    }
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error('canvas-to-blob-failed'));
        return;
      }
      resolve(blob);
    }, 'image/png');
  });
};

const normalizeDigit = (value) => {
  const raw = value === undefined || value === null ? '' : String(value);
  return /^\d$/.test(raw) ? raw : '';
};

const normalizeDigitString = (value, length = 4) => {
  const raw = value === undefined || value === null ? '' : String(value);
  const pattern = new RegExp(`^\\d{${length}}$`);
  return pattern.test(raw) ? raw : '';
};

const setFailureCooldown = (state, config) => {
  const maxFailures = Number.isFinite(config.disableAfterFailures) ? config.disableAfterFailures : 2;
  const cooldownMs = Number.isFinite(config.cooldownMs) ? config.cooldownMs : 8000;
  state.consecutiveFailures += 1;
  if (state.consecutiveFailures >= maxFailures) {
    state.disabledUntilTs = Date.now() + cooldownMs;
  }
};

const clearFailureState = (state) => {
  state.consecutiveFailures = 0;
  state.disabledUntilTs = 0;
};

const predictDigitCells = async (cellCanvases, classifierConfig, requestOptions = {}) => {
  if (!classifierConfig || !classifierConfig.enabled || !classifierConfig.endpoint) {
    return createProbeMiss('disabled');
  }
  if (!Array.isArray(cellCanvases) || !cellCanvases.length) {
    return createProbeMiss('missing-cells');
  }
  const ignoreCooldown = requestOptions && requestOptions.ignoreCooldown === true;
  if (!ignoreCooldown && Date.now() < runtimeState.disabledUntilTs) {
    return createProbeMiss('cooldown');
  }
  if (typeof fetch !== 'function' || typeof FormData === 'undefined') {
    return createProbeMiss('unsupported-environment');
  }

  const timeoutMs = Number.isFinite(classifierConfig.timeoutMs) ? classifierConfig.timeoutMs : 1800;
  const minCellConfidence = Number.isFinite(classifierConfig.minCellConfidence)
    ? classifierConfig.minCellConfidence
    : 0;
  const abortController = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const timeoutId = abortController ? setTimeout(() => abortController.abort(), timeoutMs) : null;

  try {
    const formData = new FormData();
    for (let index = 0; index < cellCanvases.length; index += 1) {
      const blob = await canvasToBlob(cellCanvases[index]);
      formData.append('images', blob, `cell_${index}.png`);
    }

    const response = await fetch(classifierConfig.endpoint, {
      method: 'POST',
      body: formData,
      signal: abortController ? abortController.signal : undefined
    });
    if (!response.ok) {
      setFailureCooldown(runtimeState, classifierConfig);
      return createProbeMiss('http-error', { status: response.status });
    }

    let payload = null;
    try {
      payload = await response.json();
    } catch (error) {
      setFailureCooldown(runtimeState, classifierConfig);
      return createProbeMiss('invalid-json');
    }

    if (!payload || !payload.ok || !Array.isArray(payload.predictions)) {
      setFailureCooldown(runtimeState, classifierConfig);
      return createProbeMiss('invalid-payload');
    }

    const predictions = [];
    let acceptedCount = 0;
    for (let index = 0; index < cellCanvases.length; index += 1) {
      const item = payload.predictions[index] || {};
      const confidence = toFiniteNumber(item.confidence) ?? 0;
      const candidateDigit = normalizeDigit(item.digit || item.predicted_digit);
      const accepted = (
        item.accepted !== false
        && !!candidateDigit
        && confidence >= minCellConfidence
      );
      if (accepted) {
        acceptedCount += 1;
      }
      predictions.push({
        digit: accepted ? candidateDigit : '',
        confidence,
        accepted
      });
    }

    if (!predictions.length) {
      setFailureCooldown(runtimeState, classifierConfig);
      return createProbeMiss('empty-predictions');
    }

    clearFailureState(runtimeState);
    return {
      ok: true,
      model: payload.model || null,
      device: payload.device || null,
      predictions,
      acceptedCount,
      minCellConfidence
    };
  } catch (error) {
    if (error && error.name === 'AbortError') {
      setFailureCooldown(runtimeState, classifierConfig);
      return createProbeMiss('timeout');
    }
    setFailureCooldown(runtimeState, classifierConfig);
    return createProbeMiss('network-error');
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  }
};

const predictDigitStrip = async (stripCanvas, stripReaderConfig, requestOptions = {}) => {
  if (!stripReaderConfig || !stripReaderConfig.enabled || !stripReaderConfig.endpoint) {
    return createProbeMiss('disabled');
  }
  if (!stripCanvas) {
    return createProbeMiss('missing-strip');
  }
  const ignoreCooldown = requestOptions && requestOptions.ignoreCooldown === true;
  if (!ignoreCooldown && Date.now() < stripRuntimeState.disabledUntilTs) {
    return createProbeMiss('cooldown');
  }
  if (typeof fetch !== 'function' || typeof FormData === 'undefined') {
    return createProbeMiss('unsupported-environment');
  }

  const timeoutMs = Number.isFinite(stripReaderConfig.timeoutMs) ? stripReaderConfig.timeoutMs : 1800;
  const minConfidence = Number.isFinite(stripReaderConfig.minConfidence)
    ? stripReaderConfig.minConfidence
    : 0;
  const abortController = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const timeoutId = abortController ? setTimeout(() => abortController.abort(), timeoutMs) : null;

  try {
    const formData = new FormData();
    const blob = await canvasToBlob(stripCanvas);
    formData.append('image', blob, 'strip.png');

    const response = await fetch(stripReaderConfig.endpoint, {
      method: 'POST',
      body: formData,
      signal: abortController ? abortController.signal : undefined
    });
    if (!response.ok) {
      setFailureCooldown(stripRuntimeState, stripReaderConfig);
      return createProbeMiss('http-error', { status: response.status });
    }

    let payload = null;
    try {
      payload = await response.json();
    } catch (error) {
      setFailureCooldown(stripRuntimeState, stripReaderConfig);
      return createProbeMiss('invalid-json');
    }

    if (!payload) {
      setFailureCooldown(stripRuntimeState, stripReaderConfig);
      return createProbeMiss('invalid-payload');
    }

    const predictedValue = normalizeDigitString(payload.value || payload.predicted_value);
    const confidence = toFiniteNumber(payload.confidence) ?? 0;
    if (!predictedValue) {
      setFailureCooldown(stripRuntimeState, stripReaderConfig);
      return createProbeMiss('invalid-payload');
    }
    const accepted = (
      payload.ok !== false
      && payload.accepted !== false
      && confidence >= minConfidence
    );
    if (!accepted) {
      return createProbeMiss('low-confidence', {
        value: predictedValue,
        confidence,
        minConfidence
      });
    }

    clearFailureState(stripRuntimeState);
    return {
      ok: true,
      model: payload.model || null,
      device: payload.device || null,
      value: predictedValue,
      confidence,
      digits: Array.isArray(payload.digits)
        ? payload.digits.map((digit) => normalizeDigit(digit))
        : predictedValue.split(''),
      digitConfidences: Array.isArray(payload.digit_confidences)
        ? payload.digit_confidences.map((value) => toFiniteNumber(value) ?? 0)
        : [],
      topKByPosition: Array.isArray(payload.top_k_by_position) ? payload.top_k_by_position : [],
      minConfidence
    };
  } catch (error) {
    if (error && error.name === 'AbortError') {
      setFailureCooldown(stripRuntimeState, stripReaderConfig);
      return createProbeMiss('timeout');
    }
    setFailureCooldown(stripRuntimeState, stripReaderConfig);
    return createProbeMiss('network-error');
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  }
};

export { predictDigitCells, predictDigitStrip };
