const path = require('path');
const { test, expect } = require('@playwright/test');

const METER_IMAGE_PATH = path.resolve(__dirname, '..', '..', 'assets', 'meter_20260214.JPEG');

const noDetectionPayload = JSON.stringify({
  ok: false,
  bbox_norm: null,
  confidence: 0,
  class_id: null,
  class_name: null,
  model: 'mock-roi'
});

const successPayload = JSON.stringify({
  ok: true,
  bbox_norm: {
    x: 0.3,
    y: 0.36,
    width: 0.2,
    height: 0.12
  },
  confidence: 0.9,
  class_id: 0,
  class_name: 'digit_window',
  model: 'mock-roi'
});

const tallRoiPayload = JSON.stringify({
  ok: true,
  bbox_norm: {
    x: 0.42,
    y: 0.43,
    width: 0.08,
    height: 0.14
  },
  confidence: 0.9,
  class_id: 0,
  class_name: 'digit_window',
  model: 'mock-roi'
});

const buildDigitClassifierPayload = (digits, confidence = 0.96) => {
  const normalized = Array.from({ length: 4 }, (_, index) => {
    const value = digits[index] || '0';
    return {
      digit: String(value),
      confidence,
      accepted: true
    };
  });

  return JSON.stringify({
    ok: true,
    model: 'mock-digit',
    device: 'cpu',
    predictions: normalized
  });
};

const buildDigitStripReaderPayload = (value = '2311', confidence = 0.97) => {
  const digits = String(value).split('');
  return JSON.stringify({
    ok: true,
    accepted: true,
    model: 'mock-strip-digit',
    device: 'cpu',
    value,
    predicted_value: value,
    confidence,
    min_confidence: 0,
    digits,
    predicted_digits: digits,
    digit_confidences: digits.map(() => confidence),
    top_k_by_position: digits.map((digit) => ([
      { digit, confidence },
      { digit: digit === '9' ? '8' : '9', confidence: 1 - confidence }
    ]))
  });
};

const buildDigitStripReader23xxPayload = (value = '2311', options = {}) => {
  const {
    accepted = true,
    confidence = 0.98,
    guardConfidence = 0.99,
    guardThreshold = 0.98
  } = options;
  const normalizedValue = String(value);
  const suffixDigits = normalizedValue.slice(2).split('');
  return JSON.stringify({
    ok: true,
    accepted,
    model: 'mock-strip-digit-23xx',
    device: 'cpu',
    value: accepted ? normalizedValue : null,
    predicted_value: normalizedValue,
    fixed_prefix: '23',
    prefix_guard: {
      label: 'second_digit_is_3',
      is_3: guardConfidence >= 0.5,
      confidence: guardConfidence,
      threshold: guardThreshold
    },
    suffix_digits: suffixDigits,
    confidence,
    guard_confidence: guardConfidence,
    guard_threshold: guardThreshold,
    suffix_confidences: suffixDigits.map(() => confidence),
    top_k_by_position: suffixDigits.map((digit) => ([
      { digit, confidence },
      { digit: digit === '9' ? '8' : '9', confidence: 1 - confidence }
    ]))
  });
};

const installDigitClassifierMock = async (page, options = {}) => {
  const {
    digits = ['0', '0', '0', '0'],
    confidence = 0.96,
    mode = 'success'
  } = options;

  let calls = 0;
  await page.route('**/digit/predict-cells', async (route) => {
    calls += 1;
    if (mode === 'network-error') {
      await route.abort('failed');
      return;
    }
    if (mode === 'http-error') {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        headers: {
          'access-control-allow-origin': '*'
        },
        body: JSON.stringify({ ok: false, error: 'mock failure' })
      });
      return;
    }

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      headers: {
        'access-control-allow-origin': '*'
      },
      body: buildDigitClassifierPayload(digits, confidence)
    });
  });

  return {
    getCalls: () => calls
  };
};

const installDigitStripReaderMock = async (page, options = {}) => {
  const {
    value = '2311',
    confidence = 0.97,
    mode = 'success',
    responses = null
  } = options;

  let calls = 0;
  await page.route('**/digit/predict-strip', async (route) => {
    calls += 1;
    if (mode === 'network-error') {
      await route.abort('failed');
      return;
    }
    if (mode === 'http-error') {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        headers: {
          'access-control-allow-origin': '*'
        },
        body: JSON.stringify({ ok: false, error: 'mock strip failure' })
      });
      return;
    }

    const response = Array.isArray(responses) && responses.length
      ? responses[Math.min(calls - 1, responses.length - 1)]
      : {};
    const responseValue = response.value || value;
    const responseConfidence = Number.isFinite(response.confidence) ? response.confidence : confidence;
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      headers: {
        'access-control-allow-origin': '*'
      },
      body: buildDigitStripReaderPayload(responseValue, responseConfidence)
    });
  });

  return {
    getCalls: () => calls
  };
};

const installDigitStripReader23xxMock = async (page, options = {}) => {
  const {
    value = '2311',
    accepted = true,
    confidence = 0.98,
    guardConfidence = 0.99,
    mode = 'success'
  } = options;

  let calls = 0;
  await page.route('**/digit/predict-strip-23xx', async (route) => {
    calls += 1;
    if (mode === 'network-error') {
      await route.abort('failed');
      return;
    }
    if (mode === 'http-error') {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        headers: {
          'access-control-allow-origin': '*'
        },
        body: JSON.stringify({ ok: false, error: 'mock strip 23xx failure' })
      });
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      headers: {
        'access-control-allow-origin': '*'
      },
      body: buildDigitStripReader23xxPayload(value, {
        accepted,
        confidence,
        guardConfidence
      })
    });
  });

  return {
    getCalls: () => calls
  };
};

const openAppAndUploadImage = async (page) => {
  await page.goto('/');
  await page.setInputFiles('#photo-input', METER_IMAGE_PATH);
};

const waitForDebugStages = async (page, stageNames) => {
  await page.waitForFunction((names) => {
    const session = document.querySelector('.debug-session');
    if (!session) {
      return false;
    }
    const cards = [...session.querySelectorAll('.debug-stage')];
    return names.every((name) => {
      const card = cards.find((item) => {
        const caption = item.querySelector('.debug-stage-name');
        return (caption && caption.textContent && caption.textContent.trim()) === name;
      });
      if (!card) {
        return false;
      }
      const image = card.querySelector('img');
      return !!(image && image.naturalWidth > 0 && image.naturalHeight > 0);
    });
  }, stageNames);
};

const fulfillNoDetection = async (route) => {
  await route.fulfill({
    status: 200,
    contentType: 'application/json',
    headers: {
      'access-control-allow-origin': '*'
    },
    body: noDetectionPayload
  });
};

test('asks for manual input when neural ROI returns no detection', async ({ page }) => {
  const digitMock = await installDigitClassifierMock(page, { digits: ['9', '9', '9', '9'] });
  const stripMock = await installDigitStripReaderMock(page, { value: '9999' });
  const strip23xxMock = await installDigitStripReader23xxMock(page, { value: '2399' });
  await page.route('**/roi/detect', fulfillNoDetection);
  await openAppAndUploadImage(page);

  await page.getByRole('button', { name: 'Read meter' }).click();

  const status = page.locator('#ocr-status');
  await expect(status).toContainText('Neural ROI failed (no-detection). Enter the measurement manually.');
  await expect(page.locator('#reading-input')).toHaveValue('');
  expect(digitMock.getCalls()).toBe(0);
  expect(stripMock.getCalls()).toBe(0);
  expect(strip23xxMock.getCalls()).toBe(0);
});

test('does not timeout on a 4.5s neural ROI response', async ({ page }) => {
  const digitMock = await installDigitClassifierMock(page, { digits: ['9', '9', '9', '9'] });
  const stripMock = await installDigitStripReaderMock(page, { value: '9999' });
  const strip23xxMock = await installDigitStripReader23xxMock(page, { value: '2399' });
  await page.route('**/roi/detect', async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 4500));
    await fulfillNoDetection(route);
  });
  await openAppAndUploadImage(page);

  await page.getByRole('button', { name: 'Read meter' }).click();

  const status = page.locator('#ocr-status');
  await expect(status).toContainText('Neural ROI failed (no-detection). Enter the measurement manually.', {
    timeout: 20_000
  });
  await expect(status).not.toContainText('timeout');
  expect(digitMock.getCalls()).toBe(0);
  expect(stripMock.getCalls()).toBe(0);
  expect(strip23xxMock.getCalls()).toBe(0);
});

test('asks for manual input when neural ROI endpoint is unreachable', async ({ page }) => {
  const digitMock = await installDigitClassifierMock(page, { digits: ['9', '9', '9', '9'] });
  const stripMock = await installDigitStripReaderMock(page, { value: '9999' });
  const strip23xxMock = await installDigitStripReader23xxMock(page, { value: '2399' });
  await page.route('**/roi/detect', async (route) => {
    await route.abort('failed');
  });
  await openAppAndUploadImage(page);

  await page.getByRole('button', { name: 'Read meter' }).click();

  await expect(page.locator('#ocr-status')).toContainText(
    'Neural ROI failed (network-error). Enter the measurement manually.'
  );
  expect(digitMock.getCalls()).toBe(0);
  expect(stripMock.getCalls()).toBe(0);
  expect(strip23xxMock.getCalls()).toBe(0);
});

test('completes with a detected reading when neural ROI and classifier succeed', async ({ page }) => {
  const digitMock = await installDigitClassifierMock(page, { digits: ['2', '3', '1', '1'], confidence: 0.98 });
  const stripMock = await installDigitStripReaderMock(page, { value: '9999', confidence: 0.99 });
  const strip23xxMock = await installDigitStripReader23xxMock(page, {
    value: '2311',
    accepted: true,
    confidence: 0.98,
    guardConfidence: 0.99
  });
  await page.route('**/roi/detect', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      headers: {
        'access-control-allow-origin': '*'
      },
      body: successPayload
    });
  });
  await openAppAndUploadImage(page);

  await page.getByRole('button', { name: 'Read meter' }).click();

  await expect(page.locator('#ocr-status')).toContainText('Reading detected: 2311. Review if needed.');
  await expect(page.locator('#reading-input')).toHaveValue('2311');
  expect(digitMock.getCalls()).toBe(4);
  expect(stripMock.getCalls()).toBeGreaterThan(0);
  expect(strip23xxMock.getCalls()).toBeGreaterThan(0);

  const selectionLog = await page.evaluate(() => {
    const logs = window.__jarvisOcrSelectionLogs || [];
    return logs[logs.length - 1] || null;
  });
  expect(selectionLog).not.toBeNull();
  expect(selectionLog.selected.value).toBe('2311');
  expect(selectionLog.stripReader.value).toBe('9999');
  expect(selectionLog.stripReader.headlineReason).toBe('selected-source');
  expect(selectionLog.stripReader.selectionRule).toBe('selected-source-first-then-confidence');
  expect(selectionLog.stripReader.selectedSourceCandidate.sourceLabel).toBe(selectionLog.selected.sourceLabel);
  expect(selectionLog.stripReader.confidenceBest.value).toBe('9999');
  expect(selectionLog.stripReader.candidates).toHaveLength(stripMock.getCalls());
  expect(selectionLog.stripReader23xx.value).toBe('2311');
  expect(selectionLog.stripReader23xx.accepted).toBe(true);
  expect(selectionLog.stripReader23xx.candidates).toHaveLength(strip23xxMock.getCalls());
  expect(selectionLog.stripReader.bySource.some((entry) => (
    entry.sourceLabel === selectionLog.selected.sourceLabel
    && entry.matchesSelectedSource === true
    && entry.okCount >= 1
  ))).toBe(true);
  expect(selectionLog.candidateTrace.some((entry) => (
    entry.stripReader && entry.stripReader.value === '9999'
  ))).toBe(true);
  expect(selectionLog.candidateTrace.some((entry) => (
    entry.stripReader23xx && entry.stripReader23xx.predictedValue === '2311'
  ))).toBe(true);
});

test('asks for manual input when classifier endpoint fails after ROI success', async ({ page }) => {
  await installDigitClassifierMock(page, { mode: 'network-error' });
  const stripMock = await installDigitStripReaderMock(page, { value: '2311', confidence: 0.99 });
  const strip23xxMock = await installDigitStripReader23xxMock(page, {
    value: '2311',
    accepted: false,
    confidence: 0.82,
    guardConfidence: 0.82
  });
  await page.route('**/roi/detect', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      headers: {
        'access-control-allow-origin': '*'
      },
      body: successPayload
    });
  });
  await openAppAndUploadImage(page);

  await page.getByRole('button', { name: 'Read meter' }).click();

  await expect(page.locator('#ocr-status')).toContainText('No clear reading detected. Enter it manually.');
  await expect(page.locator('#reading-input')).toHaveValue('');
  expect(stripMock.getCalls()).toBeGreaterThan(0);
  expect(strip23xxMock.getCalls()).toBeGreaterThan(0);

  const selectionLog = await page.evaluate(() => {
    const logs = window.__jarvisOcrSelectionLogs || [];
    return logs[logs.length - 1] || null;
  });
  expect(selectionLog).not.toBeNull();
  expect(selectionLog.selected).toBeNull();
  expect(selectionLog.stripReader.value).toBe('2311');
  expect(selectionLog.stripReader.headlineReason).toBe('highest-confidence');
  expect(selectionLog.stripReader.candidates).toHaveLength(stripMock.getCalls());
  expect(selectionLog.stripReader.bySource.length).toBeGreaterThan(0);
  expect(selectionLog.stripReader23xx.accepted).toBe(false);
  expect(selectionLog.stripReader23xx.value).toBeNull();
  expect(selectionLog.stripReader23xx.predictedValue).toBe('2311');
});

test('recovers from classifier cooldown once the backend responds again', async ({ page }) => {
  await installDigitStripReaderMock(page, { value: '2311', confidence: 0.99 });
  await installDigitStripReader23xxMock(page, { value: '2311', accepted: true });
  let classifierMode = 'network-error';
  let classifierCalls = 0;
  await page.route('**/digit/predict-cells', async (route) => {
    classifierCalls += 1;
    if (classifierMode === 'network-error') {
      await route.abort('failed');
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      headers: {
        'access-control-allow-origin': '*'
      },
      body: buildDigitClassifierPayload(['2', '3', '1', '1'], 0.98)
    });
  });
  await page.route('**/roi/detect', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      headers: {
        'access-control-allow-origin': '*'
      },
      body: successPayload
    });
  });
  await openAppAndUploadImage(page);

  await page.getByRole('button', { name: 'Read meter' }).click();
  await expect(page.locator('#ocr-status')).toContainText('No clear reading detected. Enter it manually.');

  classifierMode = 'success';
  await page.getByRole('button', { name: 'Read meter' }).click();
  await expect(page.locator('#ocr-status')).toContainText('Reading detected: 2311. Review if needed.');
  await expect(page.locator('#reading-input')).toHaveValue('2311');
  expect(classifierCalls).toBeGreaterThanOrEqual(2);
});

test('keeps ROI debug crop geometry stable for narrow neural ROI boxes', async ({ page }) => {
  await installDigitClassifierMock(page, { digits: ['2', '3', '1', '2'], confidence: 0.97 });
  await installDigitStripReaderMock(page, { value: '2312', confidence: 0.99 });
  await installDigitStripReader23xxMock(page, { value: '2312', accepted: true });
  await page.route('**/roi/detect', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      headers: {
        'access-control-allow-origin': '*'
      },
      body: tallRoiPayload
    });
  });
  await openAppAndUploadImage(page);

  await page.getByRole('button', { name: 'Read meter' }).click();

  await expect(page.locator('#ocr-status')).toContainText(
    /Reading detected|No clear reading detected|Enter it manually|Enter the measurement manually/,
    { timeout: 20_000 }
  );
  await waitForDebugStages(page, [
    '0b. neural roi crop',
    '5. detected strip crop',
    '6. OCR input candidate',
    '7. classifier cell crops',
    '8. strip reader input'
  ]);

  const dimensions = await page.evaluate(() => {
    const session = document.querySelector('.debug-session');
    if (!session) {
      return null;
    }
    const cards = [...session.querySelectorAll('.debug-stage')];
    const findStage = (name) => {
      const card = cards.find((item) => {
        const caption = item.querySelector('.debug-stage-name');
        return (caption && caption.textContent && caption.textContent.trim()) === name;
      });
      if (!card) {
        return null;
      }
      const image = card.querySelector('img');
      if (!image) {
        return null;
      }
      return {
        width: image.naturalWidth,
        height: image.naturalHeight
      };
    };
    return {
      roi: findStage('0b. neural roi crop'),
      strip: findStage('5. detected strip crop'),
      ocr: findStage('6. OCR input candidate'),
      cells: findStage('7. classifier cell crops'),
      stripReader: findStage('8. strip reader input')
    };
  });

  expect(dimensions).not.toBeNull();
  expect(dimensions.roi).not.toBeNull();
  expect(dimensions.strip).not.toBeNull();
  expect(dimensions.ocr).not.toBeNull();
  expect(dimensions.cells).not.toBeNull();
  expect(dimensions.stripReader).not.toBeNull();

  const roiArea = dimensions.roi.width * dimensions.roi.height;
  const stripArea = dimensions.strip.width * dimensions.strip.height;
  const roiMaxDim = Math.max(dimensions.roi.width, dimensions.roi.height);
  const stripMaxDim = Math.max(dimensions.strip.width, dimensions.strip.height);

  expect(stripArea).toBeGreaterThanOrEqual(Math.floor(roiArea * 0.12));
  expect(stripArea).toBeLessThanOrEqual(Math.ceil(roiArea * 0.95));
  expect(stripMaxDim).toBeGreaterThanOrEqual(Math.floor(roiMaxDim * 0.55));
  expect(Math.min(dimensions.strip.width, dimensions.strip.height)).toBeGreaterThanOrEqual(24);
  expect(dimensions.ocr.width).toBeGreaterThanOrEqual(dimensions.strip.width);
  expect(dimensions.cells.width).toBeGreaterThan(0);
  expect(dimensions.cells.height).toBeGreaterThan(0);
  expect(dimensions.stripReader.width).toBeGreaterThan(0);
  expect(dimensions.stripReader.height).toBeGreaterThan(0);
});
