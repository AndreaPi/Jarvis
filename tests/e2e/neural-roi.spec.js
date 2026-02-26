const path = require('path');
const { test, expect } = require('@playwright/test');

const METER_IMAGE_PATH = path.resolve(__dirname, '..', '..', 'assets', 'meter_02142026.JPEG');
const TESSERACT_CDN_URL = 'https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js';

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

const buildFailIfUsedTesseractStub = () => `
(() => {
  window.__jarvisCreateWorkerCalls = 0;
  window.__jarvisRecognizeCalls = 0;
  window.Tesseract = {
    PSM: { SINGLE_WORD: 8, SPARSE_TEXT: 11, SINGLE_CHAR: 10 },
    createWorker: async () => {
      window.__jarvisCreateWorkerCalls += 1;
      throw new Error('Tesseract worker should not run after neural ROI failure.');
    }
  };
})();
`;

const buildSuccessTesseractStub = (digits) => `
(() => {
  const sequence = ${JSON.stringify(digits)};
  const joined = sequence.join('');
  window.__jarvisCreateWorkerCalls = 0;
  window.__jarvisRecognizeCalls = 0;
  window.Tesseract = {
    PSM: { SINGLE_WORD: 8, SPARSE_TEXT: 11, SINGLE_CHAR: 10 },
    createWorker: async () => {
      window.__jarvisCreateWorkerCalls += 1;
      return {
        loadLanguage: async () => {},
        initialize: async () => {},
        setParameters: async () => {},
        recognize: async () => {
          window.__jarvisRecognizeCalls += 1;
          return {
            data: {
              text: joined || '0',
              confidence: 96,
              symbols: sequence.map((digit) => ({ text: digit, confidence: 96 }))
            }
          };
        }
      };
    }
  };
})();
`;

const buildSingleHitWordPassStub = (digits) => `
(() => {
  const sequence = ${JSON.stringify(digits)};
  const joined = sequence.join('');
  window.__jarvisCreateWorkerCalls = 0;
  window.__jarvisRecognizeCalls = 0;
  let currentPsm = 8;
  let servedWordPass = false;
  window.Tesseract = {
    PSM: { SINGLE_WORD: 8, SPARSE_TEXT: 11, SINGLE_CHAR: 10 },
    createWorker: async () => {
      window.__jarvisCreateWorkerCalls += 1;
      return {
        loadLanguage: async () => {},
        initialize: async () => {},
        setParameters: async (params = {}) => {
          if (Number.isFinite(params.tessedit_pageseg_mode)) {
            currentPsm = params.tessedit_pageseg_mode;
          }
        },
        recognize: async () => {
          window.__jarvisRecognizeCalls += 1;
          if (currentPsm === 8 && !servedWordPass) {
            servedWordPass = true;
            return {
              data: {
                text: joined || '0',
                confidence: 97,
                symbols: sequence.map((digit) => ({ text: digit, confidence: 97 }))
              }
            };
          }
          return {
            data: {
              text: '',
              confidence: 0,
              symbols: []
            }
          };
        }
      };
    }
  };
})();
`;

const installTesseractStub = async (page, scriptBody) => {
  await page.route(TESSERACT_CDN_URL, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/javascript',
      body: scriptBody,
      headers: {
        'access-control-allow-origin': '*'
      }
    });
  });
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
  await installTesseractStub(page, buildFailIfUsedTesseractStub());
  await page.route('**/roi/detect', fulfillNoDetection);
  await openAppAndUploadImage(page);

  await page.getByRole('button', { name: 'Read meter' }).click();

  const status = page.locator('#ocr-status');
  await expect(status).toContainText('Neural ROI failed (no-detection). Enter the measurement manually.');
  await expect(page.locator('#reading-input')).toHaveValue('');

  const workerCalls = await page.evaluate(() => window.__jarvisCreateWorkerCalls);
  expect(workerCalls).toBe(0);
});

test('does not timeout on a 4.5s neural ROI response', async ({ page }) => {
  await installTesseractStub(page, buildFailIfUsedTesseractStub());
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
});

test('asks for manual input when neural ROI endpoint is unreachable', async ({ page }) => {
  await installTesseractStub(page, buildFailIfUsedTesseractStub());
  await page.route('**/roi/detect', async (route) => {
    await route.abort('failed');
  });
  await openAppAndUploadImage(page);

  await page.getByRole('button', { name: 'Read meter' }).click();

  await expect(page.locator('#ocr-status')).toContainText(
    'Neural ROI failed (network-error). Enter the measurement manually.'
  );
});

test('completes with a detected reading when neural ROI succeeds', async ({ page }) => {
  await installTesseractStub(page, buildSuccessTesseractStub(['2', '3', '1', '1']));
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

  const workerCalls = await page.evaluate(() => window.__jarvisCreateWorkerCalls);
  expect(workerCalls).toBeGreaterThan(0);
});

test('accepts a single word-pass hit in strip-only mode', async ({ page }) => {
  await installTesseractStub(page, buildSingleHitWordPassStub(['8', '5', '8', '8']));
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

  await expect(page.locator('#ocr-status')).toContainText('Reading detected: 8588. Review if needed.');
  await expect(page.locator('#reading-input')).toHaveValue('8588');
});

test('keeps ROI debug crop geometry stable for narrow neural ROI boxes', async ({ page }) => {
  await installTesseractStub(page, buildSuccessTesseractStub(['2', '3', '1', '2']));
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
  await waitForDebugStages(page, ['0b. neural roi crop', '5. detected strip crop', '6. OCR input candidate']);

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
      ocr: findStage('6. OCR input candidate')
    };
  });

  expect(dimensions).not.toBeNull();
  expect(dimensions.roi).not.toBeNull();
  expect(dimensions.strip).not.toBeNull();
  expect(dimensions.ocr).not.toBeNull();

  const roiArea = dimensions.roi.width * dimensions.roi.height;
  const stripArea = dimensions.strip.width * dimensions.strip.height;
  const roiMaxDim = Math.max(dimensions.roi.width, dimensions.roi.height);
  const stripMaxDim = Math.max(dimensions.strip.width, dimensions.strip.height);

  expect(stripArea).toBeGreaterThanOrEqual(Math.floor(roiArea * 0.12));
  expect(stripArea).toBeLessThanOrEqual(Math.ceil(roiArea * 0.95));
  expect(stripMaxDim).toBeGreaterThanOrEqual(Math.floor(roiMaxDim * 0.55));
  expect(Math.min(dimensions.strip.width, dimensions.strip.height)).toBeGreaterThanOrEqual(24);
  expect(dimensions.ocr.width).toBeGreaterThanOrEqual(dimensions.strip.width);
});
