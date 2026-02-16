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
          const call = window.__jarvisRecognizeCalls++;
          const index = Math.floor(call / 2) % sequence.length;
          const digit = sequence[index] || '0';
          return {
            data: {
              text: digit,
              confidence: 96,
              symbols: [{ text: digit, confidence: 96 }]
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
