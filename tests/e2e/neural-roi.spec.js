const path = require('path');
const { test, expect } = require('@playwright/test');

const METER_IMAGE_PATH = path.resolve(__dirname, '..', '..', 'assets', 'meter_02142026.JPEG');

const installFailIfUsedTesseractStub = async (page) => {
  await page.evaluate(() => {
    window.__jarvisCreateWorkerCalls = 0;
    window.Tesseract = {
      PSM: {
        SINGLE_WORD: 8,
        SPARSE_TEXT: 11,
        SINGLE_CHAR: 10
      },
      createWorker: async () => {
        window.__jarvisCreateWorkerCalls += 1;
        throw new Error('Tesseract worker should not run after neural ROI failure.');
      }
    };
  });
};

const openAppAndUploadImage = async (page) => {
  await page.goto('/');
  await installFailIfUsedTesseractStub(page);
  await page.setInputFiles('#photo-input', METER_IMAGE_PATH);
};

const noDetectionPayload = JSON.stringify({
  ok: false,
  bbox_norm: null,
  confidence: 0,
  class_id: null,
  class_name: null,
  model: 'mock-roi'
});

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
  await page.route('**/roi/detect', async (route) => {
    await route.abort('failed');
  });
  await openAppAndUploadImage(page);

  await page.getByRole('button', { name: 'Read meter' }).click();

  await expect(page.locator('#ocr-status')).toContainText(
    'Neural ROI failed (network-error). Enter the measurement manually.'
  );
});
