const { test, expect } = require('@playwright/test');

test('detects a new OCR selection log when the capped buffer stays full', async ({ page }) => {
  await page.goto('/');

  const result = await page.evaluate(async () => {
    const { latestNewSelectionLog } = await import('/src/testset/run-test-set.js');
    const logs = Array.from({ length: 300 }, (_, id) => ({ id }));
    const previousLastLog = logs[logs.length - 1];
    const nextLog = { id: 300 };
    logs.push(nextLog);
    logs.shift();
    return {
      length: logs.length,
      detectedId: latestNewSelectionLog(logs, previousLastLog)?.id ?? null
    };
  });

  expect(result).toEqual({ length: 300, detectedId: 300 });
});

test('does not reuse the previous OCR selection log', async ({ page }) => {
  await page.goto('/');

  const detected = await page.evaluate(async () => {
    const { latestNewSelectionLog } = await import('/src/testset/run-test-set.js');
    const logs = [{ id: 1 }];
    return latestNewSelectionLog(logs, logs[0]);
  });

  expect(detected).toBeNull();
});
