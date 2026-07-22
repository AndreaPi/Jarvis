const { test, expect } = require('@playwright/test');

test('canvas crops intersect requests with all four image edges', async ({ page }) => {
  await page.goto('/');

  const result = await page.evaluate(async () => {
    const { cropCanvas } = await import('/src/ocr/canvas-utils.js');
    const source = document.createElement('canvas');
    source.width = 10;
    source.height = 8;
    const cases = [
      { rect: { x: -3, y: 1, width: 5, height: 4 }, size: [2, 4] },
      { rect: { x: 8, y: 1, width: 5, height: 4 }, size: [2, 4] },
      { rect: { x: 2, y: -2, width: 4, height: 5 }, size: [4, 3] },
      { rect: { x: 2, y: 6, width: 4, height: 5 }, size: [4, 2] }
    ];
    return cases.map(({ rect, size }) => {
      const cropped = cropCanvas(source, rect);
      return {
        expected: size,
        cropped: [cropped.width, cropped.height]
      };
    });
  });

  for (const entry of result) {
    expect(entry.cropped).toEqual(entry.expected);
  }
});

test('fully external canvas crops remain a single boundary pixel', async ({ page }) => {
  await page.goto('/');

  const sizes = await page.evaluate(async () => {
    const { cropCanvas } = await import('/src/ocr/canvas-utils.js');
    const source = document.createElement('canvas');
    source.width = 10;
    source.height = 8;
    const left = cropCanvas(source, { x: -20, y: 0, width: 5, height: 4 });
    const right = cropCanvas(source, { x: 20, y: 0, width: 5, height: 4 });
    return [[left.width, left.height], [right.width, right.height]];
  });

  expect(sizes).toEqual([[1, 4], [1, 4]]);
});

test('classifier cell crops preserve trained boundary geometry', async ({ page }) => {
  await page.goto('/');

  const widths = await page.evaluate(async () => {
    const { splitIntoCells } = await import('/src/ocr/canvas-utils.js');
    const source = document.createElement('canvas');
    source.width = 100;
    source.height = 20;
    return splitIntoCells(source, 4, 0.1).map((cell) => cell.width);
  });

  expect(widths).toEqual([30, 30, 30, 28]);
});
