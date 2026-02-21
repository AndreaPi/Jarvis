import { DEBUG_CONFIG } from './config.js';

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const normalizeAngle = (angle) => {
  const value = Number(angle);
  if (!Number.isFinite(value)) {
    return Number.NaN;
  }
  return ((value % 360) + 360) % 360;
};

const cloneCanvas = (source) => {
  const canvas = document.createElement('canvas');
  canvas.width = source.width;
  canvas.height = source.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(source, 0, 0);
  return canvas;
};

const getRectFromNormalized = (canvas, rect) => ({
  x: canvas.width * rect.x,
  y: canvas.height * rect.y,
  width: canvas.width * rect.width,
  height: canvas.height * rect.height
});

const drawOverlayCanvas = (source, shapes = []) => {
  const canvas = cloneCanvas(source);
  const ctx = canvas.getContext('2d');
  const strokeWidth = Math.max(2, Math.round(canvas.width / 180));
  const fontSize = Math.max(10, Math.round(canvas.width / 28));
  ctx.lineWidth = strokeWidth;
  ctx.font = `600 ${fontSize}px Manrope, sans-serif`;
  ctx.textBaseline = 'top';

  shapes.forEach((shape, index) => {
    const color = shape.color || DEBUG_CONFIG.colors[index % DEBUG_CONFIG.colors.length];
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    if (shape.type === 'circle') {
      ctx.beginPath();
      ctx.arc(shape.cx, shape.cy, shape.radius, 0, Math.PI * 2);
      ctx.stroke();
    } else {
      ctx.strokeRect(shape.x, shape.y, shape.width, shape.height);
    }
    if (shape.label) {
      const paddingX = 6;
      const paddingY = 4;
      const labelWidth = Math.ceil(ctx.measureText(shape.label).width) + paddingX * 2;
      const labelHeight = fontSize + paddingY * 2;
      const baseX = shape.type === 'circle' ? shape.cx - shape.radius : shape.x;
      const baseY = shape.type === 'circle' ? shape.cy - shape.radius : shape.y;
      const labelX = clamp(baseX, 0, Math.max(0, canvas.width - labelWidth));
      const labelY = clamp(baseY - labelHeight - 2, 0, Math.max(0, canvas.height - labelHeight));
      ctx.fillStyle = 'rgba(15, 23, 42, 0.75)';
      ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
      ctx.fillStyle = '#f8fafc';
      ctx.fillText(shape.label, labelX + paddingX, labelY + paddingY);
      ctx.fillStyle = color;
    }
  });

  return canvas;
};
const loadImageBitmap = async (file) => {
  if ('createImageBitmap' in window) {
    try {
      return await createImageBitmap(file, { imageOrientation: 'from-image' });
    } catch (error) {
      console.warn('createImageBitmap failed, falling back to Image.', error);
    }
  }

  return new Promise((resolve, reject) => {
    const img = new Image();
    const objectUrl = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(objectUrl);
      resolve(img);
    };
    img.onerror = (error) => {
      URL.revokeObjectURL(objectUrl);
      reject(error);
    };
    img.src = objectUrl;
  });
};

const drawImageToCanvas = (image, maxDimension) => {
  const canvas = document.createElement('canvas');
  const maxSide = Math.max(image.width, image.height);
  const scale = maxSide > maxDimension ? maxDimension / maxSide : 1;
  canvas.width = Math.round(image.width * scale);
  canvas.height = Math.round(image.height * scale);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
  return canvas;
};

const cropCanvas = (source, rect) => {
  const safeRect = {
    x: clamp(rect.x, 0, source.width - 1),
    y: clamp(rect.y, 0, source.height - 1),
    width: clamp(rect.width, 1, source.width),
    height: clamp(rect.height, 1, source.height)
  };
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(safeRect.width);
  canvas.height = Math.round(safeRect.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(
    source,
    safeRect.x,
    safeRect.y,
    safeRect.width,
    safeRect.height,
    0,
    0,
    canvas.width,
    canvas.height
  );
  return canvas;
};

const cropCenterSquare = (source, scale) => {
  const size = Math.round(Math.min(source.width, source.height) * scale);
  const x = Math.round((source.width - size) / 2);
  const y = Math.round((source.height - size) / 2);
  return cropCanvas(source, { x, y, width: size, height: size });
};

const rotateCanvas = (source, angle) => {
  const normalized = ((angle % 360) + 360) % 360;
  if (normalized === 0) {
    return source;
  }
  const canvas = document.createElement('canvas');
  const swap = normalized === 90 || normalized === 270;
  canvas.width = swap ? source.height : source.width;
  canvas.height = swap ? source.width : source.height;
  const ctx = canvas.getContext('2d');
  ctx.translate(canvas.width / 2, canvas.height / 2);
  ctx.rotate((normalized * Math.PI) / 180);
  ctx.drawImage(source, -source.width / 2, -source.height / 2);
  return canvas;
};

const scaleCanvas = (source, targetWidth) => {
  if (source.width >= targetWidth) {
    return source;
  }
  const scale = targetWidth / source.width;
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(source.width * scale);
  canvas.height = Math.round(source.height * scale);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas;
};

const splitIntoCells = (source, count, overlapRatio) => {
  const cells = [];
  const cellWidth = source.width / count;
  const overlap = cellWidth * overlapRatio;
  for (let i = 0; i < count; i += 1) {
    const x = cellWidth * i - overlap;
    const width = cellWidth + overlap * 2;
    cells.push(cropCanvas(source, { x, y: 0, width, height: source.height }));
  }
  return cells;
};

const computeOtsuThreshold = (data, contrast, brightness) => {
  const histogram = new Array(256).fill(0);
  for (let i = 0; i < data.length; i += 4) {
    const lum = data[i] * 0.2126 + data[i + 1] * 0.7152 + data[i + 2] * 0.0722;
    const adjusted = clamp((lum - 128) * contrast + 128 + brightness, 0, 255);
    histogram[adjusted | 0] += 1;
  }

  const total = data.length / 4;
  let sum = 0;
  for (let i = 0; i < 256; i += 1) {
    sum += i * histogram[i];
  }

  let sumB = 0;
  let weightB = 0;
  let weightF = 0;
  let maxVariance = 0;
  let threshold = 128;

  for (let i = 0; i < 256; i += 1) {
    weightB += histogram[i];
    if (!weightB) {
      continue;
    }
    weightF = total - weightB;
    if (!weightF) {
      break;
    }
    sumB += i * histogram[i];
    const meanB = sumB / weightB;
    const meanF = (sum - sumB) / weightF;
    const variance = weightB * weightF * (meanB - meanF) ** 2;
    if (variance > maxVariance) {
      maxVariance = variance;
      threshold = i;
    }
  }

  return threshold;
};

const preprocessCanvas = (source, mode) => {
  const contrast = mode === 'binary' ? 1.6 : 1.25;
  const brightness = mode === 'binary' ? 6 : 0;
  const canvas = document.createElement('canvas');
  canvas.width = source.width;
  canvas.height = source.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(source, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  const threshold = mode === 'binary' ? computeOtsuThreshold(data, contrast, brightness) : null;

  for (let i = 0; i < data.length; i += 4) {
    const lum = data[i] * 0.2126 + data[i + 1] * 0.7152 + data[i + 2] * 0.0722;
    let adjusted = clamp((lum - 128) * contrast + 128 + brightness, 0, 255);
    if (mode === 'binary') {
      adjusted = adjusted > threshold ? 255 : 0;
    }
    data[i] = adjusted;
    data[i + 1] = adjusted;
    data[i + 2] = adjusted;
    data[i + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
  return canvas;
};

const tightenCropByInk = (source, minAreaRatio = 0.15) => {
  const ctx = source.getContext('2d', { willReadFrequently: true });
  const { width, height } = source;
  const data = ctx.getImageData(0, 0, width, height).data;
  const cols = new Array(width).fill(0);
  const rows = new Array(height).fill(0);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = (y * width + x) * 4;
      const lum = data[idx] * 0.2126 + data[idx + 1] * 0.7152 + data[idx + 2] * 0.0722;
      const dark = 255 - lum;
      cols[x] += dark;
      rows[y] += dark;
    }
  }

  const sumCols = cols.reduce((acc, value) => acc + value, 0);
  const sumRows = rows.reduce((acc, value) => acc + value, 0);
  const meanCols = sumCols / width;
  const meanRows = sumRows / height;
  const maxCols = Math.max(...cols);
  const maxRows = Math.max(...rows);
  const colThreshold = meanCols + (maxCols - meanCols) * 0.25;
  const rowThreshold = meanRows + (maxRows - meanRows) * 0.25;

  let left = cols.findIndex((value) => value > colThreshold);
  let right = cols.length - 1 - [...cols].reverse().findIndex((value) => value > colThreshold);
  let top = rows.findIndex((value) => value > rowThreshold);
  let bottom = rows.length - 1 - [...rows].reverse().findIndex((value) => value > rowThreshold);

  if (left < 0 || right <= left || top < 0 || bottom <= top) {
    return source;
  }

  const paddingX = Math.round((right - left) * 0.08);
  const paddingY = Math.round((bottom - top) * 0.15);
  left = clamp(left - paddingX, 0, width - 1);
  right = clamp(right + paddingX, 1, width);
  top = clamp(top - paddingY, 0, height - 1);
  bottom = clamp(bottom + paddingY, 1, height);

  const cropWidth = right - left;
  const cropHeight = bottom - top;
  const areaRatio = (cropWidth * cropHeight) / (width * height);
  if (areaRatio < minAreaRatio || areaRatio > 0.95) {
    return source;
  }

  return cropCanvas(source, { x: left, y: top, width: cropWidth, height: cropHeight });
};

const findDigitWindowByEdges = (source) => {
  const ctx = source.getContext('2d', { willReadFrequently: true });
  const { width, height } = source;
  const data = ctx.getImageData(0, 0, width, height).data;
  const cols = new Array(width).fill(0);
  const rows = new Array(height).fill(0);
  const maxRow = Math.floor(height * 0.65);

  const lumAt = (x, y) => {
    const idx = (y * width + x) * 4;
    return data[idx] * 0.2126 + data[idx + 1] * 0.7152 + data[idx + 2] * 0.0722;
  };

  for (let y = 1; y < maxRow - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const edge = Math.abs(lumAt(x + 1, y) - lumAt(x - 1, y));
      cols[x] += edge;
      rows[y] += edge;
    }
  }

  const meanCols = cols.reduce((acc, value) => acc + value, 0) / width;
  const meanRows = rows.reduce((acc, value) => acc + value, 0) / height;
  const maxCols = Math.max(...cols);
  const maxRows = Math.max(...rows);
  const colThreshold = meanCols + (maxCols - meanCols) * 0.35;
  const rowThreshold = meanRows + (maxRows - meanRows) * 0.35;

  let left = cols.findIndex((value) => value > colThreshold);
  let right = cols.length - 1 - [...cols].reverse().findIndex((value) => value > colThreshold);
  let top = rows.findIndex((value) => value > rowThreshold);
  let bottom = rows.length - 1 - [...rows].reverse().findIndex((value) => value > rowThreshold);

  if (left < 0 || right <= left || top < 0 || bottom <= top) {
    return null;
  }

  const paddingX = Math.round((right - left) * 0.08);
  const paddingY = Math.round((bottom - top) * 0.2);
  left = clamp(left - paddingX, 0, width - 1);
  right = clamp(right + paddingX, 1, width);
  top = clamp(top - paddingY, 0, height - 1);
  bottom = clamp(bottom + paddingY, 1, height);

  const cropWidth = right - left;
  const cropHeight = bottom - top;
  const areaRatio = (cropWidth * cropHeight) / (width * height);
  if (areaRatio < 0.08 || areaRatio > 0.7) {
    return null;
  }

  return { x: left, y: top, width: cropWidth, height: cropHeight };
};

const hasInk = (canvas) => {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const { width, height } = canvas;
  const data = ctx.getImageData(0, 0, width, height).data;
  let dark = 0;
  const total = width * height;
  for (let i = 0; i < data.length; i += 4) {
    const lum = data[i] * 0.2126 + data[i + 1] * 0.7152 + data[i + 2] * 0.0722;
    if (lum < 140) {
      dark += 1;
    }
  }
  const ratio = dark / total;
  return ratio > 0.01 && ratio < 0.4;
};

const getLuminanceAt = (data, width, height, x, y) => {
  const clampedX = clamp(Math.round(x), 0, width - 1);
  const clampedY = clamp(Math.round(y), 0, height - 1);
  const idx = (clampedY * width + clampedX) * 4;
  return data[idx] * 0.2126 + data[idx + 1] * 0.7152 + data[idx + 2] * 0.0722;
};

const analyzeRegion = (data, width, height, rect) => {
  const x0 = clamp(Math.floor(rect.x), 0, width - 1);
  const y0 = clamp(Math.floor(rect.y), 0, height - 1);
  const x1 = clamp(Math.ceil(rect.x + rect.width), x0 + 1, width);
  const y1 = clamp(Math.ceil(rect.y + rect.height), y0 + 1, height);
  let samples = 0;
  let dark = 0;
  let red = 0;
  let edgeX = 0;
  let edgeY = 0;
  let lumTotal = 0;
  let prevRow = null;

  for (let y = y0; y < y1; y += 2) {
    let prevLum = null;
    const row = [];
    for (let x = x0; x < x1; x += 2) {
      const idx = (y * width + x) * 4;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const lum = r * 0.2126 + g * 0.7152 + b * 0.0722;
      lumTotal += lum;
      if (lum < 130) {
        dark += 1;
      }
      if (r > 80 && r > g * 1.15 && r > b * 1.15) {
        red += 1;
      }
      if (prevLum !== null) {
        edgeX += Math.abs(lum - prevLum);
      }
      prevLum = lum;
      row.push(lum);
      samples += 1;
    }
    if (prevRow) {
      for (let i = 0; i < row.length; i += 1) {
        edgeY += Math.abs(row[i] - prevRow[i]);
      }
    }
    prevRow = row;
  }

  const safeSamples = Math.max(1, samples);
  return {
    darkRatio: dark / safeSamples,
    redRatio: red / safeSamples,
    edgeXRatio: edgeX / (safeSamples * 255),
    edgeYRatio: edgeY / (safeSamples * 255),
    meanLum: lumTotal / safeSamples
  };
};

const normalizeRectToCanvas = (canvas, rect) => {
  const x = clamp(rect.x, 0, canvas.width - 1);
  const y = clamp(rect.y, 0, canvas.height - 1);
  const width = clamp(rect.width, 1, canvas.width - x);
  const height = clamp(rect.height, 1, canvas.height - y);
  return { x, y, width, height };
};

const adjustRectAroundCenter = (canvas, baseRect, settings) => {
  const centerX = baseRect.x + baseRect.width * 0.5;
  const centerY = baseRect.y + baseRect.height * 0.5;
  const width = baseRect.width * settings.scaleX;
  const height = baseRect.height * settings.scaleY;
  const shiftedCenterX = centerX + baseRect.width * settings.shiftX;
  const shiftedCenterY = centerY + baseRect.height * settings.shiftY;
  return normalizeRectToCanvas(canvas, {
    x: shiftedCenterX - width * 0.5,
    y: shiftedCenterY - height * 0.5,
    width,
    height
  });
};

export {
  clamp,
  normalizeAngle,
  cloneCanvas,
  getRectFromNormalized,
  drawOverlayCanvas,
  loadImageBitmap,
  drawImageToCanvas,
  cropCanvas,
  cropCenterSquare,
  rotateCanvas,
  scaleCanvas,
  splitIntoCells,
  preprocessCanvas,
  tightenCropByInk,
  findDigitWindowByEdges,
  hasInk,
  getLuminanceAt,
  analyzeRegion,
  normalizeRectToCanvas,
  adjustRectAroundCenter
};
