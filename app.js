const photoInput = document.getElementById('photo-input');
const photoPreview = document.getElementById('photo-preview');
const readBtn = document.getElementById('read-btn');
const ocrStatus = document.getElementById('ocr-status');
const readingInput = document.getElementById('reading-input');
const dateInput = document.getElementById('date-input');
const fromInput = document.getElementById('from-input');
const toInput = document.getElementById('to-input');
const subjectInput = document.getElementById('subject-input');
const bodyInput = document.getElementById('body-input');
const regenBtn = document.getElementById('regen-btn');
const sendBtn = document.getElementById('send-btn');
const mailtoLink = document.getElementById('mailto-link');
const runTestBtn = document.getElementById('run-test-btn');
const testStatus = document.getElementById('test-status');
const testResults = document.getElementById('test-results');

const DEFAULT_SUBJECT = 'Lettura acqua da F9C397';
const CUSTOMER_DETAILS = {
  name: 'Andrea Panizza',
  address: 'Via delle cinque giornate 15 piano 1 e 1/2',
  city: 'Firenze FI',
  code: 'F9C397'
};

const OCR_CONFIG = {
  maxDimension: 1400,
  meterCropScale: 0.85,
  digitCrops: [
    { name: 'primary', x: 0.02, y: 0.16, width: 0.62, height: 0.36 },
    { name: 'wide', x: 0.0, y: 0.08, width: 0.68, height: 0.48 }
  ],
  preferredDigits: 4,
  digitCellCount: 4,
  digitCellOverlap: 0.08,
  minDigitWidth: 96,
  minDigits: 3,
  earlyStopScore: 0.84,
  fallbackScoreThreshold: 0.72,
  fallbackCandidates: 2,
  minScaleWidth: 320
};

let currentPhotoFile = null;
let bodyTouched = false;
let subjectTouched = false;
let ocrWorker = null;
let ocrLogger = null;

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

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
  const ctx = canvas.getContext('2d');
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
  const ctx = canvas.getContext('2d');
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

const tightenCropByInk = (source) => {
  const ctx = source.getContext('2d');
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
  const colThreshold = meanCols + (maxCols - meanCols) * 0.45;
  const rowThreshold = meanRows + (maxRows - meanRows) * 0.45;

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
  if (areaRatio < 0.15 || areaRatio > 0.95) {
    return source;
  }

  return cropCanvas(source, { x: left, y: top, width: cropWidth, height: cropHeight });
};

const buildDigitCandidates = (source) => {
  const meterCrop = cropCenterSquare(source, OCR_CONFIG.meterCropScale);
  const rotations = [0, 90, 180, 270];
  const candidates = [];

  rotations.forEach((angle) => {
    const rotated = rotateCanvas(meterCrop, angle);
    OCR_CONFIG.digitCrops.forEach((crop) => {
      const rect = {
        x: rotated.width * crop.x,
        y: rotated.height * crop.y,
        width: rotated.width * crop.width,
        height: rotated.height * crop.height
      };
      let digitCanvas = cropCanvas(rotated, rect);
      digitCanvas = tightenCropByInk(digitCanvas);
      digitCanvas = scaleCanvas(digitCanvas, OCR_CONFIG.minScaleWidth);
      candidates.push({ canvas: digitCanvas, label: `${angle}-${crop.name}` });
    });
  });

  return candidates;
};

const getWorker = async (logger) => {
  ocrLogger = logger;
  if (ocrWorker) {
    return ocrWorker;
  }
  ocrWorker = Tesseract.createWorker({
    logger: (message) => {
      if (ocrLogger) {
        ocrLogger(message);
      }
    }
  });
  await ocrWorker.load();
  await ocrWorker.loadLanguage('eng');
  await ocrWorker.initialize('eng');
  await ocrWorker.setParameters({
    tessedit_char_whitelist: '0123456789',
    tessedit_pageseg_mode: Tesseract.PSM.SINGLE_WORD,
    classify_bln_numeric_mode: 1
  });
  return ocrWorker;
};

const buildCandidateScores = (data, canvas) => {
  const candidates = [];

  const pushCandidate = (value, confidence, areaRatio) => {
    if (!value) {
      return;
    }
    candidates.push({
      value,
      confidence: confidence ?? data.confidence ?? 0,
      areaRatio: areaRatio ?? 0
    });
  };

  const textMatches = (data.text || '').match(/\d+/g);
  if (textMatches) {
    textMatches.forEach((chunk) => {
      if (chunk.length >= OCR_CONFIG.minDigits) {
        pushCandidate(chunk, data.confidence, 0.15);
        if (chunk.length > OCR_CONFIG.preferredDigits) {
          pushCandidate(chunk.slice(0, OCR_CONFIG.preferredDigits), data.confidence, 0.15);
          pushCandidate(chunk.slice(-OCR_CONFIG.preferredDigits), data.confidence, 0.15);
        }
      }
    });
  }

  if (data.words) {
    data.words.forEach((word) => {
      const digits = (word.text || '').replace(/\D/g, '');
      if (digits.length >= OCR_CONFIG.minDigits) {
        const box = word.bbox || {};
        const area = (box.x1 - box.x0 || 0) * (box.y1 - box.y0 || 0);
        const ratio = area / (canvas.width * canvas.height);
        pushCandidate(digits, word.confidence, ratio);
        if (digits.length > OCR_CONFIG.preferredDigits) {
          pushCandidate(digits.slice(0, OCR_CONFIG.preferredDigits), word.confidence, ratio);
          pushCandidate(digits.slice(-OCR_CONFIG.preferredDigits), word.confidence, ratio);
        }
      }
    });
  }

  return candidates;
};

const scoreCandidate = (candidate) => {
  const length = candidate.value.length;
  const lengthScore = length === OCR_CONFIG.preferredDigits ? 1 : length === 5 ? 0.75 : length === 3 ? 0.45 : 0.2;
  const confidenceScore = clamp(candidate.confidence / 100, 0, 1);
  const areaScore = clamp(candidate.areaRatio * 4, 0, 1);
  return confidenceScore * 0.6 + lengthScore * 0.3 + areaScore * 0.1;
};

const selectBestReading = (data, canvas) => {
  const candidates = buildCandidateScores(data, canvas);
  const preferred = candidates.filter((candidate) => candidate.value.length === OCR_CONFIG.preferredDigits);
  const shortlist = preferred.length ? preferred : candidates;
  let best = null;
  shortlist.forEach((candidate) => {
    const score = scoreCandidate(candidate);
    if (!best || score > best.score) {
      best = { ...candidate, score };
    }
  });
  return best;
};

const readDigitsByCells = async (worker, source, setProgress) => {
  const trimmed = tightenCropByInk(source);
  const cellCanvases = splitIntoCells(trimmed, OCR_CONFIG.digitCellCount, OCR_CONFIG.digitCellOverlap);
  const digits = [];
  let confidenceTotal = 0;

  for (let i = 0; i < cellCanvases.length; i += 1) {
    if (setProgress) {
      setProgress(`Refining digits (${i + 1}/${cellCanvases.length})`);
    }
    let cell = preprocessCanvas(cellCanvases[i], 'binary');
    cell = scaleCanvas(cell, OCR_CONFIG.minDigitWidth);
    const { data } = await worker.recognize(cell);
    const symbol = data.symbols && data.symbols.find((item) => /\d/.test(item.text || ''));
    const match = (data.text || '').match(/\d/);
    const digit = match ? match[0] : symbol ? symbol.text : '';
    if (!digit) {
      return null;
    }
    digits.push(digit);
    const confidence = symbol && typeof symbol.confidence === 'number' ? symbol.confidence : data.confidence ?? 0;
    confidenceTotal += confidence;
  }

  const value = digits.join('');
  const averageConfidence = confidenceTotal / cellCanvases.length;
  return {
    value,
    confidence: averageConfidence,
    areaRatio: 0.25,
    score: scoreCandidate({ value, confidence: averageConfidence, areaRatio: 0.25 }) + 0.06
  };
};

const runMeterOcr = async (file, setProgress) => {
  const image = await loadImageBitmap(file);
  const baseCanvas = drawImageToCanvas(image, OCR_CONFIG.maxDimension);
  const candidates = buildDigitCandidates(baseCanvas);
  const modes = ['binary', 'soft'];
  let bestResult = null;
  const candidateScores = new Map();
  let pass = 0;
  const worker = await getWorker((message) => {
    if (message.status === 'recognizing text') {
      const progress = Math.round(message.progress * 100);
      if (setProgress) {
        setProgress(`Reading image ${progress}% (pass ${pass}/${candidates.length * modes.length})`);
      }
    }
  });

  for (const candidate of candidates) {
    if (!candidateScores.has(candidate.label)) {
      candidateScores.set(candidate.label, { score: -1, canvas: candidate.canvas });
    }
    for (const mode of modes) {
      pass += 1;
      if (setProgress) {
        setProgress(`Analyzing meter (${pass}/${candidates.length * modes.length})`);
      }
      const processed = preprocessCanvas(candidate.canvas, mode);
      const { data } = await worker.recognize(processed);
      const candidateBest = selectBestReading(data, processed);
      if (candidateBest && (!bestResult || candidateBest.score > bestResult.score)) {
        bestResult = candidateBest;
      }
      if (candidateBest) {
        const existing = candidateScores.get(candidate.label);
        if (!existing || candidateBest.score > existing.score) {
          candidateScores.set(candidate.label, { score: candidateBest.score, canvas: candidate.canvas });
        }
      }
      if (bestResult && bestResult.score >= OCR_CONFIG.earlyStopScore && bestResult.value.length === OCR_CONFIG.preferredDigits) {
        return bestResult;
      }
    }
  }

  if (bestResult && bestResult.score >= OCR_CONFIG.fallbackScoreThreshold && bestResult.value.length === OCR_CONFIG.preferredDigits) {
    return bestResult;
  }

  const fallbackPool = [...candidateScores.values()]
    .sort((a, b) => b.score - a.score)
    .slice(0, OCR_CONFIG.fallbackCandidates);

  if (fallbackPool.length) {
    await worker.setParameters({
      tessedit_pageseg_mode: Tesseract.PSM.SINGLE_CHAR,
      classify_bln_numeric_mode: 1
    });
    for (const candidate of fallbackPool) {
      if (setProgress) {
        setProgress('Refining digits...');
      }
      const processed = preprocessCanvas(candidate.canvas, 'binary');
      const refined = await readDigitsByCells(worker, processed, setProgress);
      if (refined && (!bestResult || refined.score > bestResult.score)) {
        bestResult = refined;
      }
    }
    await worker.setParameters({
      tessedit_pageseg_mode: Tesseract.PSM.SINGLE_WORD,
      classify_bln_numeric_mode: 1
    });
  }

  return bestResult;
};

const formatItalianDate = (dateValue) => {
  if (!dateValue) {
    return '';
  }
  const [year, month, day] = dateValue.split('-').map(Number);
  if (!year || !month || !day) {
    return '';
  }
  const localDate = new Date(year, month - 1, day);
  return new Intl.DateTimeFormat('it-IT').format(localDate);
};

const buildBodyTemplate = (dateDisplay, readingValue) => {
  const safeReading = readingValue || '____';
  return [
    `Intestatario: ${CUSTOMER_DETAILS.name}`,
    CUSTOMER_DETAILS.address,
    CUSTOMER_DETAILS.city,
    `Codice Utente: ${CUSTOMER_DETAILS.code}`,
    `Data: ${dateDisplay}`,
    `Lettura: ${safeReading}`
  ].join('\n');
};

const replaceLine = (body, label, value) => {
  if (!body.trim()) {
    return '';
  }
  const lines = body.split('\n');
  const index = lines.findIndex((line) => line.trim().startsWith(label));
  const nextLine = `${label} ${value}`;
  if (index >= 0) {
    lines[index] = nextLine;
  } else {
    lines.push(nextLine);
  }
  return lines.join('\n');
};

const setStatus = (message) => {
  ocrStatus.textContent = message;
};

const setTestStatus = (message) => {
  if (testStatus) {
    testStatus.textContent = message;
  }
};

const buildEmailDraft = () => {
  return {
    to: toInput.value.trim(),
    subject: subjectInput.value.trim() || DEFAULT_SUBJECT,
    body: bodyInput.value.trim()
  };
};

const updateMailLinks = () => {
  const { to, subject, body } = buildEmailDraft();
  const gmailUrl = `https://mail.google.com/mail/?view=cm&fs=1&to=${encodeURIComponent(to)}&su=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
  sendBtn.dataset.gmailUrl = gmailUrl;
  mailtoLink.href = `mailto:${encodeURIComponent(to)}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
};

const parseCsv = (text) => {
  return text
    .trim()
    .split('\n')
    .slice(1)
    .map((line) => line.split(',').map((cell) => cell.trim()))
    .filter((parts) => parts.length >= 2 && parts[0] && parts[1])
    .map(([filename, value]) => ({ filename, value }));
};

const renderTestResults = (results, correct, total) => {
  if (!testResults) {
    return;
  }

  testResults.innerHTML = '';

  const table = document.createElement('table');
  const header = document.createElement('tr');
  ['File', 'Expected', 'Detected', 'Score', 'Result'].forEach((label) => {
    const th = document.createElement('th');
    th.textContent = label;
    header.appendChild(th);
  });
  table.appendChild(header);

  results.forEach((result) => {
    const row = document.createElement('tr');
    const statusClass = result.match ? 'pass' : 'fail';
    const scoreDisplay = result.score !== null ? result.score.toFixed(2) : 'n/a';
    [result.filename, result.expected, result.detected || '—', scoreDisplay].forEach((value) => {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.appendChild(cell);
    });
    const statusCell = document.createElement('td');
    statusCell.textContent = result.match ? 'Pass' : 'Fail';
    statusCell.className = statusClass;
    row.appendChild(statusCell);
    table.appendChild(row);
  });

  testResults.appendChild(table);

  const summary = document.createElement('p');
  summary.className = 'summary';
  summary.textContent = `Accuracy: ${correct}/${total} (${Math.round((correct / total) * 100)}%)`;
  testResults.appendChild(summary);
};

const runTestSet = async () => {
  if (!runTestBtn) {
    return;
  }
  runTestBtn.disabled = true;
  setTestStatus('Loading test set...');
  if (testResults) {
    testResults.innerHTML = '';
  }

  try {
    const csvResponse = await fetch('assets/meter_readings.csv', { cache: 'no-store' });
    if (!csvResponse.ok) {
      throw new Error('Unable to load meter_readings.csv');
    }
    const csvText = await csvResponse.text();
    const rows = parseCsv(csvText);
    if (!rows.length) {
      throw new Error('No test rows found.');
    }

    const results = [];
    let correct = 0;

    for (let i = 0; i < rows.length; i += 1) {
      const row = rows[i];
      setTestStatus(`Reading ${i + 1}/${rows.length}: ${row.filename}`);
      const imageResponse = await fetch(`assets/${row.filename}`, { cache: 'no-store' });
      if (!imageResponse.ok) {
        results.push({
          filename: row.filename,
          expected: row.value,
          detected: '',
          match: false,
          score: null
        });
        continue;
      }
      const blob = await imageResponse.blob();
      const file = new File([blob], row.filename, { type: blob.type || 'image/jpeg' });
      const result = await runMeterOcr(file, (message) => {
        setTestStatus(`Test ${i + 1}/${rows.length}: ${message}`);
      });
      const detected = result && result.value ? result.value : '';
      const match = detected === row.value;
      if (match) {
        correct += 1;
      }
      results.push({
        filename: row.filename,
        expected: row.value,
        detected,
        match,
        score: result ? result.score : null
      });
    }

    setTestStatus(`Done. ${correct}/${rows.length} correct.`);
    renderTestResults(results, correct, rows.length);
  } catch (error) {
    console.error(error);
    setTestStatus('Test run failed. Check the console for details.');
  } finally {
    runTestBtn.disabled = false;
  }
};

const updateSubject = ({ force = false } = {}) => {
  if (force || !subjectTouched) {
    subjectInput.value = DEFAULT_SUBJECT;
    subjectTouched = false;
  }
  updateMailLinks();
};

const updateBody = ({ force = false } = {}) => {
  const dateDisplay = formatItalianDate(dateInput.value);
  const readingValue = readingInput.value.trim();

  if (force || !bodyTouched) {
    bodyInput.value = buildBodyTemplate(dateDisplay, readingValue);
    bodyTouched = false;
  } else {
    let updated = bodyInput.value;
    updated = replaceLine(updated, 'Data:', dateDisplay);
    updated = replaceLine(updated, 'Lettura:', readingValue || '____');
    bodyInput.value = updated || buildBodyTemplate(dateDisplay, readingValue);
  }

  updateMailLinks();
};

photoInput.addEventListener('click', () => {
  photoInput.value = '';
});

photoInput.addEventListener('change', () => {
  const file = photoInput.files && photoInput.files[0];
  if (!file) {
    currentPhotoFile = null;
    photoPreview.innerHTML = '<p class="muted">No photo loaded yet.</p>';
    setStatus('Waiting for a photo.');
    return;
  }

  currentPhotoFile = file;
  const reader = new FileReader();
  reader.onload = () => {
    photoPreview.innerHTML = '';
    const img = document.createElement('img');
    img.src = reader.result;
    img.alt = 'Water meter preview';
    photoPreview.appendChild(img);
  };
  reader.onerror = () => {
    photoPreview.innerHTML = '<p class="muted">Preview unavailable.</p>';
  };
  reader.readAsDataURL(file);
  setStatus('Photo ready. Click "Read meter".');
});

readBtn.addEventListener('click', async () => {
  if (!currentPhotoFile) {
    setStatus('Upload a photo first.');
    return;
  }

  if (!window.Tesseract) {
    setStatus('OCR library not available. Check your connection and try again.');
    return;
  }

  readBtn.disabled = true;
  setStatus('Reading image...');

  try {
    const result = await runMeterOcr(currentPhotoFile, (message) => setStatus(message));
    if (result && result.value) {
      readingInput.value = result.value;
      setStatus(`Reading detected: ${result.value}. Review if needed.`);
      updateBody();
    } else {
      setStatus('No clear reading detected. Enter it manually.');
    }
  } catch (error) {
    console.error(error);
    setStatus('OCR failed. Enter the reading manually.');
  } finally {
    readBtn.disabled = false;
  }
});

readingInput.addEventListener('input', () => {
  const sanitized = readingInput.value.replace(/\D/g, '');
  if (sanitized !== readingInput.value) {
    readingInput.value = sanitized;
  }
  updateBody();
});

readingInput.addEventListener('blur', () => {
  updateBody();
});

dateInput.addEventListener('change', () => {
  updateBody();
});

toInput.addEventListener('input', () => {
  updateMailLinks();
});

subjectInput.addEventListener('input', () => {
  subjectTouched = true;
  updateMailLinks();
});

bodyInput.addEventListener('input', () => {
  bodyTouched = true;
  updateMailLinks();
});

if (runTestBtn) {
  runTestBtn.addEventListener('click', () => {
    runTestSet();
  });
}

regenBtn.addEventListener('click', () => {
  bodyTouched = false;
  subjectTouched = false;
  updateSubject({ force: true });
  updateBody({ force: true });
});

sendBtn.addEventListener('click', () => {
  if (!sendBtn.dataset.gmailUrl) {
    updateMailLinks();
  }
  window.open(sendBtn.dataset.gmailUrl, '_blank', 'noopener');
});

const init = () => {
  const now = new Date();
  const isoDate = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, '0'),
    String(now.getDate()).padStart(2, '0')
  ].join('-');

  dateInput.value = isoDate;
  fromInput.value = fromInput.value || 'andrea.panizza75@gmail.com';

  updateSubject({ force: true });
  updateBody({ force: true });
  updateMailLinks();
};

init();
