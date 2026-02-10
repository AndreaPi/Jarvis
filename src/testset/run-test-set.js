const parseCsv = (text) => {
  return text
    .trim()
    .split('\n')
    .slice(1)
    .map((line) => line.split(',').map((cell) => cell.trim()))
    .filter((parts) => parts.length >= 2 && parts[0] && parts[1])
    .map(([filename, value]) => ({ filename, value }));
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const computeDebugScore = (expected, detected) => {
  const expectedDigits = String(expected || '').replace(/\D/g, '');
  const detectedDigits = String(detected || '').replace(/\D/g, '');
  if (!expectedDigits) {
    return null;
  }
  if (!detectedDigits) {
    return 0;
  }

  const expectedNumber = Number.parseInt(expectedDigits, 10);
  const detectedNumber = Number.parseInt(detectedDigits, 10);
  if (Number.isFinite(expectedNumber) && Number.isFinite(detectedNumber)) {
    const denominator = Math.max(Math.abs(expectedNumber), 1);
    const mse = ((expectedNumber - detectedNumber) / denominator) ** 2;
    return clamp(1 - mse, 0, 1);
  }

  const maxLength = Math.max(expectedDigits.length, detectedDigits.length);
  if (!maxLength) {
    return null;
  }
  let mismatches = 0;
  for (let i = 0; i < maxLength; i += 1) {
    if ((expectedDigits[i] || '') !== (detectedDigits[i] || '')) {
      mismatches += 1;
    }
  }
  return clamp(1 - mismatches / maxLength, 0, 1);
};

const renderTestResults = (resultsEl, results, correct, total) => {
  if (!resultsEl) {
    return;
  }

  resultsEl.innerHTML = '';

  const table = document.createElement('table');
  const header = document.createElement('tr');
  ['File', 'Expected', 'Detected', 'Score', 'OCR', 'Result'].forEach((label) => {
    const th = document.createElement('th');
    th.textContent = label;
    header.appendChild(th);
  });
  table.appendChild(header);

  results.forEach((result) => {
    const row = document.createElement('tr');
    const statusClass = result.match ? 'pass' : 'fail';
    const scoreDisplay = result.score !== null ? result.score.toFixed(2) : 'n/a';
    const ocrDisplay = result.ocrScore !== null ? result.ocrScore.toFixed(2) : 'n/a';
    [result.filename, result.expected, result.detected || '—', scoreDisplay, ocrDisplay].forEach((value) => {
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

  resultsEl.appendChild(table);

  const summary = document.createElement('p');
  summary.className = 'summary';
  summary.textContent = `Accuracy: ${correct}/${total} (${Math.round((correct / total) * 100)}%)`;
  resultsEl.appendChild(summary);
};

const createTestSetRunner = ({
  runButton,
  statusEl,
  resultsEl,
  runMeterOcr
}) => {
  const setStatus = (message) => {
    if (statusEl) {
      statusEl.textContent = message;
    }
  };

  const runTestSet = async () => {
    if (!runButton) {
      return;
    }

    runButton.disabled = true;
    setStatus('Loading test set...');
    if (resultsEl) {
      resultsEl.innerHTML = '';
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
        setStatus(`Reading ${i + 1}/${rows.length}: ${row.filename}`);
        const imageResponse = await fetch(`assets/${row.filename}`, { cache: 'no-store' });
        if (!imageResponse.ok) {
          const debugScore = computeDebugScore(row.value, '');
          results.push({
            filename: row.filename,
            expected: row.value,
            detected: '',
            match: false,
            score: debugScore,
            ocrScore: null
          });
          continue;
        }

        const blob = await imageResponse.blob();
        const file = new File([blob], row.filename, { type: blob.type || 'image/jpeg' });
        const result = await runMeterOcr(file, (message) => {
          setStatus(`Test ${i + 1}/${rows.length}: ${message}`);
        });

        const detected = result && result.value ? result.value : '';
        const match = detected === row.value;
        const debugScore = computeDebugScore(row.value, detected);
        if (match) {
          correct += 1;
        }

        results.push({
          filename: row.filename,
          expected: row.value,
          detected,
          match,
          score: debugScore,
          ocrScore: result ? result.score : null
        });
      }

      setStatus(`Done. ${correct}/${rows.length} correct.`);
      renderTestResults(resultsEl, results, correct, rows.length);
    } catch (error) {
      console.error(error);
      setStatus('Test run failed. Check the console for details.');
    } finally {
      runButton.disabled = false;
    }
  };

  return {
    runTestSet
  };
};

export { createTestSetRunner };
