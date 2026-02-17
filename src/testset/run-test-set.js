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

const getSelectionLogs = () => {
  if (typeof window === 'undefined' || !Array.isArray(window.__jarvisOcrSelectionLogs)) {
    return [];
  }
  return window.__jarvisOcrSelectionLogs;
};

const incrementHistogram = (histogram, key, amount = 1) => {
  if (!(histogram instanceof Map)) {
    return;
  }
  if (!Number.isFinite(amount) || amount <= 0) {
    return;
  }
  const normalized = key ? String(key) : 'unknown';
  histogram.set(normalized, (histogram.get(normalized) || 0) + amount);
};

const histogramRows = (histogram) => {
  if (!(histogram instanceof Map) || !histogram.size) {
    return [];
  }
  return [...histogram.entries()]
    .map(([reason, count]) => ({ reason, count }))
    .sort((a, b) => b.count - a.count || a.reason.localeCompare(b.reason));
};

const topRejectReason = (selectionLog) => {
  if (!selectionLog || !Array.isArray(selectionLog.rejectSummary) || !selectionLog.rejectSummary.length) {
    return null;
  }
  const top = selectionLog.rejectSummary[0];
  if (!top || !top.reason) {
    return null;
  }
  return String(top.reason);
};

const renderHistogram = (title, rows, total) => {
  const section = document.createElement('section');
  section.className = 'histogram';

  const heading = document.createElement('h4');
  heading.textContent = `${title} (${total})`;
  section.appendChild(heading);

  const table = document.createElement('table');
  const header = document.createElement('tr');
  ['Reason', 'Count', 'Share'].forEach((label) => {
    const th = document.createElement('th');
    th.textContent = label;
    header.appendChild(th);
  });
  table.appendChild(header);

  rows.forEach((entry) => {
    const row = document.createElement('tr');
    const share = total ? `${((entry.count / total) * 100).toFixed(1)}%` : '0.0%';
    [entry.reason, String(entry.count), share].forEach((value) => {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.appendChild(cell);
    });
    table.appendChild(row);
  });

  section.appendChild(table);
  return section;
};

const renderTestResults = (resultsEl, results, correct, total, histograms = {}) => {
  if (!resultsEl) {
    return;
  }

  resultsEl.innerHTML = '';

  const table = document.createElement('table');
  const header = document.createElement('tr');
  ['File', 'Expected', 'Detected', 'Value Match', 'Failure Reason', 'Result'].forEach((label) => {
    const th = document.createElement('th');
    th.textContent = label;
    header.appendChild(th);
  });
  table.appendChild(header);

  results.forEach((result) => {
    const row = document.createElement('tr');
    const statusClass = result.match ? 'pass' : 'fail';
    const scoreDisplay = result.score !== null ? result.score.toFixed(2) : 'n/a';
    const failureDisplay = result.match ? '—' : (result.failureReason || 'unknown');
    [result.filename, result.expected, result.detected || '—', scoreDisplay, failureDisplay].forEach((value) => {
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

  const failureRows = Array.isArray(histograms.failureReasons) ? histograms.failureReasons : [];
  const rejectRows = Array.isArray(histograms.rejectReasons) ? histograms.rejectReasons : [];
  const failureTotal = Number.isFinite(histograms.failureTotal) ? histograms.failureTotal : 0;
  const rejectTotal = Number.isFinite(histograms.rejectTotal) ? histograms.rejectTotal : 0;

  if (failureRows.length) {
    resultsEl.appendChild(renderHistogram('Failure reason histogram', failureRows, failureTotal));
  }
  if (rejectRows.length) {
    resultsEl.appendChild(renderHistogram('OCR reject histogram', rejectRows, rejectTotal));
  }
};

const createTestSetRunner = ({
  runButton,
  statusEl,
  resultsEl,
  runMeterOcr
}) => {
  const formatError = (error) => {
    if (!error) {
      return 'unknown error';
    }
    if (typeof error === 'string') {
      return error;
    }
    if (error instanceof Error) {
      return error.message || error.name || 'error';
    }
    return String(error);
  };

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
      let rowErrors = 0;
      const failureReasonHistogram = new Map();
      const rejectReasonHistogram = new Map();

      for (let i = 0; i < rows.length; i += 1) {
        const row = rows[i];
        try {
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
              failureReason: 'missing-image'
            });
            incrementHistogram(failureReasonHistogram, 'missing-image');
            continue;
          }

          const blob = await imageResponse.blob();
          const file = new File([blob], row.filename, { type: blob.type || 'image/jpeg' });
          const selectionLogCountBefore = getSelectionLogs().length;
          const result = await runMeterOcr(file, (message) => {
            setStatus(`Test ${i + 1}/${rows.length}: ${message}`);
          });
          const selectionLogs = getSelectionLogs();
          const selectionLog = selectionLogs.length > selectionLogCountBefore
            ? selectionLogs[selectionLogs.length - 1]
            : null;

          const detected = result && result.value ? result.value : '';
          const match = detected === row.value;
          const debugScore = computeDebugScore(row.value, detected);
          const rejectReason = topRejectReason(selectionLog);
          const failureReason = match
            ? '—'
            : (
              (detected ? 'mismatch' : null)
              || rejectReason
              || (selectionLog && selectionLog.branchUsed ? `branch:${selectionLog.branchUsed}` : null)
              || 'no-reading'
            );
          if (match) {
            correct += 1;
          } else {
            incrementHistogram(failureReasonHistogram, failureReason);
          }
          if (selectionLog && Array.isArray(selectionLog.rejectSummary)) {
            selectionLog.rejectSummary.forEach((entry) => {
              const reason = entry && entry.reason ? String(entry.reason) : 'unknown';
              const count = Number.isFinite(entry && entry.count) ? entry.count : 1;
              incrementHistogram(rejectReasonHistogram, reason, count);
            });
          }

          results.push({
            filename: row.filename,
            expected: row.value,
            detected,
            match,
            score: debugScore,
            failureReason
          });
        } catch (rowError) {
          rowErrors += 1;
          console.error(`Test row failed for ${row.filename}`, rowError);
          const debugScore = computeDebugScore(row.value, '');
          results.push({
            filename: row.filename,
            expected: row.value,
            detected: '',
            match: false,
            score: debugScore,
            failureReason: `error:${formatError(rowError)}`
          });
          incrementHistogram(failureReasonHistogram, `error:${formatError(rowError)}`);
        }
      }

      const failureReasons = histogramRows(failureReasonHistogram);
      const rejectReasons = histogramRows(rejectReasonHistogram);
      const failureTotal = failureReasons.reduce((sum, entry) => sum + entry.count, 0);
      const rejectTotal = rejectReasons.reduce((sum, entry) => sum + entry.count, 0);
      const runHistogram = {
        generatedAt: new Date().toISOString(),
        totalRows: rows.length,
        correct,
        failed: rows.length - correct,
        failureReasons,
        failureTotal,
        rejectReasons,
        rejectTotal
      };
      if (typeof window !== 'undefined') {
        window.__jarvisLastTestSetHistogram = runHistogram;
      }

      setStatus(
        rowErrors
          ? `Done. ${correct}/${rows.length} correct. ${rowErrors} row error(s); see console.`
          : `Done. ${correct}/${rows.length} correct.`
      );
      renderTestResults(resultsEl, results, correct, rows.length, runHistogram);
    } catch (error) {
      console.error(error);
      setStatus(`Test run failed: ${formatError(error)}.`);
    } finally {
      runButton.disabled = false;
    }
  };

  return {
    runTestSet
  };
};

export { createTestSetRunner };
