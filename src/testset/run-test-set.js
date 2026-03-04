const parseCsv = (text) => {
  return text
    .trim()
    .split('\n')
    .slice(1)
    .map((line) => line.split(',').map((cell) => cell.trim()))
    .filter((parts) => parts.length >= 2 && parts[0] && parts[1])
    .map(([filename, value]) => ({ filename, value }));
};

const parseMeterValue = (value) => {
  const digits = String(value || '').replace(/\D/g, '');
  if (!digits) {
    return null;
  }
  const parsed = Number.parseInt(digits, 10);
  return Number.isFinite(parsed) ? parsed : null;
};

const computeAbsoluteError = (expected, detected) => {
  const expectedValue = parseMeterValue(expected);
  const detectedValue = parseMeterValue(detected);
  if (!Number.isFinite(expectedValue) || !Number.isFinite(detectedValue)) {
    return null;
  }
  return Math.abs(expectedValue - detectedValue);
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

const renderTestResults = (resultsEl, results, total, histograms = {}) => {
  if (!resultsEl) {
    return;
  }

  resultsEl.innerHTML = '';

  const table = document.createElement('table');
  const header = document.createElement('tr');
  ['File', 'Expected', 'Detected', 'Absolute Error', 'Failure Reason', 'Result'].forEach((label) => {
    const th = document.createElement('th');
    th.textContent = label;
    header.appendChild(th);
  });
  table.appendChild(header);

  results.forEach((result) => {
    const row = document.createElement('tr');
    const statusClass = result.match ? 'pass' : 'fail';
    const errorDisplay = Number.isFinite(result.absoluteError) ? String(result.absoluteError) : '—';
    const failureDisplay = result.match ? '—' : (result.failureReason || 'unknown');
    [result.filename, result.expected, result.detected || '—', errorDisplay, failureDisplay].forEach((value) => {
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
  const absoluteErrors = results
    .map((result) => result.absoluteError)
    .filter((value) => Number.isFinite(value));
  const mae = absoluteErrors.length
    ? absoluteErrors.reduce((sum, value) => sum + value, 0) / absoluteErrors.length
    : null;
  const maeText = mae === null
    ? 'n/a'
    : mae.toFixed(2);
  const exactMatchCount = results.filter((result) => result.match).length;
  const noReadCount = results.filter((result) => !result.detected).length;
  const exactMatchRate = total ? ((exactMatchCount / total) * 100).toFixed(1) : '0.0';
  const noReadRate = total ? ((noReadCount / total) * 100).toFixed(1) : '0.0';
  summary.textContent = `MAE: ${maeText} (${absoluteErrors.length}/${total} reads) | Exact Match: ${exactMatchCount}/${total} (${exactMatchRate}%) | No-read: ${noReadCount}/${total} (${noReadRate}%)`;
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
      let absoluteErrorSum = 0;
      let absoluteErrorCount = 0;
      let noReadCount = 0;
      const failureReasonHistogram = new Map();
      const rejectReasonHistogram = new Map();

      for (let i = 0; i < rows.length; i += 1) {
        const row = rows[i];
        try {
          setStatus(`Reading ${i + 1}/${rows.length}: ${row.filename}`);
          const imageResponse = await fetch(`assets/${row.filename}`, { cache: 'no-store' });
          if (!imageResponse.ok) {
            noReadCount += 1;
            results.push({
              filename: row.filename,
              expected: row.value,
              detected: '',
              match: false,
              absoluteError: null,
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
          const absoluteError = computeAbsoluteError(row.value, detected);
          if (Number.isFinite(absoluteError)) {
            absoluteErrorSum += absoluteError;
            absoluteErrorCount += 1;
          }
          if (!detected) {
            noReadCount += 1;
          }
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
            absoluteError,
            failureReason
          });
        } catch (rowError) {
          rowErrors += 1;
          console.error(`Test row failed for ${row.filename}`, rowError);
          noReadCount += 1;
          results.push({
            filename: row.filename,
            expected: row.value,
            detected: '',
            match: false,
            absoluteError: null,
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
        exactMatchRate: rows.length ? correct / rows.length : 0,
        noReadCount,
        noReadRate: rows.length ? noReadCount / rows.length : 0,
        mae: absoluteErrorCount ? absoluteErrorSum / absoluteErrorCount : null,
        maeRows: absoluteErrorCount,
        maeSum: absoluteErrorSum,
        failureReasons,
        failureTotal,
        rejectReasons,
        rejectTotal
      };
      if (typeof window !== 'undefined') {
        window.__jarvisLastTestSetHistogram = runHistogram;
      }

      const maeText = absoluteErrorCount ? (absoluteErrorSum / absoluteErrorCount).toFixed(2) : 'n/a';
      const exactMatchRateText = rows.length ? ((correct / rows.length) * 100).toFixed(1) : '0.0';
      const noReadRateText = rows.length ? ((noReadCount / rows.length) * 100).toFixed(1) : '0.0';
      setStatus(
        rowErrors
          ? `Done. MAE ${maeText}. Exact Match ${correct}/${rows.length} (${exactMatchRateText}%). No-read ${noReadCount}/${rows.length} (${noReadRateText}%). ${rowErrors} row error(s); see console.`
          : `Done. MAE ${maeText}. Exact Match ${correct}/${rows.length} (${exactMatchRateText}%). No-read ${noReadCount}/${rows.length} (${noReadRateText}%).`
      );
      renderTestResults(resultsEl, results, rows.length, runHistogram);
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
