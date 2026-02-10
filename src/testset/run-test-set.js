const parseCsv = (text) => {
  return text
    .trim()
    .split('\n')
    .slice(1)
    .map((line) => line.split(',').map((cell) => cell.trim()))
    .filter((parts) => parts.length >= 2 && parts[0] && parts[1])
    .map(([filename, value]) => ({ filename, value }));
};

const renderTestResults = (resultsEl, results, correct, total) => {
  if (!resultsEl) {
    return;
  }

  resultsEl.innerHTML = '';

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
          setStatus(`Test ${i + 1}/${rows.length}: ${message}`);
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
