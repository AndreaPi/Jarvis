const http = require('node:http');
const path = require('node:path');
const { spawn } = require('node:child_process');

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const requestOk = (url) => new Promise((resolve) => {
  const request = http.get(url, (response) => {
    response.resume();
    resolve((response.statusCode || 0) >= 200 && (response.statusCode || 0) < 300);
  });
  request.on('error', () => resolve(false));
  request.setTimeout(1000, () => {
    request.destroy();
    resolve(false);
  });
});

const requestJsonStatus = (url) => new Promise((resolve) => {
  const request = http.get(url, (response) => {
    const chunks = [];
    response.on('data', (chunk) => chunks.push(chunk));
    response.on('end', () => {
      const statusCode = response.statusCode || 0;
      if (statusCode < 200 || statusCode >= 300) {
        resolve({ reachable: true, health: null, error: `HTTP ${statusCode}` });
        return;
      }
      try {
        resolve({ reachable: true, health: JSON.parse(Buffer.concat(chunks).toString('utf8')), error: '' });
      } catch (error) {
        resolve({
          reachable: true,
          health: null,
          error: `invalid JSON (${error instanceof Error ? error.message : String(error)})`
        });
      }
    });
  });
  request.on('error', () => resolve({ reachable: false, health: null, error: 'unreachable' }));
  request.setTimeout(1000, () => {
    request.destroy();
    resolve({ reachable: false, health: null, error: 'timeout' });
  });
});

const validateQaBackendHealth = (health, rootDir) => {
  const issues = [];
  if (!health || typeof health !== 'object') {
    issues.push('health payload is missing');
  } else {
    if (health.roi_ready !== true) {
      issues.push('roi_ready is not true');
    }
    if (health.digit_ready !== true) {
      issues.push('digit_ready is not true');
    }

    const expectedRoiPath = path.resolve(rootDir, 'backend', 'models', 'roi-rotaug-e30-640.pt');
    const expectedDigitPath = path.resolve(rootDir, 'backend', 'models', 'digit_classifier.pt');
    const actualRoiPath = typeof health.model_path === 'string' ? path.resolve(health.model_path) : '';
    const actualDigitPath = typeof health.digit_model_path === 'string'
      ? path.resolve(health.digit_model_path)
      : '';
    if (actualRoiPath !== expectedRoiPath) {
      issues.push(`ROI checkpoint is ${actualRoiPath || 'unknown'}, expected ${expectedRoiPath}`);
    }
    if (actualDigitPath !== expectedDigitPath) {
      issues.push(`digit checkpoint is ${actualDigitPath || 'unknown'}, expected ${expectedDigitPath}`);
    }
  }

  if (issues.length) {
    throw new Error(`Backend is not ready for canonical QA: ${issues.join('; ')}`);
  }
  return health;
};

const spawnTrackedProcess = (command, args, options = {}) => {
  const child = spawn(command, args, {
    cwd: options.cwd,
    env: options.env || process.env,
    stdio: ['ignore', 'pipe', 'pipe']
  });
  child.stdout.on('data', (chunk) => process.stdout.write(`[${options.label || command}] ${chunk}`));
  child.stderr.on('data', (chunk) => process.stderr.write(`[${options.label || command}] ${chunk}`));
  return {
    stop: async () => {
      if (child.exitCode !== null || child.signalCode !== null) {
        return;
      }
      child.kill('SIGTERM');
      for (let index = 0; index < 30; index += 1) {
        if (child.exitCode !== null || child.signalCode !== null) {
          return;
        }
        await sleep(100);
      }
      if (child.exitCode === null && child.signalCode === null) {
        child.kill('SIGKILL');
      }
    }
  };
};

const ensureQaServices = async ({ rootDir, frontendUrl, backendUrl, attempts = 100 }) => {
  const processes = [];
  const healthUrl = `${backendUrl}/health`;

  try {
    if (!(await requestOk(frontendUrl))) {
      processes.push(spawnTrackedProcess('npm', ['run', 'serve'], {
        cwd: rootDir,
        label: 'frontend'
      }));
    }

    const initialBackend = await requestJsonStatus(healthUrl);
    if (initialBackend.reachable) {
      if (!initialBackend.health) {
        throw new Error(`Backend health check failed: ${initialBackend.error}`);
      }
      validateQaBackendHealth(initialBackend.health, rootDir);
    } else {
      processes.push(spawnTrackedProcess(
        path.join(rootDir, 'backend', '.venv', 'bin', 'uvicorn'),
        ['backend.app:app', '--host', '127.0.0.1', '--port', '8001'],
        { cwd: rootDir, label: 'backend' }
      ));
    }

    let lastBackendError = '';
    for (let attempt = 0; attempt < attempts; attempt += 1) {
      const frontendReady = await requestOk(frontendUrl);
      const backendStatus = await requestJsonStatus(healthUrl);
      let backendReady = false;
      if (backendStatus.health) {
        try {
          validateQaBackendHealth(backendStatus.health, rootDir);
          backendReady = true;
        } catch (error) {
          lastBackendError = error instanceof Error ? error.message : String(error);
        }
      } else if (backendStatus.reachable) {
        lastBackendError = `Backend health check failed: ${backendStatus.error}`;
      }
      if (frontendReady && backendReady) {
        return processes;
      }
      await sleep(250);
    }
    throw new Error(
      `Timed out waiting for canonical QA services.${lastBackendError ? ` ${lastBackendError}` : ''}`
    );
  } catch (error) {
    await Promise.all(processes.map((process) => process.stop()));
    throw error;
  }
};

module.exports = {
  ensureQaServices,
  validateQaBackendHealth
};
