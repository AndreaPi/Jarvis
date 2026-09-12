const assert = require('node:assert/strict');
const test = require('node:test');
const { trackedProcess, waitFor } = require('./export-full-image-digit-shadow-qa.cjs');

test('failed frontend/backend readiness stops the process before losing its handle', async () => {
  for (const label of ['frontend', 'shadow backend']) {
    const tracked = trackedProcess(process.execPath, ['-e', 'setInterval(() => {}, 1000)']);
    try {
      await assert.rejects(waitFor(async () => { throw new Error('health incomplete'); }, tracked, label, 10), /was not ready.*health incomplete/);
      assert.ok(tracked.child.exitCode !== null || tracked.child.signalCode !== null);
      assert.throws(() => process.kill(tracked.child.pid, 0), { code: 'ESRCH' });
    } finally {
      await tracked.stop();
    }
  }
});

test('early exit and spawn failure are reported and cleaned up', async () => {
  const exited = {
    child: { exitCode: 1, signalCode: null },
    output: () => ({ stdout: '', stderr: 'invalid checkpoint' }),
    stop: async () => { exited.stopped = true; }
  };
  await assert.rejects(waitFor(async () => true, exited, 'backend'), /exited early.*\nstdout:\n\nstderr:\ninvalid checkpoint/);
  assert.equal(exited.stopped, true);
  const missing = trackedProcess('/nonexistent-jarvis-qa-python', []);
  await assert.rejects(waitFor(async () => { throw new Error('not ready'); }, missing, 'backend', 2000), /could not start/);
});

test('successful readiness leaves process ownership with the caller', async () => {
  const tracked = {
    child: { exitCode: null, signalCode: null },
    stop: async () => { throw new Error('must not stop a ready service'); }
  };
  assert.deepEqual(await waitFor(async () => ({ ready: true }), tracked, 'backend'), { ready: true });
});
