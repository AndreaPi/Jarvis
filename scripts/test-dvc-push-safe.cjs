const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const test = require('node:test');

const sourceScript = path.resolve(__dirname, 'dvc-push-safe.sh');

const createFixture = () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'jarvis-dvc-push-'));
  const scriptsDir = path.join(root, 'scripts');
  const venvBin = path.join(root, 'backend', '.venv', 'bin');
  const fakeBin = path.join(root, 'fake-bin');
  fs.mkdirSync(scriptsDir, { recursive: true });
  fs.mkdirSync(venvBin, { recursive: true });
  fs.mkdirSync(fakeBin, { recursive: true });
  fs.copyFileSync(sourceScript, path.join(scriptsDir, 'dvc-push-safe.sh'));
  fs.writeFileSync(
    path.join(venvBin, 'activate'),
    'export PATH="$FAKE_DVC_BIN:$PATH"\n'
  );
  fs.writeFileSync(
    path.join(fakeBin, 'dvc'),
    `#!/usr/bin/env bash
set -euo pipefail
if [[ "\${1:-}" == "config" && "\${2:-}" == "core.remote" ]]; then
  printf '%s\\n' fixture
elif [[ "\${1:-}" == "remote" && "\${2:-}" == "list" ]]; then
  printf 'fixture\\t%s\\n' "$FAKE_DVC_REMOTE_URL"
elif [[ "\${1:-}" == "push" ]]; then
  printf '%s\\n' "$*" >> "$FAKE_DVC_LOG"
else
  exit 2
fi
`
  );
  fs.chmodSync(path.join(fakeBin, 'dvc'), 0o755);
  return {
    root,
    script: path.join(scriptsDir, 'dvc-push-safe.sh'),
    fakeBin,
    log: path.join(root, 'dvc.log')
  };
};

const runFixture = (fixture, remoteUrl) => spawnSync(
  'bash',
  [fixture.script, 'artifact.dvc'],
  {
    cwd: fixture.root,
    encoding: 'utf8',
    env: {
      ...process.env,
      FAKE_DVC_BIN: fixture.fakeBin,
      FAKE_DVC_LOG: fixture.log,
      FAKE_DVC_REMOTE_URL: remoteUrl
    }
  }
);

test('rejects file URL DVC remotes without invoking push', (t) => {
  const fixture = createFixture();
  t.after(() => fs.rmSync(fixture.root, { recursive: true, force: true }));

  for (const remoteUrl of ['file:///tmp/dvc-cache', 'FILE:///tmp/dvc-cache']) {
    const result = runFixture(fixture, remoteUrl);
    assert.equal(result.status, 1, result.stderr);
    assert.match(result.stderr, /is a local path/);
  }
  assert.equal(fs.existsSync(fixture.log), false);
});

test('allows an object-store URL and forwards push arguments', (t) => {
  const fixture = createFixture();
  t.after(() => fs.rmSync(fixture.root, { recursive: true, force: true }));

  const result = runFixture(fixture, 's3://jarvis-artifacts/dvc');

  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Using DVC remote 'fixture'/);
  assert.equal(fs.readFileSync(fixture.log, 'utf8'), 'push artifact.dvc\n');
});
