const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const test = require('node:test');

const sourceScript = path.resolve(__dirname, 'package-tier1-artifacts.sh');

test('writes a checksum that can be verified from the archive directory', (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'jarvis-package-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  fs.mkdirSync(path.join(root, 'scripts'), { recursive: true });
  fs.mkdirSync(path.join(root, 'assets'), { recursive: true });
  fs.copyFileSync(sourceScript, path.join(root, 'scripts', 'package-tier1-artifacts.sh'));
  fs.writeFileSync(path.join(root, 'assets', 'meter_fixture.JPEG'), 'fixture\n');

  const packageResult = spawnSync(
    'bash',
    [
      path.join(root, 'scripts', 'package-tier1-artifacts.sh'),
      '--output-dir',
      'output/release',
      '--archive-name',
      'fixture-artifacts'
    ],
    { cwd: root, encoding: 'utf8' }
  );
  assert.equal(packageResult.status, 0, packageResult.stderr);

  const outputDir = path.join(root, 'output', 'release');
  const checksumName = 'fixture-artifacts.tar.gz.sha256';
  const checksum = fs.readFileSync(path.join(outputDir, checksumName), 'utf8');
  assert.match(checksum, /^[a-f0-9]{64}  fixture-artifacts\.tar\.gz\n$/);
  assert.equal(checksum.includes(root), false);

  const verifyResult = spawnSync('sha256sum', ['-c', checksumName], {
    cwd: outputDir,
    encoding: 'utf8'
  });
  assert.equal(verifyResult.status, 0, verifyResult.stderr);
  assert.equal(verifyResult.stdout, 'fixture-artifacts.tar.gz: OK\n');
});
