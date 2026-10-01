// sync-subtree.sh against throwaway repositories, including a host that tracks a file named HEAD.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT = resolve(fileURLToPath(import.meta.url), '..', 'sync-subtree.sh');
const ENV = {
  ...process.env, GIT_AUTHOR_NAME: 't', GIT_AUTHOR_EMAIL: 't@example.invalid', GIT_COMMITTER_NAME: 't',
  GIT_COMMITTER_EMAIL: 't@example.invalid', GIT_CONFIG_GLOBAL: '/dev/null', GIT_CONFIG_NOSYSTEM: '1',
};
const hasGit = spawnSync('git', ['--version']).status === 0;

const git = (cwd, ...args) => execFileSync('git', args, { cwd, env: ENV, encoding: 'utf8' }).trim();
const sync = (cwd, ...args) => spawnSync('sh', [SCRIPT, ...args], { cwd, env: ENV, encoding: 'utf8' });
const put = (dir, rel, text) => { mkdirSync(join(dir, rel, '..'), { recursive: true }); writeFileSync(join(dir, rel), text); };
const commit = (dir, msg) => { git(dir, 'add', '-A'); git(dir, 'commit', '-q', '-m', msg); };

function repos() {
  const root = mkdtempSync(join(tmpdir(), 'ctxr-sync-'));
  const src = join(root, 'src'), host = join(root, 'host');
  for (const d of [src, host]) { mkdirSync(d); git(d, 'init', '-q'); git(d, 'checkout', '-q', '-b', 'main'); }
  put(src, 'context-relay/README.md', 'v1\n');
  put(src, 'other/ignored.txt', 'not synced\n');
  commit(src, 'source v1');
  put(host, 'HEAD', 'ref: refs/heads/master\n');   // a tracked file named HEAD, as in kgirl
  put(host, 'app.py', 'print(1)\n');
  commit(host, 'host');
  return { src, host };
}

test('add, pull with local edits, no-op, refusals', { skip: !hasGit && 'needs git' }, () => {
  const { src, host } = repos();
  const subtree = spawnSync('git', ['subtree', 'add', '--prefix=x', '--squash', src, 'main'], { cwd: host, env: ENV, encoding: 'utf8' });
  assert.notEqual(subtree.status, 0, 'git subtree itself fails in this host (why the script exists)');

  let r = sync(host, src, 'main');
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stdout, /added context-relay\//);
  assert.equal(readFileSync(join(host, 'context-relay/README.md'), 'utf8'), 'v1\n');
  assert.ok(!existsSync(join(host, 'other')), 'only the prefix is copied');
  assert.equal(git(host, 'log', '-1', '--format=%P').split(' ').length, 2, 'a merge commit');
  assert.equal(git(host, 'status', '--porcelain'), '');

  put(host, 'context-relay/LOCAL.md', 'host edit\n');
  commit(host, 'host edit inside the subtree');
  put(src, 'context-relay/README.md', 'v2\n');
  put(src, 'context-relay/tools/new.mjs', 'export {};\n');
  commit(src, 'source v2');

  r = sync(host, src, 'main');
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stdout, /updated context-relay\//);
  assert.equal(readFileSync(join(host, 'context-relay/README.md'), 'utf8'), 'v2\n');
  assert.ok(existsSync(join(host, 'context-relay/tools/new.mjs')));
  assert.equal(readFileSync(join(host, 'context-relay/LOCAL.md'), 'utf8'), 'host edit\n', 'local edits survive');
  assert.equal(readFileSync(join(host, 'HEAD'), 'utf8'), 'ref: refs/heads/master\n', 'the HEAD file is untouched');

  const before = git(host, 'rev-parse', '--verify', 'HEAD^{commit}');
  r = sync(host, src, 'main');
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stdout, /already at/);
  assert.equal(git(host, 'rev-parse', '--verify', 'HEAD^{commit}'), before, 'no commit when nothing changed');

  put(host, 'app.py', 'print(2)\n');
  r = sync(host, src, 'main');
  assert.notEqual(r.status, 0);
  assert.match(r.stderr, /commit or stash/);
});

test('refuses a directory it did not add', { skip: !hasGit && 'needs git' }, () => {
  const { src, host } = repos();
  put(host, 'context-relay/hand-made.md', 'x\n');
  commit(host, 'hand-made dir');
  const r = sync(host, src, 'main');
  assert.notEqual(r.status, 0);
  assert.match(r.stderr, /already exists here/);
});
