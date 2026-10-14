// The kgirl Atlas bridge, against a fixture built from Atlas's own schema and, when
// KGIRL_SRC points at kgirl/src, against a database written by kgirl itself.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { atlasModel } from '../lib/atlas.mjs';
import { normalizeModel, propagate, runChecks } from '../lib/engine.mjs';
import { callTool } from '../server/tools.mjs';
import { workspace } from './fixtures.mjs';

const HERE = resolve(fileURLToPath(import.meta.url), '..');
const sqlite = process.getBuiltinModule?.('node:sqlite');
const skip = sqlite ? false : 'needs node:sqlite (Node 22.13+)';
const sha = (p) => createHash('sha256').update(readFileSync(p)).digest('hex');

// Three repos: app imports core by a dotted name; other imports core's db.py by a bare
// name (an Atlas guess); core and other share one cloned body; one core file fails to parse.
function fixtureDb() {
  const { root } = workspace();
  const path = join(root, 'atlas.db');
  const db = new sqlite.DatabaseSync(path);
  db.exec(readFileSync(join(HERE, 'atlas-schema.sql'), 'utf8'));
  db.exec(`
    INSERT INTO repos(id, name, root, origin, head) VALUES
      (1, 'core', '/r/core', 'https://tok@github.com/acme/core.git', '${'c'.repeat(40)}'),
      (2, 'app', '/r/app', '', 'not-a-sha'),
      (3, 'Other Repo', '/r/other', 'git@github.com:acme/other.git', '');
    INSERT INTO files(id, repo_id, path, lang, parse_error) VALUES
      (10, 1, 'core/lib.py', 'python', NULL), (11, 1, 'db.py', 'python', NULL), (12, 1, 'core/broken.py', 'python', 'SyntaxError'),
      (13, 1, 'core/util.py', 'python', NULL),
      (20, 2, 'app/main.py', 'python', NULL), (30, 3, 'x.py', 'python', NULL);
    INSERT INTO imports(id, file_id, target, names, level, line, kind, resolved_file_id) VALUES
      (1, 20, 'core.lib', 'f', 0, 3, 'import', 10),
      (2, 30, 'db', 'get', 0, 1, 'import', 11),
      (3, 13, 'lib', 'f', 1, 2, 'import', 10),
      (4, 20, 'os', '', 0, 1, 'import', NULL);
    INSERT INTO symbols(id, file_id, kind, name, qualname, body_hash) VALUES
      (100, 10, 'function', 'f', 'f', 'h1'), (101, 30, 'function', 'g', 'g', 'h1'), (102, 20, 'function', 'm', 'm', '');
  `);
  db.close();
  return { root, path };
}

test('repo granularity: import and clone couplings, credentials and junk heads dropped', { skip }, () => {
  const { path } = fixtureDb();
  const before = sha(path);
  const { model, evidence, warnings } = atlasModel(path);
  assert.deepEqual(model.components.map((c) => c.id), ['repo:app', 'repo:core', 'repo:other-repo']);
  const core = model.components.find((c) => c.id === 'repo:core');
  assert.deepEqual({ remote: core.remote, commit: core.commit, files: core.files, parseErrors: core.parseErrors }, { remote: 'github.com/acme/core', commit: 'c'.repeat(12), files: 4, parseErrors: 1 });
  assert.equal(model.components.find((c) => c.id === 'repo:app').commit, undefined);
  const e = new Set(model.edges.map((x) => x.join('>')));
  for (const want of ['repo:core>repo:app>change', 'repo:core>repo:app>runtime', 'repo:core>repo:other-repo>runtime', 'repo:core>repo:other-repo>change', 'repo:other-repo>repo:core>change']) assert.ok(e.has(want), want);
  assert.deepEqual(evidence.find((x) => x.kind === 'clone'), { from: 'repo:core', to: 'repo:other-repo', kind: 'clone', count: 1, via: 'core/lib.py == x.py' });
  assert.equal(evidence.find((x) => x.to === 'repo:app').bareName, 0);
  assert.equal(evidence.find((x) => x.kind === 'import' && x.to === 'repo:other-repo').bareName, 1);
  assert.match(warnings[0], /1 indexed file\(s\) failed to parse/);
  assert.ok(warnings.some((w) => w.includes('repo:core -> repo:other-repo rests only on bare-name resolution')));
  assert.deepEqual(runChecks(model), []);
  assert.equal(sha(path), before, 'the database must be byte-identical after reading');
});

test('exclude_bare_names drops guessed imports but keeps clones', { skip }, () => {
  const { path } = fixtureDb();
  const { model } = atlasModel(path, { excludeBareNames: true });
  assert.ok(!model.edges.some((x) => x[0] === 'repo:core' && x[1] === 'repo:other-repo' && x[2] === 'runtime'));
  assert.ok(model.edges.some((x) => x[0] === 'repo:core' && x[1] === 'repo:other-repo' && x[2] === 'change'));
  assert.equal(atlasModel(path, { includeClones: false, excludeBareNames: true }).model.edges.some((x) => x[1] === 'repo:other-repo'), false);
});

test('file granularity: files of one repo plus the repos they touch', { skip }, () => {
  const { path } = fixtureDb();
  const { model } = atlasModel(path, { granularity: 'file', repo: 'core' });
  const ids = model.components.map((c) => c.id);
  assert.deepEqual(ids.filter((i) => i.startsWith('core:')), ['core:core/broken.py', 'core:core/lib.py', 'core:core/util.py', 'core:db.py']);
  assert.ok(ids.includes('repo:app') && ids.includes('repo:other-repo'));
  const r = propagate(normalizeModel(model), 'core:core/lib.py', 'runtime');
  assert.deepEqual(r.unmitigated, ['core:core/util.py', 'repo:app']);
  assert.equal(model.components.find((c) => c.id === 'core:core/broken.py').parseError, 'SyntaxError');
  assert.deepEqual(runChecks(model), []);
  const prefixed = atlasModel(path, { granularity: 'file', repo: 'core', pathPrefix: 'core/' }).model;
  assert.ok(!prefixed.components.some((c) => c.id === 'core:db.py'));
});

test('file granularity refuses oversized graphs and unknown repos with guidance', { skip }, () => {
  const { path } = fixtureDb();
  assert.throws(() => atlasModel(path, { granularity: 'file', repo: 'core', maxFiles: 2 }), /4 files match; the limit is 2/);
  assert.throws(() => atlasModel(path, { granularity: 'file', repo: 'nope' }), /not in the Atlas; indexed repos: app, core, Other Repo/);
});

test('an incompatible database is refused, naming the missing columns', { skip }, () => {
  const { root } = workspace();
  const path = join(root, 'old.db');
  const db = new sqlite.DatabaseSync(path);
  db.exec('CREATE TABLE repos(id, name); CREATE TABLE files(id); CREATE TABLE symbols(id); CREATE TABLE imports(id)');
  db.close();
  assert.throws(() => atlasModel(path), /table repos lacks origin, head/);
  writeFileSync(join(root, 'junk.db'), 'not sqlite at all, just text that is long enough to not be empty');
  assert.throws(() => atlasModel(join(root, 'junk.db')), /Atlas|open/);
});

test('atlas_import: the configured file is allowed, anything else must be inside the roots', { skip }, () => {
  const { root, path } = fixtureDb();
  const elsewhere = workspace();
  const tool = (ctx, args) => callTool(ctx, 'atlas_import', args);
  const unconfigured = tool({ roots: [elsewhere.root], cache: new Map(), atlasDb: null }, {});
  assert.match(unconfigured.content[0].text, /no Atlas database configured/);
  const outside = tool({ roots: [elsewhere.root], cache: new Map(), atlasDb: null }, { db: path });
  assert.match(outside.content[0].text, /outside the allowed roots/);
  const configured = tool({ roots: [elsewhere.root], cache: new Map(), atlasDb: path }, {});
  assert.equal(configured.isError, undefined);
  const data = JSON.parse(configured.content[0].text);
  assert.equal(data.summary.components, 3);
  assert.ok(data.next.includes('model_inline'));
  const inRoots = tool({ roots: [root], cache: new Map(), atlasDb: null }, { db: 'atlas.db', granularity: 'file', repo: 'core' });
  assert.equal(JSON.parse(inRoots.content[0].text).summary.byLayer.file, 4);
  assert.match(tool({ roots: [root], cache: new Map(), atlasDb: null }, { granularity: 'file' }).content[0].text, /needs repo/);
});

// Contract with the real Atlas: run kgirl's own indexer and blast radius, compare results.
const KGIRL_SRC = process.env.KGIRL_SRC;
const contractSkip = skip || (!KGIRL_SRC ? 'set KGIRL_SRC=/path/to/kgirl/src to run the contract test' : false);

test('contract: a database written by kgirl Atlas reads cleanly and blast radii agree', { skip: contractSkip }, () => {
  const { root, w } = workspace();
  w('lib/pkg/__init__.py', '');
  const body = 'def shared(values, scale):\n    total = 0\n    for v in values:\n        if v > 0:\n            total += v * scale\n    return total\n';
  w('lib/pkg/core.py', `${body}\ndef helper():\n    return 3\n`);
  w('app/main.py', 'from pkg.core import shared\n\ndef run():\n    return shared([1, 2], 3)\n');
  w('app/more.py', 'from main import run\n\ndef again():\n    return run()\n');
  w('copy/dup.py', body);
  const home = join(root, 'home');
  const py = (args) => spawnSync('python3', ['-m', 'kgirl.harness', '--home', home, ...args], { env: { ...process.env, PYTHONPATH: KGIRL_SRC }, encoding: 'utf8' });
  const idx = py(['index', join(root, 'lib'), join(root, 'app'), join(root, 'copy')]);
  assert.equal(idx.status, 0, idx.stderr);
  const db = join(home, 'atlas.db');
  const { model, evidence } = atlasModel(db);
  assert.deepEqual(runChecks(model), []);
  assert.ok(evidence.some((e) => e.kind === 'import' && e.from === 'repo:lib' && e.to === 'repo:app' && e.bareName === 0));
  assert.ok(evidence.some((e) => e.kind === 'clone'));
  const file = atlasModel(db, { granularity: 'file', repo: 'lib', includeClones: false }).model;
  const ours = propagate(normalizeModel(file), 'lib:pkg/core.py', 'runtime').unmitigated;
  const blast = py(['blast', 'lib:pkg/core.py', '--json']);
  assert.equal(blast.status, 0, blast.stderr);
  const theirs = JSON.parse(blast.stdout).impacts.filter((i) => i.via !== 'clone');
  assert.equal(theirs.length, 2); // app/main.py (d1) and app/more.py (d2)
  assert.deepEqual(ours, ['repo:app']); // at file granularity for lib, other repos collapse to one node
});
