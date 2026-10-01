// Merging models from different sources into one graph.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { atlasModel } from '../lib/atlas.mjs';
import { discover } from '../lib/discover.mjs';
import { mergeModels, propagate, runChecks } from '../lib/engine.mjs';
import { callTool } from '../server/tools.mjs';
import { porterWorkspace, workspace } from './fixtures.mjs';

const HERE = resolve(fileURLToPath(import.meta.url), '..');
const sqlite = process.getBuiltinModule?.('node:sqlite');

// An Atlas that indexed the same repos under directory names: Core-Checkout and rag-dir.
function atlasOfSameRepos() {
  const { root } = workspace();
  const path = join(root, 'atlas.db');
  const db = new sqlite.DatabaseSync(path);
  db.exec(readFileSync(join(HERE, 'atlas-schema.sql'), 'utf8'));
  db.exec(`
    INSERT INTO repos(id, name, root, origin, head) VALUES
      (1, 'Core-Checkout', '/w/core', 'https://github.com/acme/core.git', ''),
      (2, 'rag-dir', '/w/rag', 'git@github.com:acme/rag.git', '');
    INSERT INTO files(id, repo_id, path, lang) VALUES (1, 1, 'porter_core/api.py', 'python'), (2, 2, 'rag/query.py', 'python');
    INSERT INTO imports(id, file_id, target, names, level, line, kind, resolved_file_id) VALUES (1, 2, 'porter_core.api', 'q', 0, 1, 'import', 1);
    INSERT INTO symbols(id, file_id, kind, name, qualname, body_hash) VALUES (1, 1, 'function', 'q', 'q', 'hh'), (2, 2, 'function', 'q2', 'q2', 'hh');
  `);
  db.close();
  return atlasModel(path).model;
}

test('same remote, different names: Atlas repos fold into the discovered ids', { skip: !sqlite && 'needs node:sqlite' }, () => {
  const { root } = porterWorkspace();
  const found = discover([root]).model;
  const atlas = atlasOfSameRepos();
  const { model, renamed, conflicts } = mergeModels([found, atlas]);
  assert.deepEqual(renamed.map((r) => `${r.from}->${r.to}`).sort(), ['repo:core-checkout->repo:core', 'repo:rag-dir->repo:rag']);
  assert.ok(!model.components.some((c) => c.id === 'repo:core-checkout' || c.id === 'repo:rag-dir'));
  assert.equal(model.edges.filter((e) => e[0] === 'repo:core' && e[1] === 'repo:rag' && e[2] === 'runtime').length, 1, 'duplicate edges collapse');
  assert.ok(model.edges.some((e) => e[0] === 'repo:rag' && e[1] === 'repo:core' && e[2] === 'change'), 'the clone edge arrives under the merged ids');
  assert.equal(model.capabilities.length, found.capabilities.length);
  assert.ok(Array.isArray(conflicts));
  assert.deepEqual(runChecks(model), []);
  assert.ok(propagate(model, 'repo:rag', 'change').unmitigated.includes('repo:core'));
});

test('edge controls union; conflicting definitions keep the first and are reported', () => {
  const a = { components: [{ id: 'x' }, { id: 'y', layer: 'repo' }], controls: { C: { strength: 'full' } }, edges: [['x', 'y', 'change', 'C']], failureModes: [{ id: 'F1', at: 'x', kind: 'change', title: 't', inherent: [1, 1, 1], residual: [1, 1, 1], controls: [] }] };
  const b = { components: [{ id: 'y', layer: 'service' }, { id: 'z' }], controls: { C: { strength: 'partial' }, D: { strength: 'partial' } }, edges: [['x', 'y', 'change', 'D'], ['y', 'z', 'runtime']], failureModes: [{ id: 'F1', at: 'y', kind: 'change', title: 'other', inherent: [2, 2, 2], residual: [2, 2, 2], controls: [] }] };
  const { model, conflicts } = mergeModels([a, b]);
  assert.deepEqual(model.edges.find((e) => e[0] === 'x' && e[1] === 'y')[3], ['C', 'D']);
  assert.equal(model.controls.C.strength, 'full');
  assert.equal(model.components.find((c) => c.id === 'y').layer, 'repo');
  assert.equal(model.failureModes.length, 1);
  assert.ok(conflicts.some((c) => c.startsWith('control C')));
  assert.ok(conflicts.some((c) => c.startsWith('component y.layer')));
  assert.ok(conflicts.some((c) => c.startsWith('failure mode F1')));
  assert.deepEqual(runChecks(model), []);
});

test('merge_models tool: inline models in, merged model out; needs two', () => {
  const { root } = porterWorkspace();
  const ctx = { roots: [root], cache: new Map(), atlasDb: null };
  const found = discover([root]).model;
  const extra = { components: [{ id: 'svc:billing', layer: 'service' }, { id: 'repo:core' }], edges: [['repo:core', 'svc:billing', 'runtime']] };
  const r = callTool(ctx, 'merge_models', { models_inline: [found, extra] });
  assert.equal(r.isError, undefined, r.content[0].text);
  const data = JSON.parse(r.content[0].text);
  assert.ok(data.model.components.some((c) => c.id === 'svc:billing'));
  assert.deepEqual(data.problems, []);
  const one = callTool(ctx, 'merge_models', { models_inline: [found] });
  assert.match(one.content[0].text, /at least two models/);
});
