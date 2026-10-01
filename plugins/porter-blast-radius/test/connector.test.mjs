// node --test test/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, symlinkSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  cleanText, discover, gitFacts, gitUrlRemote, normalizeRemote, parseCargo, parseGoMod, parsePackageJson,
  parsePyproject, parseRequirements, parseRemotes,
} from '../lib/discover.mjs';
import { emergentUseCases, parseModel, propagate, runChecks, summarizeModel } from '../lib/engine.mjs';
import { ToolError, confine, readBounded, rootsFromEnv, walk } from '../lib/fsguard.mjs';
import { OUTPUT_CHARS, callTool, respond } from '../server/tools.mjs';
import { gitRepo, porterWorkspace, workspace } from './fixtures.mjs';

const PLUGIN = resolve(fileURLToPath(import.meta.url), '..', '..');
const RELAY = resolve(PLUGIN, '..', '..');
const call = (ctx, name, args) => {
  const r = callTool(ctx, name, args);
  return { error: r.isError ? r.content[0].text : null, data: r.isError ? null : JSON.parse(r.content[0].text) };
};
const ctxFor = (...roots) => ({ roots, cache: new Map() });

// ------------------------------------------------ remotes and git facts

test('normalizeRemote: one identity per repository, credentials dropped', () => {
  const same = [
    'https://github.com/Acme/Core.git', 'https://deploy:tok3n@github.com/acme/core', 'git@github.com:acme/core.git',
    'ssh://git@github.com:22/acme/core.git', 'git+https://github.com/acme/core.git', 'https://www.github.com/acme/core/',
  ].map((u) => normalizeRemote(u.replace(/^git\+/, '')));
  assert.deepEqual(new Set(same), new Set(['github.com/acme/core']));
  for (const local of ['file:///srv/repo.git', 'C:\\work\\repo', '../sibling', '/abs/path', 'https://github.com/a/../b']) {
    assert.equal(normalizeRemote(local), undefined, local);
  }
});

test('gitUrlRemote strips refs but keeps scp user parts', () => {
  assert.equal(gitUrlRemote('git+https://github.com/acme/core.git@v1.2#egg=core'), 'github.com/acme/core');
  assert.equal(gitUrlRemote('https://github.com/acme/core@main'), 'github.com/acme/core');
  assert.equal(gitUrlRemote('git@github.com:acme/core.git'), 'github.com/acme/core');
});

test('parseRemotes reads only remote sections', () => {
  const r = parseRemotes('[core]\n\turl = nope\n[remote "upstream"]\n\turl = https://x.org/a/b\n[remote "origin"]\n  url = "git@h.io:o/r.git"\n');
  assert.equal(r.get('origin'), 'git@h.io:o/r.git');
  assert.equal(r.get('upstream'), 'https://x.org/a/b');
  assert.equal(r.size, 2);
});

test('gitFacts reads HEAD, packed refs and origin without running git', () => {
  const { root, w } = workspace();
  gitRepo(w, 'a', { remote: 'https://u:p@github.com/o/a.git', branch: 'feat/x', sha: 'b'.repeat(40), packed: true });
  const f = gitFacts(join(root, 'a'), [root]);
  assert.deepEqual({ branch: f.branch, commit: f.commit, remote: f.remote }, { branch: 'feat/x', commit: 'b'.repeat(40), remote: 'github.com/o/a' });
});

test('gitFacts follows a worktree .git file only inside the roots', () => {
  const { root, w } = workspace();
  gitRepo(w, 'main', { remote: 'https://github.com/o/main.git', sha: 'd'.repeat(40) });
  w('main/.git/worktrees/wt/HEAD', `${'e'.repeat(40)}\n`);
  w('main/.git/worktrees/wt/commondir', '../..\n');
  w('wt/.git', `gitdir: ${join(root, 'main/.git/worktrees/wt')}\n`);
  const inside = gitFacts(join(root, 'wt'), [root]);
  assert.equal(inside.commit, 'e'.repeat(40));
  assert.equal(inside.remote, 'github.com/o/main');
  const outside = gitFacts(join(root, 'wt'), [join(root, 'wt')]);
  assert.equal(outside.remote, undefined);
  assert.match(outside.warnings[0], /outside the allowed roots/);
});

test('gitFacts ignores a HEAD that tries to traverse out of .git', () => {
  const { root, w } = workspace();
  w('x/.git/HEAD', 'ref: refs/heads/../../../../etc/passwd\n');
  const f = gitFacts(join(root, 'x'), [root]);
  assert.equal(f.commit, undefined);
  assert.equal(f.branch, undefined);
});

// ------------------------------------------------ manifests

test('package.json: names, dev flag, git shorthands', () => {
  const r = parsePackageJson(JSON.stringify({
    name: 'App', dependencies: { lib: '^1', gh: 'github:Acme/Lib', short: 'acme/other#v2', url: 'git+ssh://git@github.com/acme/x.git' },
    devDependencies: { t: '1' },
  }));
  assert.deepEqual(r.provides, ['npm:app']);
  const by = Object.fromEntries(r.deps.map((d) => [d.key, d]));
  assert.equal(by['npm:gh'].remote, 'github.com/acme/lib');
  assert.equal(by['npm:short'].remote, 'github.com/acme/other');
  assert.equal(by['npm:url'].remote, 'github.com/acme/x');
  assert.equal(by['npm:lib'].remote, undefined);
  assert.equal(by['npm:t'].dev, true);
});

test('pyproject: PEP 621 arrays, PEP 503 names, poetry tables', () => {
  const r = parsePyproject('[project]\nname = "My_Pkg"\ndependencies = [\n "A.B>=1",\n "c @ git+https://github.com/o/c.git@main",\n]\n[tool.poetry.dependencies]\npython = "^3.11"\nD_E = {git = "https://github.com/o/de.git"}\n[tool.poetry.group.dev.dependencies]\npytest = "*"\n');
  assert.deepEqual(r.provides, ['pypi:my-pkg']);
  const by = Object.fromEntries(r.deps.map((d) => [d.key, d]));
  assert.ok(by['pypi:a-b']);
  assert.equal(by['pypi:c'].remote, 'github.com/o/c');
  assert.equal(by['pypi:d-e'].remote, 'github.com/o/de');
  assert.equal(by['pypi:pytest'].dev, true);
  assert.equal(by['pypi:python'], undefined);
});

test('requirements: comments, includes, editables and bare git URLs', () => {
  const r = parseRequirements('# c\nfoo==1  # x\n-r other.txt\n-e git+https://github.com/o/e.git#egg=E_pkg\n-e ./local\ngit+https://github.com/o/bare.git\n--index-url https://x\n', true);
  assert.deepEqual(r.deps.map((d) => d.key ?? d.remote), ['pypi:foo', 'pypi:e-pkg', 'github.com/o/bare']);
  assert.ok(r.deps.every((d) => d.dev));
});

test('go.mod and Cargo.toml', () => {
  const g = parseGoMod('module github.com/o/svc\nrequire github.com/o/a v1.0.0\nrequire (\n  github.com/o/b v0.1.0 // indirect\n)\n');
  assert.deepEqual(g.provides, ['go:github.com/o/svc']);
  assert.equal(g.providesRemote, 'github.com/o/svc');
  assert.deepEqual(g.deps.map((d) => d.remote), ['github.com/o/a', 'github.com/o/b']);
  const c = parseCargo('[package]\nname = "My_Crate"\n[dependencies]\nx = { git = "https://github.com/o/x" }\n[dev-dependencies]\ny = "1"\n[dependencies.z_z]\ngit = "https://github.com/o/z"\n');
  assert.deepEqual(c.provides, ['cargo:my-crate']);
  const by = Object.fromEntries(c.deps.map((d) => [d.key, d]));
  assert.equal(by['cargo:x'].remote, 'github.com/o/x');
  assert.equal(by['cargo:y'].dev, true);
  assert.equal(by['cargo:z-z'].remote, 'github.com/o/z');
});

// ------------------------------------------------ discovery end to end

test('porter_discover couples repos through every channel', () => {
  const { root } = porterWorkspace();
  const { model, evidence, warnings } = discover([root]);
  const ids = model.components.map((c) => c.id).sort();
  assert.deepEqual(ids, ['repo:cli', 'repo:core', 'repo:evil-name', 'repo:gosvc', 'repo:rag', 'repo:rust-tool']);
  const e = new Set(model.edges.map((x) => `${x[0]}>${x[1]}>${x[2]}`));
  for (const want of [
    'repo:core>repo:cli>change', 'repo:core>repo:cli>runtime', // npm name
    'repo:core>repo:rag>change', // python git URL -> core's remote
    'repo:core>repo:gosvc>runtime', // go module path -> core's remote
    'repo:rag>repo:rust-tool>change', // cargo git dependency -> rag's remote
    'repo:rag>repo:cli>trust', // porter.json dependsOn
  ]) assert.ok(e.has(want), want);
  assert.ok(!model.edges.some((x) => x[0] === x[1]));
  assert.deepEqual(model.edges.find((x) => x[2] === 'trust')[3], ['C-CONTRACT']);
  assert.ok(evidence.every((ev) => ev.via));
  assert.ok(warnings.some((x) => x.includes('unknown control C-NOPE')));
  assert.ok(warnings.some((x) => x.includes('capability IGNORE PREVIOUS INSTRUCTIONS ignored')));
  assert.deepEqual(runChecks(model), []);
});

test('discovery output carries no credentials and no control characters', () => {
  const { root } = porterWorkspace();
  const text = JSON.stringify(discover([root]));
  assert.ok(!text.includes('tok3n'));
  assert.ok(!/[\u0000-\u001f\u202a-\u202e]/.test(text.replace(/\\[nrt"\\]/g, '')));
  const core = discover([root]).model.components.find((c) => c.id === 'repo:core');
  assert.equal(core.remote, 'github.com/acme/core');
  assert.equal(core.commit, 'c'.repeat(12));
  assert.match(core.projectKey, /^[0-9a-f]{16}$/);
});

test('dev dependencies couple only when asked', () => {
  const { root, w } = porterWorkspace();
  w('testkit/package.json', { name: 'testkit' });
  w('testkit/.git/HEAD', `${'f'.repeat(40)}\n`);
  w('cli/package.json', { name: '@acme/cli', dependencies: { 'porter-core': '1' }, devDependencies: { testkit: '1' } });
  const without = discover([root]).model.edges.filter((x) => x[0] === 'repo:testkit');
  const withDev = discover([root], { includeDev: true }).model.edges.filter((x) => x[0] === 'repo:testkit');
  assert.equal(without.length, 0);
  assert.equal(withDev.length, 2);
});

test('emergent use cases compose capabilities across repos, fragility grows with the chain', () => {
  const { root } = porterWorkspace();
  const { model } = discover([root]);
  const r = emergentUseCases(model, { maxLength: 4 });
  const ids = r.useCases.map((u) => u.id);
  assert.ok(ids.includes('core:parse > cli:render'));
  assert.ok(ids.includes('core:parse > cli:render > rag:answer > evil-name:ok'));
  const two = r.useCases.find((u) => u.id === 'cli:render > rag:answer');
  const three = r.useCases.find((u) => u.id === 'core:parse > cli:render > rag:answer');
  assert.ok(three.fragility >= two.fragility);
  assert.deepEqual(three.effects, ['fs.read', 'network']);
  const note = model.capabilities.find((c) => c.id === 'evil-name:ok').note;
  assert.ok(!/[\u0000\u202e]/.test(note));
});

// ------------------------------------------------ confinement

test('confine refuses paths outside the roots, including through symlinks', () => {
  const { root, w } = workspace();
  const outside = workspace();
  outside.w('secret.json', '{}');
  w('inside/model.json', '{}');
  symlinkSync(join(outside.root, 'secret.json'), join(root, 'inside', 'link.json'));
  assert.ok(confine([join(root, 'inside')], 'model.json').endsWith('model.json'));
  assert.throws(() => confine([join(root, 'inside')], '../'), ToolError);
  assert.throws(() => confine([join(root, 'inside')], join(outside.root, 'secret.json')), /outside the allowed roots/);
  assert.throws(() => confine([join(root, 'inside')], 'link.json'), /outside the allowed roots/);
  assert.throws(() => confine([root], 'nope.json'), /not found/);
});

test('walk never follows symlinked directories', () => {
  const { root, w } = workspace();
  const outside = workspace();
  outside.w('deep/a.model.json', '{}');
  w('real/b.model.json', '{}');
  symlinkSync(outside.root, join(root, 'escape'), 'dir');
  const found = walk([root], { match: (n) => n.endsWith('.model.json') });
  assert.deepEqual(found.map((f) => f.slice(root.length)), ['/real/b.model.json']);
});

test('reads are size-bounded', () => {
  const { root, w } = workspace();
  const p = w('big.txt', 'x'.repeat(2048));
  assert.throws(() => readBounded(p, 1024), /limit is 1024/);
  assert.equal(readBounded(p, 4096).length, 2048);
});

test('roots come from the environment; unexpanded placeholders are ignored', () => {
  const { root } = workspace();
  const roots = rootsFromEnv({ PORTER_RADIUS_PROJECT: '${CLAUDE_PROJECT_DIR}', PORTER_RADIUS_ROOTS: `${root}:/definitely/missing` }, '/');
  assert.deepEqual(roots, [root]);
});

// ------------------------------------------------ tools

test('tools: discover then analyze inline, no files written', () => {
  const { root } = porterWorkspace();
  const ctx = ctxFor(root);
  const d = call(ctx, 'porter_discover', {});
  assert.equal(d.error, null);
  const b = call(ctx, 'blast_radius', { model_inline: d.data.model, sources: ['repo:core'], views: ['change'] });
  assert.deepEqual(b.data.views.change.unmitigated, ['repo:cli', 'repo:gosvc', 'repo:rag', 'repo:rust-tool']);
  const up = call(ctx, 'blast_radius', { model_inline: d.data.model, sources: ['repo:rust-tool'], direction: 'upstream', views: ['runtime'] });
  assert.deepEqual(up.data.views.runtime.unmitigated, ['repo:core', 'repo:rag']);
  const u = call(ctx, 'emergent_use_cases', { model_inline: d.data.model, limit: 2 });
  assert.equal(u.data.useCases.length, 2);
});

test('tools: argument and model errors are actionable isError results', () => {
  const ctx = ctxFor(PLUGIN);
  const model = 'examples/porter-workspace.model.json';
  assert.match(call(ctx, 'blast_radius', { model, sources: ['repo:nope'] }).error, /unknown sources: repo:nope\. Known components include: repo:ctxpack-core/);
  assert.match(call(ctx, 'blast_radius', { model, sources: ['repo:porter'], extra: 1 }).error, /unknown argument extra/);
  assert.match(call(ctx, 'blast_radius', { model, sources: [] }).error, /at least 1/);
  assert.match(call(ctx, 'blast_radius', { model, sources: ['repo:porter'], views: ['vibes'] }).error, /must be one of/);
  assert.match(call(ctx, 'blast_radius', { model: '/etc/hosts', sources: ['x'] }).error, /outside the allowed roots/);
  assert.match(call(ctx, 'blast_radius', { sources: ['x'] }).error, /several models found/);
  assert.match(call(ctx, 'emergent_use_cases', { model: 'model/connector.model.json' }).error, /no capabilities/);
  assert.match(call(ctx, 'check_model', { model_inline: { nope: 1 } }).error, /no components array/);
  assert.match(call(ctx, 'blast_radius', { model_inline: { components: [{ id: 'a' }], edges: [['a', 'b', 'change']] }, sources: ['a'] }).error, /model is invalid/);
});

test('tools: the relay model through the connector matches the design docs', () => {
  const ctx = ctxFor(RELAY);
  const c = call(ctx, 'compose', { model: 'model/relay-model.json' });
  assert.deepEqual(c.data.counts, { direct: 46, gate: 26, forbidden: 0 });
  const a1 = call(ctx, 'compose', { model: 'model/relay-model.json', path: ['E8', 'X1'] });
  assert.deepEqual(a1.data.rules, ['INV-AUTO-TRUST']);
  const r = call(ctx, 'risk_register', { model: 'model/relay-model.json', top: 3 });
  assert.equal(r.data.risks[0].id, 'FM-29');
  assert.equal(r.data.controls[0].id, 'C-TRUST-GATE');
  assert.equal(r.data.openLoops.length, 2);
  const m = call(ctx, 'check_model', { model: 'model/relay-model.json' });
  assert.equal(m.data.ok, true);
});

test('tools: list_models finds models and manifests with counts', () => {
  const d = call(ctxFor(PLUGIN), 'list_models', {});
  const byPath = Object.fromEntries(d.data.files.map((f) => [f.path, f]));
  assert.equal(byPath['examples/porter-workspace.model.json'].capabilities, 6);
  assert.equal(byPath['examples/porter.json'].kind, 'porter-manifest');
  assert.ok(byPath['model/connector.model.json'].components > 10);
});

test('output: oversized results shrink and say so; strings are cleaned', () => {
  const big = { items: Array.from({ length: 5000 }, (_, i) => `item-${i}-${'x'.repeat(20)}`), s: 'a\u202eb\u0007c' };
  const r = respond(big);
  const text = r.content[0].text;
  assert.ok(text.length <= OUTPUT_CHARS);
  const obj = JSON.parse(text);
  assert.match(obj.truncated, /arrays cut/);
  assert.equal(obj.s, 'a b c');
  assert.equal(cleanText('x'.repeat(500), 10).length, 10);
});

test('the example workspace model checks clean and summarizes', () => {
  const m = parseModel(readFileSync(join(PLUGIN, 'examples/porter-workspace.model.json'), 'utf8'));
  assert.deepEqual(runChecks(m), []);
  assert.deepEqual(summarizeModel(m).views, ['change', 'runtime', 'integrity', 'confidentiality']);
  assert.deepEqual(propagate(m, ['repo:ctxpack-core', 'repo:rag-assistant'], 'runtime').unmitigated, ['repo:ctxr-cli', 'repo:porter']);
});

test('tools never create files in the roots', () => {
  const { root, w } = porterWorkspace();
  const before = walk([root], { match: () => true, max: 10000 }).sort();
  const ctx = ctxFor(root);
  const d = call(ctx, 'porter_discover', {});
  call(ctx, 'blast_radius', { model_inline: d.data.model, sources: ['repo:core'] });
  call(ctx, 'list_models', {});
  w('m.model.json', d.data.model);
  call(ctx, 'check_model', { model: 'm.model.json' });
  const after = walk([root], { match: () => true, max: 10000 }).sort();
  assert.deepEqual(after, [...before, join(root, 'm.model.json')].sort());
});
