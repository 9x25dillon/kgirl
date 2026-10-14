// Findings from inspecting the kgirl harness (main 2e6638b: 26748c4, 80f8b85, #55, #59 and #60), as executable
// claims over examples/kgirl-harness.model.json: as built (proposed controls absent), as designed, and
// records of what the fixes changed: `before` is main before kgirl#55 (MCP hardening) and #59 (forge trust
// boundary); `pre60` is main before #60 (MCP-started trajectories staged).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { asBuilt, parseModel, persistentCycles, persistentExposure, propagate, runChecks } from '../lib/engine.mjs';
import { callTool } from '../server/tools.mjs';

const PLUGIN = resolve(fileURLToPath(import.meta.url), '..', '..');
const m = parseModel(readFileSync(join(PLUGIN, 'examples/kgirl-harness.model.json'), 'utf8'));
const built = asBuilt(m);
const PR55 = ['C-STAGE-MCP', 'C-VERIFY-ALLOWLIST', 'C-APPLY-CONFIRM', 'C-VERSION-NEGOTIATION'];
const PR59 = ['C-FORGE-STAGED', 'C-FORGE-INLINE', 'C-FORGE-FLOOR'];
const PR60 = ['C-STAGE-TRAJECTORY'];
const before = new Set([...built, ...PR55, ...PR59, ...PR60]);
const pre60 = new Set([...built, ...PR60]);
const openLoops = (disabled) => persistentCycles(m, { disabled }).filter((c) => !c.contained).map((c) => c.nodes.join(' -> '));
const exposed = (disabled) => persistentExposure(m, { disabled }).map((x) => x.data).sort();

test('the #55, #59 and #60 controls are implemented; what is left is framing and redaction', () => {
  for (const c of [...PR55, ...PR59, ...PR60]) assert.equal(m.controls[c].status, 'implemented', c);
  assert.deepEqual([...built].sort(), ['C-DATA-FRAMING', 'C-REDACT']);
});

test('before #55 and #59, agent-written content came back through memory, trajectories and exported skills', () => {
  assert.ok(runChecks(m, { disabled: before }).every((p) => p.startsWith('G-TRUST-CYCLE: ')));
  assert.deepEqual(exposed(before), ['skills-dir', 'soup-db', 'trajectories']);
});

test('the soup_remember loop is invisible to elementary-cycle enumeration; the SCC check is what finds it', () => {
  const onlyStage = new Set(['C-STAGE-MCP']);
  assert.deepEqual(openLoops(onlyStage), []);
  assert.deepEqual(exposed(onlyStage), ['soup-db']);
});

test('as designed, every persistent loop is contained', () => {
  assert.deepEqual(runChecks(m), []);
});

test('before #60 the one open loop was a swarm goal persisting as an active trajectory (KFM-12); as built it is closed', () => {
  assert.deepEqual(exposed(pre60), ['skills-dir', 'trajectories']);
  assert.deepEqual(openLoops(pre60), ['kgirl-mcp -> jev-swarm -> trajectories -> skill-export -> skills-dir -> agent']);
  assert.deepEqual(exposed(built), []);
  assert.deepEqual(openLoops(built), []);
  assert.deepEqual(runChecks(m, { disabled: built }), []);
});

test('only the person-run CLI writes SKILL.md, so agent content reaches it damped, never unmitigated', () => {
  for (const d of [before, pre60]) assert.ok(propagate(m, 'agent', 'integrity', { disabled: d }).damped.includes('skills-dir'));
  for (const d of [built, new Set()]) assert.ok(propagate(m, 'agent', 'integrity', { disabled: d }).contained.includes('skills-dir'));
  assert.deepEqual(m.edges.filter((e) => e[1] === 'skills-dir').map((e) => e[0]), ['skill-export']);
  assert.deepEqual(m.edges.filter((e) => e[0] === 'skill-forge' && e[2] === 'trust').map((e) => e[1]).sort(), ['kgirl-mcp', 'soup']);
});

test('soup_remember notes never feed intuition, the forge or the export: only swarm trajectories do', () => {
  for (const n of ['intuition', 'skill-forge', 'skill-export']) {
    assert.deepEqual(m.edges.filter((e) => e[1] === n && e[2] === 'trust').map((e) => e[0]).filter((f) => f !== 'kgirl-mcp'), ['trajectories']);
  }
});

test('before #55 an agent reached code execution through jev_swarm_task verify; as built the allowlist contains it', () => {
  assert.ok(propagate(m, 'agent', 'integrity', { disabled: before }).unmitigated.includes('user-host'));
  assert.ok(propagate(m, 'agent', 'integrity', { disabled: built }).contained.includes('user-host'));
});

test('as built, repo secrets reach the scout model unredacted; redaction only damps it', () => {
  assert.ok(propagate(m, 'local-repos', 'confidentiality', { disabled: built }).unmitigated.includes('anthropic-api'));
  assert.ok(propagate(m, 'local-repos', 'confidentiality').damped.includes('anthropic-api'));
});

test('the porter bridge writes nothing into kgirl: its only outflow is sanitized output to the agent', () => {
  assert.deepEqual(m.edges.filter((e) => e[0] === 'porter-bridge').map((e) => `${e[1]}/${e[2]}`), ['agent/trust']);
  const r = propagate(m, 'porter-bridge', 'integrity');
  assert.deepEqual(r.unmitigated, []);
  assert.ok(r.damped.includes('agent'));
});

test('risk_register as built: soup bypass ranks first by inherent RPN; no persistent exposure is left open', () => {
  const ctx = { roots: [PLUGIN], cache: new Map(), atlasDb: null };
  const r = JSON.parse(callTool(ctx, 'risk_register', { model: 'examples/kgirl-harness.model.json', as_built: true, top: 3 }).content[0].text);
  assert.equal(r.risks[0].id, 'KFM-01');
  assert.deepEqual(r.persistentExposure.map((x) => x.data).sort(), []);
  const designed = JSON.parse(callTool(ctx, 'check_model', { model: 'examples/kgirl-harness.model.json' }).content[0].text);
  assert.equal(designed.ok, true);
  const today = JSON.parse(callTool(ctx, 'check_model', { model: 'examples/kgirl-harness.model.json', as_built: true }).content[0].text);
  assert.equal(today.ok, true);
  assert.deepEqual([...today.disabledControls].sort(), ['C-DATA-FRAMING', 'C-REDACT']);
  assert.ok(!today.disabledControls.includes('C-STAGE-TRAJECTORY'));
});

// Tool-name contract: every MCP tool kgirl serves is mapped to the component it reaches.
const mcp = m.components.find((c) => c.id === 'kgirl-mcp');
const KGIRL_SRC = process.env.KGIRL_SRC;

test('every kgirl MCP tool maps to a component the server is wired to', () => {
  const names = Object.keys(mcp.tools);
  assert.match(mcp.name, new RegExp(`\\(${names.length} tools\\)`));
  const wired = new Set(m.edges.flatMap((e) => (e[0] === 'kgirl-mcp' ? [e[1]] : e[1] === 'kgirl-mcp' ? [e[0]] : [])));
  for (const [tool, target] of Object.entries(mcp.tools)) assert.ok(wired.has(target), `${tool} -> ${target} has no edge to kgirl-mcp`);
});

test('contract: the model lists exactly the tools kgirl serves', { skip: KGIRL_SRC ? false : 'set KGIRL_SRC=/path/to/kgirl/src to run the contract test' }, () => {
  const py = spawnSync('python3', ['-c', 'import json; from kgirl.harness.mcp_server import _tools; print(json.dumps(sorted(_tools(None))))'],
    { env: { ...process.env, PYTHONPATH: KGIRL_SRC }, encoding: 'utf8' });
  assert.equal(py.status, 0, py.stderr);
  assert.deepEqual(Object.keys(mcp.tools).sort(), JSON.parse(py.stdout));
});
