// Findings from inspecting the kgirl harness (commit 26748c4), as executable claims over
// examples/kgirl-harness.model.json: as built (proposed controls absent) versus as designed.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { asBuilt, parseModel, persistentCycles, persistentExposure, propagate, runChecks } from '../lib/engine.mjs';
import { callTool } from '../server/tools.mjs';

const PLUGIN = resolve(fileURLToPath(import.meta.url), '..', '..');
const m = parseModel(readFileSync(join(PLUGIN, 'examples/kgirl-harness.model.json'), 'utf8'));
const built = asBuilt(m);

test('as built, model-written memory comes back in later sessions (soup loop)', () => {
  assert.deepEqual(runChecks(m, { disabled: built }), [
    'G-TRUST-CYCLE: agent and soup-db share an uncontained loop (agent, assistant, kgirl-mcp, soup, soup-db)',
  ]);
});

test('the soup loop is invisible to elementary-cycle enumeration; the SCC check is what finds it', () => {
  assert.equal(persistentCycles(m, { disabled: built }).filter((c) => !c.contained).length, 0);
  assert.equal(persistentExposure(m, { disabled: built }).length, 1);
});

test('staging MCP writes through the curator closes the loop', () => {
  assert.deepEqual(runChecks(m), []);
  const onlyStage = new Set([...built].filter((c) => c !== 'C-STAGE-MCP'));
  assert.deepEqual(persistentExposure(m, { disabled: onlyStage }), []);
});

test('as built, an agent reaches code execution through jev_swarm_task verify; the allowlist contains it', () => {
  assert.ok(propagate(m, 'agent', 'integrity', { disabled: built }).unmitigated.includes('user-host'));
  assert.ok(propagate(m, 'agent', 'integrity').contained.includes('user-host'));
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

test('risk_register as built ranks the soup bypass first', () => {
  const ctx = { roots: [PLUGIN], cache: new Map(), atlasDb: null };
  const r = JSON.parse(callTool(ctx, 'risk_register', { model: 'examples/kgirl-harness.model.json', as_built: true, top: 3 }).content[0].text);
  assert.equal(r.risks[0].id, 'KFM-01');
  assert.equal(r.persistentExposure[0].data, 'soup-db');
  const designed = JSON.parse(callTool(ctx, 'check_model', { model: 'examples/kgirl-harness.model.json' }).content[0].text);
  assert.equal(designed.ok, true);
  const today = JSON.parse(callTool(ctx, 'check_model', { model: 'examples/kgirl-harness.model.json', as_built: true }).content[0].text);
  assert.equal(today.ok, false);
  assert.ok(today.disabledControls.includes('C-STAGE-MCP'));
});
