// The connector's own blast radius, as executable claims about its design.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseModel, persistentCycles, propagate, rankRisks, runChecks } from '../lib/engine.mjs';

const PLUGIN = resolve(fileURLToPath(import.meta.url), '..', '..');
const m = parseModel(readFileSync(join(PLUGIN, 'model/connector.model.json'), 'utf8'));

test('the self-model checks clean', () => {
  assert.deepEqual(runChecks(m), []);
});

test('a hostile repository reaches the agent only damped, and never executes code', () => {
  const r = propagate(m, 'repo-files', 'integrity');
  assert.ok(r.damped.includes('client'));
  assert.ok(r.contained.includes('user-host'));
  assert.equal(r.unmitigated.length, 0);
});

test('agent-supplied arguments and paths are fully contained', () => {
  const r = propagate(m, 'client', 'integrity');
  assert.deepEqual(r.unmitigated, []);
  assert.deepEqual(r.damped, []);
  assert.ok(r.contained.includes('fsguard') && r.contained.includes('user-host'));
});

test('files outside the roots cannot reach the agent; without C-ROOTS they would', () => {
  assert.ok(propagate(m, 'host-files', 'confidentiality').contained.includes('client'));
  assert.ok(propagate(m, 'host-files', 'confidentiality', { disabled: new Set(['C-ROOTS']) }).unmitigated.includes('client'));
});

test('the connector adds no persistence: no data components, no trust loops', () => {
  assert.equal(m.components.filter((c) => c.layer === 'data').length, 0);
  assert.deepEqual(persistentCycles(m), []);
});

test('every implemented control names a verification that exists', () => {
  for (const [id, c] of Object.entries(m.controls)) {
    for (const f of c.verify.match(/test\/[\w.-]+\.mjs/g) ?? []) assert.ok(existsSync(join(PLUGIN, f)), `${id} cites missing ${f}`);
  }
});

test('ranking: misreading an incomplete model outranks every security risk; injection leads the trust risks', () => {
  const r = rankRisks(m);
  assert.equal(r[0].id, 'CFM-05');
  assert.equal(r.find((f) => f.kind === 'trust').id, 'CFM-01');
});
