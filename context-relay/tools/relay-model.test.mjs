// node --test tools/relay-model.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  loadModel, validateModel, globalInvariants, runChecks, checkWorkflow, routeMatrix,
  propagate, persistentCycles, controlValue, rankRisks, staleDocs,
} from './relay-model.mjs';

const model = loadModel();
const clone = () => structuredClone(model);
const edge = (m, from, to, kind) => m.edges.find((e) => e[0] === from && e[1] === to && e[2] === kind);

test('model is well formed and passes every check', () => {
  assert.deepEqual(validateModel(model), []);
  assert.deepEqual(globalInvariants(model), []);
  assert.deepEqual(runChecks(model), []);
});

test('generated doc sections are in sync with the model', () => {
  assert.deepEqual(staleDocs(model), []);
});

for (const w of model.workflows) {
  test(`workflow ${w.id} passes: ${w.name}`, () => {
    const r = checkWorkflow(model, w);
    assert.equal(r.ok, true, JSON.stringify(r.violations));
  });
}

for (const a of model.antiWorkflows) {
  test(`anti-workflow ${a.id} fires exactly ${a.expect.join(', ')}`, () => {
    assert.deepEqual(checkWorkflow(model, a).rules, [...a.expect].sort());
  });
}

test('route matrix: every composition is realizable, trust decides which need a gate', () => {
  const cells = routeMatrix(model);
  const at = (e, x) => cells.find((c) => c.entry === e && c.exit === x).verdict;
  assert.equal(cells.length, model.entries.length * model.exits.length);
  assert.equal(cells.filter((c) => c.verdict === 'forbidden').length, 0);
  assert.equal(at('E1', 'X1'), 'direct');   // own session -> next session
  assert.equal(at('E1', 'X5'), 'gate');     // even own session needs review before AGENTS.md
  assert.equal(at('E6', 'X1'), 'gate');     // repo pack never auto-injects
  assert.equal(at('E8', 'X3'), 'gate');     // model-authored drafts are invisible to MCP
  assert.equal(at('E3', 'X6'), 'direct');   // foreign pack can still be handed on as a file
});

test('a failed approval confers nothing', () => {
  const r = checkWorkflow(model, { path: ['E3', { op: 'approve', at: 'cli', by: 'model' }, 'X1'] });
  assert.deepEqual(r.rules, ['INV-ACTOR', 'INV-AUTO-TRUST']);
});

test('approval is bound to the content hash: editing after approval revokes it', () => {
  const r = checkWorkflow(model, { path: ['E3', 'gate', { op: 'edit', at: 'cli', by: 'human' }, 'X5'] });
  assert.deepEqual(r.rules, ['INV-INSTR-TRUST']);
});

// ------------------------------------------------ mutation tests: the checker must catch breakage

test('mutation: removing the trust gate opens an unaccepted persistent loop', () => {
  const m = clone();
  edge(m, 'store-drafts', 'store-data', 'trust')[3] = null;
  assert.ok(runChecks(m).some((p) => p.startsWith('G-TRUST-CYCLE: uncontained')));
});

test('mutation: un-accepting the continuity loop fails the check, as cycles and as shared exposure', () => {
  const m = clone();
  m.acceptedCycles = [];
  const problems = runChecks(m).filter((p) => p.startsWith('G-TRUST-CYCLE'));
  assert.equal(problems.filter((p) => p.includes('uncontained persistent loop')).length, 2);
  assert.deepEqual(problems.filter((p) => p.includes('share an uncontained loop')), [
    'G-TRUST-CYCLE: cc-session and store-data share an uncontained loop (capture, cc-session, hooks, mcp, store-data)',
  ]);
});

test('mutation: a brief budget that leaves no room for the wrapper breaks W1', () => {
  const m = clone();
  m.formats.brief.maxChars = 9500;
  assert.deepEqual(checkWorkflow(m, m.workflows[0]).rules, ['INV-BUDGET']);
});

test('mutation: any network effect violates INV-NO-NET', () => {
  const m = clone();
  m.primitives.render.effects = ['network'];
  assert.ok(globalInvariants(m).some((p) => p.startsWith('INV-NO-NET')));
});

test('mutation: exec reachable by a model violates INV-EXEC-HUMAN', () => {
  const m = clone();
  m.primitives.via.actors = ['human', 'model'];
  assert.ok(globalInvariants(m).some((p) => p.startsWith('INV-EXEC-HUMAN')));
});

test('mutation: a hook surface without fail-open violates INV-FAILOPEN', () => {
  const m = clone();
  delete m.components.find((c) => c.id === 'hooks').failPolicy;
  assert.ok(globalInvariants(m).some((p) => p.startsWith('INV-FAILOPEN')));
});

test('mutation: dangling references are reported', () => {
  const m = clone();
  m.edges.push(['spec', 'nowhere', 'change', 'C-NOPE']);
  const errs = validateModel(m);
  assert.ok(errs.some((e) => e.includes('unknown target')));
  assert.ok(errs.some((e) => e.includes('unknown control C-NOPE')));
});

test('mutation: a workflow with no starting value is a validation error, not a crash', () => {
  const m = clone();
  m.workflows.push({ id: 'W-bad', name: 'starts at an exit', path: ['X1'] });
  m.exits[0].steps[2] = { op: 'inject', at: 'hooks', by: 'hook' };
  const errs = validateModel(m);
  assert.ok(errs.some((e) => e.startsWith('W-bad: must begin')));
  assert.ok(errs.some((e) => e.includes('inject needs a sink')));
  assert.ok(runChecks(m).length > 0);
});

// ------------------------------------------------ propagation semantics on a toy graph

const toy = {
  components: [{ id: 'a' }, { id: 'b' }, { id: 'c' }, { id: 'd' }, { id: 's', secretOrigin: true }],
  controls: {
    F: { strength: 'full' },
    P: { strength: 'partial' },
    R: { strength: 'partial', threats: ['confidentiality'] },
  },
  edges: [
    ['a', 'b', 'runtime', 'F'],
    ['a', 'c', 'runtime', 'P'],
    ['c', 'd', 'runtime'],
    ['b', 'd', 'runtime'],
    ['a', 'd', 'runtime', null, 'baseline'],
    ['s', 'a', 'trust', 'R'],
  ],
};

test('propagation: full contains, partial damps, the worst path wins', () => {
  const r = propagate(toy, 'a', 'runtime');
  assert.deepEqual(r, { unmitigated: [], damped: ['c', 'd'], contained: ['b'] });
});

test('propagation: baseline edges count only when asked', () => {
  assert.deepEqual(propagate(toy, 'a', 'runtime', { baseline: true }).unmitigated, ['d']);
});

test('propagation: controls apply only to their declared threats', () => {
  assert.deepEqual(propagate(toy, 's', 'integrity').unmitigated, ['a']);
  assert.deepEqual(propagate(toy, 's', 'confidentiality').damped, ['a']);
});

// ------------------------------------------------ findings the docs rely on

test('finding: redaction is the only barrier toward git and other agents; claude.ai also has host scoping', () => {
  const egress = model.components.filter((c) => c.egress).map((c) => c.id);
  const on = propagate(model, 'cc-session', 'confidentiality');
  const off = propagate(model, 'cc-session', 'confidentiality', { disabled: new Set(['C-REDACT']) });
  for (const id of egress) assert.ok(on.damped.includes(id), `${id} should be damped`);
  for (const id of ['git-remote', 'other-agents']) assert.ok(off.unmitigated.includes(id), `${id} should be unmitigated without C-REDACT`);
  assert.ok(off.damped.includes('cai-chat'), 'C-NMH-SCOPE still damps the claude.ai path');
});

test('finding: the trust gate is the highest-value control', () => {
  assert.equal(controlValue(model)[0].id, 'C-TRUST-GATE');
});

test('finding: exactly the two continuity loops remain open, both accepted', () => {
  const open = persistentCycles(model).filter((c) => !c.contained);
  assert.equal(open.length, 2);
  assert.ok(open.every((c) => c.accepted === 'D-008'));
  assert.ok(open.every((c) => c.nodes.includes('capture') && c.nodes.includes('cc-session')));
});

test('risk ranking is sorted by residual RPN', () => {
  const r = rankRisks(model);
  for (let i = 1; i < r.length; i++) assert.ok(r[i - 1].residualRpn >= r[i].residualRpn);
});
