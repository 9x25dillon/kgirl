// Blast-radius engine: typed workflow checking and lattice propagation over a coupling model.
// Pure functions, no I/O. Shared by the porter-blast-radius MCP connector and the Context Relay
// design-doc tool (context-relay/tools/relay-model.mjs), so the two can never drift.

// Fills every optional section so a sparse model (for example a Porter coupling model
// with only components and edges) goes through the same code paths as the full relay model.
export function normalizeModel(m) {
  const defaults = {
    components: [], edges: [], controls: {}, primitives: {}, sinks: {}, formats: {}, actors: [],
    entries: [], exits: [], workflows: [], antiWorkflows: [], acceptedCycles: [], failureModes: [],
    capabilities: [], trust: { autoInject: [], instruction: [] }, wrapOverheadChars: 0,
  };
  const out = { ...m };
  for (const [k, v] of Object.entries(defaults)) if (out[k] === undefined || out[k] === null) out[k] = structuredClone(v);
  return out;
}

export const parseModel = (text) => normalizeModel(JSON.parse(text));

// ---------------------------------------------------------------- structure

const KINDS = ['change', 'runtime', 'trust'];
const THREATS = ['integrity', 'confidentiality'];

// A view is one propagation question over one edge kind. Trust edges are data flow;
// integrity asks where attacker content goes, confidentiality where secrets go.
export const VIEWS = {
  change: { kind: 'change', threat: null },
  runtime: { kind: 'runtime', threat: null },
  integrity: { kind: 'trust', threat: 'integrity' },
  confidentiality: { kind: 'trust', threat: 'confidentiality', sources: (m) => m.components.filter((c) => c.secretOrigin).map((c) => c.id) },
};
const TYPES = new Set(['Source', 'External', 'Pack', 'Text', 'Wrapped', 'Sink']);

export function edges(m) {
  return m.edges.map(([from, to, kind, controls = null, flag = null]) => ({
    from, to, kind, flag, baseline: flag === 'baseline',
    controls: controls === null ? [] : [].concat(controls),
  }));
}

export function validateModel(raw) {
  const m = normalizeModel(raw);
  const errs = [];
  const ids = new Set();
  for (const c of m.components) {
    if (ids.has(c.id)) errs.push(`duplicate component ${c.id}`);
    ids.add(c.id);
  }
  for (const e of edges(m)) {
    const tag = `edge ${e.from}->${e.to} (${e.kind})`;
    if (!ids.has(e.from)) errs.push(`${tag}: unknown source`);
    if (!ids.has(e.to)) errs.push(`${tag}: unknown target`);
    if (!KINDS.includes(e.kind)) errs.push(`${tag}: unknown kind`);
    for (const c of e.controls) if (!m.controls[c]) errs.push(`${tag}: unknown control ${c}`);
    if (e.flag !== null && e.flag !== 'baseline') errs.push(`${tag}: unknown flag ${e.flag}`);
  }
  for (const [id, c] of Object.entries(m.controls)) {
    if (c.strength !== 'full' && c.strength !== 'partial') errs.push(`control ${id}: strength must be full|partial`);
    for (const t of c.threats ?? []) if (!THREATS.includes(t)) errs.push(`control ${id}: unknown threat ${t}`);
  }
  for (const [op, p] of Object.entries(m.primitives)) {
    if (!TYPES.has(p.in) || !TYPES.has(p.out)) errs.push(`primitive ${op}: unknown type`);
    if (p.surfaces !== '*') for (const s of p.surfaces) if (!ids.has(s)) errs.push(`primitive ${op}: unknown surface ${s}`);
  }
  const steps = [];
  for (const r of [...m.entries, ...m.exits]) steps.push(...r.steps.map((s) => [r.id, s]));
  for (const w of [...m.workflows, ...m.antiWorkflows]) {
    const first = w.path[0];
    if (!w.start && !(typeof first === 'string' && routeById(m).get(first)?.start)) errs.push(`${w.id}: must begin with an entry route or declare start`);
    for (const item of w.path) {
      if (typeof item === 'object') steps.push([w.id, item]);
      else if (item !== 'gate' && !routeById(m).has(item)) errs.push(`${w.id}: unknown route ${item}`);
    }
  }
  if (m.gate) steps.push(['gate', m.gate]);
  else if ([...m.workflows, ...m.antiWorkflows].some((w) => w.path.includes('gate'))) errs.push('a workflow uses "gate" but the model defines no gate step');
  for (const [owner, s] of steps) {
    if (!m.primitives[s.op]) errs.push(`${owner}: unknown op ${s.op}`);
    if (!ids.has(s.at)) errs.push(`${owner}: unknown surface ${s.at}`);
    if (!m.actors.includes(s.by)) errs.push(`${owner}: unknown actor ${s.by}`);
    if (s.sink && !m.sinks[s.sink]) errs.push(`${owner}: unknown sink ${s.sink}`);
    if (s.fmt && !(s.fmt in m.formats)) errs.push(`${owner}: unknown format ${s.fmt}`);
    if ((s.op === 'inject' || s.op === 'emit') && !s.sink) errs.push(`${owner}: ${s.op} needs a sink`);
    if (s.op === 'render' && !s.fmt) errs.push(`${owner}: render needs a fmt`);
  }
  for (const f of m.failureModes) {
    if (!ids.has(f.at)) errs.push(`${f.id}: unknown component ${f.at}`);
    for (const c of f.controls) if (!m.controls[c]) errs.push(`${f.id}: unknown control ${c}`);
    for (const t of [f.inherent, f.residual]) {
      if (t.length !== 3 || t.some((x) => !Number.isInteger(x) || x < 1 || x > 5)) errs.push(`${f.id}: S/L/D must be integers 1..5`);
    }
  }
  for (const c of m.acceptedCycles) for (const n of c.nodes) if (!ids.has(n)) errs.push(`accepted cycle: unknown node ${n}`);
  const capIds = new Set();
  for (const c of m.capabilities) {
    if (capIds.has(c.id)) errs.push(`duplicate capability ${c.id}`);
    capIds.add(c.id);
    if (!ids.has(c.repo)) errs.push(`capability ${c.id}: unknown component ${c.repo}`);
    if (typeof c.in !== 'string' || typeof c.out !== 'string') errs.push(`capability ${c.id}: in and out must be type names`);
  }
  return errs;
}

// Invariants that hold over the primitive set and components, independent of any workflow.
export function globalInvariants(raw) {
  const m = normalizeModel(raw);
  const errs = [];
  for (const [op, p] of Object.entries(m.primitives)) {
    if (p.effects.includes('network')) errs.push(`INV-NO-NET: ${op} declares a network effect`);
    if (p.effects.includes('exec')) {
      const humanOnly = Array.isArray(p.actors) && p.actors.length === 1 && p.actors[0] === 'human';
      const cliOnly = Array.isArray(p.surfaces) && p.surfaces.length === 1 && p.surfaces[0] === 'cli';
      if (!humanOnly || !cliOnly) errs.push(`INV-EXEC-HUMAN: ${op} executes commands but is not human-only on the CLI`);
    }
  }
  const byId = new Map(m.components.map((c) => [c.id, c]));
  const hookSurfaces = new Set();
  const collect = (s) => { if (s.by === 'hook') hookSurfaces.add(s.at); };
  for (const r of [...m.entries, ...m.exits]) r.steps.forEach(collect);
  for (const at of hookSurfaces) {
    if (byId.get(at)?.failPolicy !== 'open') errs.push(`INV-FAILOPEN: ${at} runs in a host hook without failPolicy "open"`);
  }
  return errs;
}

// ---------------------------------------------------------------- workflows

const routeCache = new WeakMap();
function routeById(m) {
  if (!routeCache.has(m)) routeCache.set(m, new Map([...m.entries, ...m.exits].map((r) => [r.id, r])));
  return routeCache.get(m);
}

// A workflow path is a list of route ids, the literal 'gate', and inline steps.
// An entry route (one with `start`) opens a new segment: the pack crossed a boundary.
export function expand(m, wf) {
  const segments = [];
  let cur = null;
  const openSegment = (start) => { cur = { start, steps: [] }; segments.push(cur); };
  if (wf.start) openSegment(wf.start);
  for (const item of wf.path) {
    if (item === 'gate') { cur.steps.push(m.gate); continue; }
    if (typeof item === 'object') { cur.steps.push(item); continue; }
    const r = routeById(m).get(item);
    if (r.start) openSegment(r.start);
    cur.steps.push(...r.steps);
  }
  return segments;
}

export const effectiveTrust = (v) => (v.labels.has('approved') ? 'user' : v.origin);

// Abstract interpretation: the value carries a type, a label set and an origin.
// A step that violates any rule confers no labels, so a failed approval grants nothing.
export function checkWorkflow(m, wf) {
  const violations = [];
  expand(m, wf).forEach((seg, si) => {
    let v = { t: seg.start.t, origin: seg.start.trust ?? null, labels: new Set(), fmt: null };
    seg.steps.forEach((step, i) => {
      const where = `${si + 1}.${i + 1}:${step.op}@${step.at}/${step.by}`;
      const p = m.primitives[step.op];
      let ok = true;
      const fail = (rule, msg) => { ok = false; violations.push({ rule, where, msg }); };

      if (step.op === 'inject') {
        if (v.t !== 'Wrapped') fail('INV-WRAP', `inject needs Wrapped, got ${v.t}`);
      } else if (step.op === 'emit') {
        if (v.t !== 'Text' && v.t !== 'Wrapped') fail('INV-TYPE', `emit needs Text, got ${v.t}`);
      } else if (v.t !== p.in) {
        fail('INV-TYPE', `${step.op} needs ${p.in}, got ${v.t}`);
      }
      if (p.surfaces !== '*' && !p.surfaces.includes(step.at)) fail('INV-SURFACE', `${step.op} is not offered on ${step.at}`);
      if (p.actors !== '*' && !p.actors.includes(step.by)) fail('INV-ACTOR', `${step.op} cannot be performed by ${step.by}`);
      for (const l of p.requires ?? []) if (!v.labels.has(l)) fail('INV-REQ', `${step.op} requires ${l}`);
      if (step.op === 'capture' && wf.cyclic && !step.stripInjected) fail('INV-LOOP-ECHO', 'cyclic capture must strip previously injected regions');

      const trust = effectiveTrust(v);
      const serve = m.trust.serve?.[step.at];
      if (step.op === 'render' && serve && !serve.includes(trust)) fail('INV-SERVE', `${step.at} serves only ${serve.join('|')} packs, got ${trust}`);

      if (step.op === 'inject' || step.op === 'emit') {
        const sink = m.sinks[step.sink];
        if (step.op === 'inject' && sink.class !== 'context') fail('INV-SINK-CLASS', `inject into ${sink.class} sink ${step.sink}`);
        if (step.op === 'emit' && sink.class === 'context') fail('INV-SINK-CLASS', `context sink ${step.sink} must be reached through wrap+inject`);
        if (step.op === 'inject' && step.by === 'hook') {
          if (!m.trust.autoInject.includes(trust)) fail('INV-AUTO-TRUST', `auto-inject of a ${trust} pack`);
          if (!v.labels.has('sealed')) fail('INV-AUTO-SEALED', 'auto-inject of an unsealed pack');
        }
        if (sink.class === 'instruction' && !m.trust.instruction.includes(trust)) fail('INV-INSTR-TRUST', `${trust} pack into instruction sink ${step.sink}`);
        if (sink.requiresSealed && !v.labels.has('sealed')) fail('INV-SHARE-SEALED', `${step.sink} requires a sealed pack`);
        if (sink.capacityChars != null) {
          const max = m.formats[v.fmt]?.maxChars ?? null;
          const need = max === null ? Infinity : max + (v.t === 'Wrapped' ? m.wrapOverheadChars : 0);
          if (need > sink.capacityChars) fail('INV-BUDGET', `${v.fmt} render (${need} chars) exceeds ${step.sink} (${sink.capacityChars})`);
        }
      }

      const next = { t: p.out, origin: v.origin, labels: new Set(v.labels), fmt: v.fmt };
      if (p.origin) { next.origin = p.origin; next.labels = new Set(); }
      if (step.op === 'render') next.fmt = step.fmt;
      if (ok) for (const l of p.adds ?? []) next.labels.add(l);
      for (const l of p.removes ?? []) next.labels.delete(l);
      v = next;
    });
  });
  return { ok: violations.length === 0, violations, rules: [...new Set(violations.map((x) => x.rule))].sort() };
}

const GATEABLE = new Set(['INV-AUTO-TRUST', 'INV-INSTR-TRUST', 'INV-SERVE']);

// Every entry composed with every exit. A cell is 'direct' when it checks as is,
// 'gate' when only trust rules fail and inserting a human approval fixes it.
export function routeMatrix(m) {
  const cells = [];
  for (const e of m.entries) {
    for (const x of m.exits) {
      const direct = checkWorkflow(m, { path: [e.id, x.id] });
      let verdict = 'direct';
      let rules = [];
      if (!direct.ok) {
        rules = direct.rules;
        const gated = direct.rules.every((r) => GATEABLE.has(r)) && checkWorkflow(m, { path: [e.id, 'gate', x.id] }).ok;
        verdict = gated ? 'gate' : 'forbidden';
      }
      cells.push({ entry: e.id, exit: x.id, verdict, rules });
    }
  }
  return cells;
}

// ---------------------------------------------------------------- blast radius

// Exposure levels: 3 unmitigated, 2 damped (only partial controls on the best path), 1 contained.
export const LEVEL = { unmitigated: 3, damped: 2, contained: 1 };

// reverse=true walks edges backwards: from a node to everything that can reach it.
function adjacency(m, kind, baseline, reverse = false) {
  const adj = new Map();
  for (const e of edges(m)) {
    if (e.kind !== kind || (e.baseline && !baseline)) continue;
    const [a, b] = reverse ? [e.to, e.from] : [e.from, e.to];
    if (!adj.has(a)) adj.set(a, []);
    adj.get(a).push({ ...e, to: b });
  }
  return adj;
}

// Controls without `threats` apply to every view; the others only to their threats.
const applies = (control, threat) => !control.threats || control.threats.includes(threat);

function edgeStrength(m, e, threat, disabled) {
  const live = e.controls.filter((c) => !disabled.has(c) && applies(m.controls[c], threat)).map((c) => m.controls[c].strength);
  return live.includes('full') ? 'full' : live.includes('partial') ? 'partial' : null;
}

// The level of a node is the best path's minimum edge cap (none 3, partial 2, full 1),
// which is direction-independent, so reverse propagation answers "what can reach me".
// `source` may be one id or several (a change set); sources are excluded from the result.
export function propagate(m, source, view, { disabled = new Set(), baseline = false, reverse = false } = {}) {
  const { kind, threat } = VIEWS[view];
  const adj = adjacency(m, kind, baseline, reverse);
  const sources = [].concat(source);
  const level = new Map(sources.map((s) => [s, LEVEL.unmitigated]));
  const queue = [...sources];
  while (queue.length) {
    const u = queue.shift();
    const s = level.get(u);
    for (const e of adj.get(u) ?? []) {
      const strength = edgeStrength(m, e, threat, disabled);
      const ns = strength === 'full' ? LEVEL.contained : strength === 'partial' ? Math.min(s, LEVEL.damped) : s;
      if ((level.get(e.to) ?? 0) < ns) { level.set(e.to, ns); queue.push(e.to); }
    }
  }
  for (const s of sources) level.delete(s);
  const pick = (n) => [...level].filter(([, l]) => l === n).map(([id]) => id).sort();
  return { unmitigated: pick(3), damped: pick(2), contained: pick(1) };
}

export function radiusTable(m, view, opts = {}) {
  const { kind, sources: pick } = VIEWS[view];
  const sources = pick ? pick(m) : [...new Set(edges(m).filter((e) => e.kind === kind && (!e.baseline || opts.baseline)).map((e) => e.from))];
  return sources
    .map((id) => ({ id, ...propagate(m, id, view, opts) }))
    .filter((r) => r.unmitigated.length + r.damped.length + r.contained.length > 0)
    .sort((a, b) => b.unmitigated.length - a.unmitigated.length || b.damped.length - a.damped.length || a.id.localeCompare(b.id));
}

// Marginal value of a control: exposure pairs (source, node) that would become
// unmitigated if the control were removed, summed over all sources and kinds.
export function controlValue(m) {
  const used = new Set(edges(m).filter((e) => !e.baseline).flatMap((e) => e.controls));
  const unmitigatedPairs = (view, disabled) =>
    radiusTable(m, view, { disabled }).reduce((n, r) => n + r.unmitigated.length, 0);
  const base = Object.fromEntries(Object.keys(VIEWS).map((v) => [v, unmitigatedPairs(v, new Set())]));
  return [...used]
    .map((id) => {
      const off = new Set([id]);
      const byView = Object.fromEntries(
        Object.keys(VIEWS).map((v) => [v, unmitigatedPairs(v, off) - base[v]]).filter(([, n]) => n > 0),
      );
      return { id, strength: m.controls[id].strength, status: m.controls[id].status, byView, protected: Object.values(byView).reduce((a, b) => a + b, 0) };
    })
    .sort((a, b) => b.protected - a.protected || a.id.localeCompare(b.id));
}

// Elementary cycles of the trust graph through a data-layer node and an agent:
// the loops through which injected content can persist across sessions.
export function persistentCycles(m, { baseline = false, disabled = new Set() } = {}) {
  const adj = adjacency(m, 'trust', baseline);
  const layer = new Map(m.components.map((c) => [c.id, c.layer]));
  const agent = new Set(m.components.filter((c) => c.agent).map((c) => c.id));
  const nodes = m.components.map((c) => c.id);
  const idx = new Map(nodes.map((n, i) => [n, i]));
  const found = [];
  for (const s of nodes) {
    const path = [];
    const onPath = new Set([s]);
    const dfs = (u) => {
      for (const e of adj.get(u) ?? []) {
        if (e.to === s) found.push([...path, e]);
        else if (idx.get(e.to) > idx.get(s) && !onPath.has(e.to)) {
          onPath.add(e.to); path.push(e); dfs(e.to); path.pop(); onPath.delete(e.to);
        }
      }
    };
    dfs(s);
  }
  const accepted = m.acceptedCycles.map((c) => ({ ...c, key: [...c.nodes].sort().join(',') }));
  return found
    .filter((cyc) => cyc.some((e) => layer.get(e.from) === 'data') && cyc.some((e) => agent.has(e.from)))
    .map((cyc) => {
      const nodesOn = cyc.map((e) => e.from);
      const controls = [...new Set(cyc.flatMap((e) => e.controls).filter((c) => !disabled.has(c) && applies(m.controls[c], 'integrity')))];
      const contained = cyc.some((e) => edgeStrength(m, e, 'integrity', disabled) === 'full');
      const key = [...nodesOn].sort().join(',');
      const acc = accepted.find((a) => a.key === key) ?? null;
      return { nodes: nodesOn, controls, contained, accepted: acc?.decision ?? null, baseline: cyc.some((e) => e.baseline) };
    });
}

// Elementary cycles miss loops that pass one node twice (agent -> server -> store -> server ->
// agent). The complete criterion: drop every edge a full integrity control closes; an agent
// that still shares a strongly connected component with a data node can have content it
// wrote (or was fed) come back to it in a later session.
export function persistentExposure(m, { disabled = new Set() } = {}) {
  const adj = new Map();
  for (const e of edges(m)) {
    if (e.kind !== 'trust' || e.baseline || edgeStrength(m, e, 'integrity', disabled) === 'full') continue;
    if (!adj.has(e.from)) adj.set(e.from, []);
    adj.get(e.from).push(e.to);
  }
  // Tarjan's algorithm, iterative to stay safe on deep graphs.
  let counter = 0;
  const index = new Map();
  const low = new Map();
  const onStack = new Set();
  const stack = [];
  const sccOf = new Map();
  for (const root of m.components.map((c) => c.id)) {
    if (index.has(root)) continue;
    const work = [[root, 0]];
    index.set(root, counter); low.set(root, counter++); stack.push(root); onStack.add(root);
    while (work.length) {
      const frame = work[work.length - 1];
      const [v, i] = frame;
      const next = (adj.get(v) ?? [])[i];
      if (next !== undefined) {
        frame[1]++;
        if (!index.has(next)) {
          index.set(next, counter); low.set(next, counter++); stack.push(next); onStack.add(next);
          work.push([next, 0]);
        } else if (onStack.has(next)) {
          low.set(v, Math.min(low.get(v), index.get(next)));
        }
        continue;
      }
      work.pop();
      if (work.length) low.set(work[work.length - 1][0], Math.min(low.get(work[work.length - 1][0]), low.get(v)));
      if (low.get(v) === index.get(v)) {
        const comp = [];
        let w;
        do { w = stack.pop(); onStack.delete(w); comp.push(w); } while (w !== v);
        for (const n of comp) sccOf.set(n, comp);
      }
    }
  }
  const layer = new Map(m.components.map((c) => [c.id, c.layer]));
  const out = [];
  for (const c of m.components.filter((x) => x.agent)) {
    const comp = sccOf.get(c.id) ?? [c.id];
    for (const d of comp.filter((n) => layer.get(n) === 'data')) {
      const accepted = m.acceptedCycles.find((a) => a.nodes.includes(c.id) && a.nodes.includes(d))?.decision ?? null;
      out.push({ agent: c.id, data: d, component: [...comp].sort(), accepted });
    }
  }
  return out;
}

// ---------------------------------------------------------------- risks

export const rpn = ([s, l, d]) => s * l * d;

export function rankRisks(m) {
  return m.failureModes
    .map((f) => ({ ...f, inherentRpn: rpn(f.inherent), residualRpn: rpn(f.residual) }))
    .sort((a, b) => b.residualRpn - a.residualRpn || b.residual[0] - a.residual[0] || a.id.localeCompare(b.id));
}

// ---------------------------------------------------------------- full check

// `disabled` evaluates the model with those controls absent, for example every control whose
// status is "proposed" (see asBuilt) to check the system as it exists today.
export function runChecks(raw, { disabled = new Set() } = {}) {
  const m = normalizeModel(raw);
  const problems = [...validateModel(m), ...globalInvariants(m)];
  if (problems.length) return problems; // later checks assume a well-formed model
  for (const w of m.workflows) {
    const r = checkWorkflow(m, w);
    for (const v of r.violations) problems.push(`${w.id} ${v.rule} at ${v.where}: ${v.msg}`);
  }
  for (const a of m.antiWorkflows) {
    const got = checkWorkflow(m, a).rules.join(',');
    const want = [...a.expect].sort().join(',');
    if (got !== want) problems.push(`${a.id} expected [${want}] got [${got}]`);
  }
  const cycles = persistentCycles(m, { disabled });
  for (const c of cycles) {
    if (!c.contained && !c.accepted) problems.push(`G-TRUST-CYCLE: uncontained persistent loop ${c.nodes.join(' -> ')}`);
  }
  for (const x of persistentExposure(m, { disabled })) {
    if (!x.accepted) problems.push(`G-TRUST-CYCLE: ${x.agent} and ${x.data} share an uncontained loop (${x.component.join(', ')})`);
  }
  const live = new Set(cycles.filter((c) => !c.contained).map((c) => [...c.nodes].sort().join(',')));
  for (const a of m.acceptedCycles) {
    if (!live.has([...a.nodes].sort().join(','))) problems.push(`G-TRUST-CYCLE: accepted cycle ${a.nodes.join(' -> ')} is contained or absent; remove it`);
  }
  return problems;
}

// ---------------------------------------------------------------- porter extensions

// Controls that exist only on paper: disable them to see the system as built today.
export const asBuilt = (m) => new Set(Object.entries(normalizeModel(m).controls).filter(([, c]) => c.status === 'proposed').map(([id]) => id));

// Views whose edge kind actually occurs in the model.
export function applicableViews(m) {
  const kinds = new Set(edges(m).map((e) => e.kind));
  return Object.keys(VIEWS).filter((v) => kinds.has(VIEWS[v].kind) && (!VIEWS[v].sources || VIEWS[v].sources(m).length));
}

export function summarizeModel(raw) {
  const m = normalizeModel(raw);
  const byLayer = {};
  for (const c of m.components) byLayer[c.layer ?? 'unlayered'] = (byLayer[c.layer ?? 'unlayered'] ?? 0) + 1;
  const byKind = {};
  for (const e of m.edges) byKind[e[2]] = (byKind[e[2]] ?? 0) + 1;
  return {
    components: m.components.length, byLayer, edges: m.edges.length, byKind,
    controls: Object.keys(m.controls).length, capabilities: m.capabilities.length,
    workflows: m.workflows.length, entries: m.entries.length, exits: m.exits.length,
    failureModes: m.failureModes.length, views: applicableViews(m),
  };
}

// Chains of capabilities whose types line up (out of one = in of the next), spanning
// repos: the use cases that emerge from composing repos' functional processes.
// Fragility = every component whose change or runtime failure can break the chain:
// its own repos plus everything upstream of them (listed in `upstream`).
export function emergentUseCases(raw, { maxLength = 3, crossRepoOnly = true, limit = 50 } = {}) {
  const m = normalizeModel(raw);
  const byIn = new Map();
  for (const c of m.capabilities) byIn.set(c.in, [...(byIn.get(c.in) ?? []), c]);
  const chains = [];
  const walk = (path) => {
    if (path.length >= 2) {
      const repos = [...new Set(path.map((c) => c.repo))];
      if (!crossRepoOnly || repos.length > 1) chains.push({ path, repos });
    }
    if (path.length === maxLength) return;
    for (const next of byIn.get(path.at(-1).out) ?? []) if (!path.includes(next)) walk([...path, next]);
  };
  for (const c of m.capabilities) walk([c]);
  const views = ['change', 'runtime'].filter((v) => applicableViews(m).includes(v));
  const fragility = new Map();
  const upstream = (repos) => {
    const key = [...repos].sort().join(',');
    if (!fragility.has(key)) {
      const hit = new Set();
      for (const v of views) {
        const r = propagate(m, repos, v, { reverse: true });
        for (const id of [...r.unmitigated, ...r.damped]) hit.add(id);
      }
      fragility.set(key, [...hit].sort());
    }
    return fragility.get(key);
  };
  const out = chains.map(({ path, repos }) => {
    const up = upstream(repos);
    return {
      id: path.map((c) => c.id).join(' > '),
      capabilities: path.map((c) => c.id),
      types: [path[0].in, ...path.map((c) => c.out)],
      repos,
      effects: [...new Set(path.flatMap((c) => c.effects ?? []))].sort(),
      fragility: repos.length + up.length,
      upstream: up,
    };
  });
  out.sort((a, b) => a.capabilities.length - b.capabilities.length || a.fragility - b.fragility || a.id.localeCompare(b.id));
  return { total: out.length, useCases: out.slice(0, limit) };
}

// Union of several models into one graph. Components that name the same remote are one
// repository seen by two sources (Atlas names repos after directories, discovery after
// remotes), so later ids are renamed to the first id seen for that remote. Every other
// collision keeps the first definition and is reported, never silently overwritten.
export function mergeModels(raws) {
  const models = raws.map(normalizeModel);
  const out = normalizeModel({ modelVersion: 1, about: `Merged from ${models.length} models.` });
  const renamed = [];
  const conflicts = [];
  const byId = new Map();
  const byRemote = new Map();
  models.forEach((m, mi) => {
    const rename = new Map();
    for (const c of m.components) {
      const twin = c.remote ? byRemote.get(c.remote) : undefined;
      const id = twin && twin !== c.id ? twin : c.id;
      if (id !== c.id) { rename.set(c.id, id); renamed.push({ model: mi, from: c.id, to: id, remote: c.remote }); }
      const prev = byId.get(id);
      if (!prev) {
        const comp = { ...c, id };
        byId.set(id, comp);
        out.components.push(comp);
        if (c.remote) byRemote.set(c.remote, id);
        continue;
      }
      for (const [k, v] of Object.entries(c)) {
        if (k === 'id' || v === undefined) continue;
        if (prev[k] === undefined) prev[k] = v;
        else if (JSON.stringify(prev[k]) !== JSON.stringify(v) && k !== 'name' && k !== 'path') conflicts.push(`component ${id}.${k}: kept ${JSON.stringify(prev[k])}, dropped ${JSON.stringify(v)} (model ${mi})`);
      }
    }
    const r = (id) => rename.get(id) ?? id;
    for (const [id, c] of Object.entries(m.controls)) {
      if (!out.controls[id]) out.controls[id] = c;
      else if (out.controls[id].strength !== c.strength) conflicts.push(`control ${id}: kept strength ${out.controls[id].strength}, dropped ${c.strength} (model ${mi})`);
    }
    for (const [from, to, kind, controls = null, flag = null] of m.edges) {
      const key = `${r(from)}>${r(to)}>${kind}>${flag ?? ''}`;
      const existing = out.edges.find((e) => `${e[0]}>${e[1]}>${e[2]}>${e[4] ?? ''}` === key);
      const ctl = controls === null ? [] : [].concat(controls);
      if (!existing) {
        const edge = [r(from), r(to), kind];
        if (ctl.length || flag) edge.push(ctl.length ? ctl : null);
        if (flag) edge.push(flag);
        out.edges.push(edge);
      } else if (ctl.length) {
        const merged = [...new Set([...(existing[3] === null || existing[3] === undefined ? [] : [].concat(existing[3])), ...ctl])];
        existing[3] = merged;
      }
    }
    const seenCap = new Set(out.capabilities.map((c) => c.id));
    for (const c of m.capabilities) {
      if (seenCap.has(c.id)) { conflicts.push(`capability ${c.id}: kept the first definition (model ${mi})`); continue; }
      seenCap.add(c.id);
      out.capabilities.push({ ...c, repo: r(c.repo) });
    }
    const seenFm = new Set(out.failureModes.map((f) => f.id));
    for (const f of m.failureModes) {
      if (seenFm.has(f.id)) { conflicts.push(`failure mode ${f.id}: kept the first definition (model ${mi})`); continue; }
      seenFm.add(f.id);
      out.failureModes.push({ ...f, at: r(f.at) });
    }
    for (const a of m.acceptedCycles) out.acceptedCycles.push({ ...a, nodes: a.nodes.map(r) });
  });
  return { model: out, renamed, conflicts };
}
