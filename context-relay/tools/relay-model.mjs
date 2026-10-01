#!/usr/bin/env node
// Executable model of Context Relay: workflow checking and blast-radius analysis.
// Zero dependencies. Node >= 18.
//
//   node tools/relay-model.mjs check              validate model, workflows, cycles, doc freshness
//   node tools/relay-model.mjs report <section>   print one generated markdown section
//   node tools/relay-model.mjs sync-docs          rewrite generated sections in docs/*.md


import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  checkWorkflow, controlValue, parseModel, persistentCycles, radiusTable, rankRisks, routeMatrix, runChecks,
} from '../plugins/porter-blast-radius/lib/engine.mjs';

// The engine is shared with the porter-blast-radius connector; this file adds the
// relay model's location, generated doc sections and the CLI.
export * from '../plugins/porter-blast-radius/lib/engine.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
export const ROOT = resolve(HERE, '..');
export const MODEL_PATH = join(ROOT, 'model', 'relay-model.json');
export const DOC_PATHS = [join(ROOT, 'docs', 'WORKFLOWS.md'), join(ROOT, 'docs', 'BLAST_RADIUS.md')];

export const loadModel = (path = MODEL_PATH) => parseModel(readFileSync(path, 'utf8'));


// ---------------------------------------------------------------- markdown

const esc = (s) => String(s).replace(/\|/g, '\\|');
function table(headers, rows) {
  return [
    `| ${headers.map(esc).join(' | ')} |`,
    `|${headers.map(() => '---').join('|')}|`,
    ...rows.map((r) => `| ${r.map(esc).join(' | ')} |`),
  ].join('\n');
}
const list = (xs) => (xs.length ? xs.join(', ') : '—');
const pathText = (path) => path.map((p) => (typeof p === 'string' ? p : `${p.op}@${p.at}`)).join(' → ');

export const SECTIONS = {
  primitives(m) {
    const any = (x) => (x === '*' ? 'any' : x.join(', '));
    return table(
      ['Op', 'Type', 'Requires', 'Adds', 'Removes', 'Origin', 'Surfaces', 'Actors', 'Effects'],
      Object.entries(m.primitives).map(([op, p]) => [
        op, `${p.in} → ${p.out}`, list(p.requires ?? []), list(p.adds ?? []), list(p.removes ?? []),
        p.origin ?? '—', any(p.surfaces), any(p.actors), list(p.effects),
      ]),
    );
  },

  workflows(m) {
    return table(
      ['ID', 'Workflow', 'Composition', 'Scenarios', 'Verdict'],
      m.workflows.map((w) => {
        const r = checkWorkflow(m, w);
        return [w.id, w.name + (w.cyclic ? ' (cyclic)' : ''), pathText(w.path), list(w.scenarios ?? []), r.ok ? 'passes' : r.rules.join(', ')];
      }),
    );
  },

  anti(m) {
    return table(
      ['ID', 'Rejected composition', 'Path', 'Rules fired'],
      m.antiWorkflows.map((a) => [a.id, a.name, pathText(a.path), checkWorkflow(m, a).rules.join(', ')]),
    );
  },

  routes(m) {
    const cells = routeMatrix(m);
    const sym = { direct: 'ok', gate: 'gate', forbidden: 'no' };
    const rows = m.entries.map((e) => [
      `${e.id} ${e.name}`,
      ...m.exits.map((x) => sym[cells.find((c) => c.entry === e.id && c.exit === x.id).verdict]),
    ]);
    const n = (v) => cells.filter((c) => c.verdict === v).length;
    const why = [...new Set(cells.filter((c) => c.verdict === 'gate').flatMap((c) => c.rules))].sort();
    return [
      table(['Entry \\ Exit', ...m.exits.map((x) => x.id)], rows),
      '',
      table(['Exit', 'Name'], m.exits.map((x) => [x.id, x.name])),
      '',
      `${m.entries.length} entries + ${m.exits.length} exits = ${m.entries.length + m.exits.length} route definitions → ` +
        `${cells.length} compositions: ${n('direct')} direct, ${n('gate')} need a human approval (${why.join(', ')}), ${n('forbidden')} forbidden.`,
    ].join('\n');
  },

  'radius-integrity': (m) => radiusSection(m, 'integrity'),
  'radius-confidentiality': (m) => radiusSection(m, 'confidentiality'),
  'radius-runtime': (m) => radiusSection(m, 'runtime'),
  'radius-change': (m) => radiusSection(m, 'change'),

  controls(m) {
    return table(
      ['Control', 'Strength', 'Status', 'Pairs protected', 'By view', 'Mechanism'],
      controlValue(m).map((c) => [
        c.id, c.strength, c.status, c.protected,
        list(Object.entries(c.byView).map(([k, n]) => `${k} ${n}`)),
        m.controls[c.id].mechanism,
      ]),
    );
  },

  cycles(m) {
    const rows = (cs) => cs.map((c) => [
      [...c.nodes, c.nodes[0]].join(' → '),
      list(c.controls),
      c.contained ? 'contained' : c.accepted ? `accepted (${c.accepted})` : 'OPEN',
    ]);
    const delta = persistentCycles(m);
    const absolute = persistentCycles(m, { baseline: true }).filter((c) => c.baseline);
    return [
      '**Added by Context Relay (baseline edges excluded):**',
      '',
      table(['Loop', 'Controls on loop', 'Status'], rows(delta)),
      '',
      '**Present only because of host capabilities (baseline edges), shown for completeness:**',
      '',
      table(['Loop', 'Controls on loop', 'Status'], rows(absolute)),
    ].join('\n');
  },

  risks(m) {
    return table(
      ['Rank', 'ID', 'Failure mode', 'Kind', 'Inherent S·L·D', 'Residual S·L·D', 'RPN', 'Controls', 'Level'],
      rankRisks(m).map((f, i) => [
        i + 1, f.id, f.title, f.kind,
        `${f.inherent.join('·')} = ${f.inherentRpn}`, `${f.residual.join('·')} = ${f.residualRpn}`,
        f.residualRpn, f.controls.join(', '), `L${f.level}`,
      ]),
    );
  },

  backlog(m) {
    const partial = Object.entries(m.controls).filter(([, c]) => c.strength === 'partial');
    return table(
      ['Control', 'θ measured', 'θ target', 'Harness'],
      partial.map(([id, c]) => [id, c.theta ?? 'unmeasured', c.thetaTarget ?? 'to set', c.verify]),
    );
  },
};

// Impact column: the agents, egress points and host reached, with their exposure level.
// For confidentiality only egress counts: a secret that stays on the host is not exposed.
function radiusSection(m, view) {
  const counts = view === 'confidentiality' ? (c) => c.egress : (c) => c.agent || c.egress || c.layer === 'sink';
  const marked = new Set(m.components.filter(counts).map((c) => c.id));
  const name = { 3: 'unmitigated', 2: 'damped', 1: 'contained' };
  const rows = radiusTable(m, view).map((r) => {
    const impact = [[r.unmitigated, 3], [r.damped, 2], [r.contained, 1]]
      .flatMap(([ids, l]) => ids.filter((id) => marked.has(id)).map((id) => `${id} ${name[l]}`));
    return [r.id, r.unmitigated.length, r.damped.length, r.contained.length, list(r.unmitigated), list(impact)];
  });
  const label = view === 'confidentiality' ? 'Egress impact' : 'Agent / egress / host impact';
  return table(['Source', 'Unmitigated', 'Damped', 'Contained', 'Unmitigated reach', label], rows);
}

// ---------------------------------------------------------------- doc sync

const BLOCK = /<!-- relay-model:begin ([\w-]+) -->\n[\s\S]*?<!-- relay-model:end \1 -->/g;

export function syncText(m, text) {
  return text.replace(BLOCK, (_, name) => {
    if (!SECTIONS[name]) throw new Error(`unknown generated section ${name}`);
    return `<!-- relay-model:begin ${name} -->\n${SECTIONS[name](m)}\n<!-- relay-model:end ${name} -->`;
  });
}

export function staleDocs(m, paths = DOC_PATHS) {
  return paths.filter((p) => { const t = readFileSync(p, 'utf8'); return syncText(m, t) !== t; });
}

// ---------------------------------------------------------------- cli

function main([cmd, arg]) {
  const m = loadModel();
  if (cmd === 'check') {
    const problems = runChecks(m);
    if (!problems.length) for (const p of staleDocs(m)) problems.push(`stale generated section in ${p}; run sync-docs`);
    const cells = routeMatrix(m);
    console.log(`model: ${m.components.length} components, ${m.edges.length} edges, ${Object.keys(m.controls).length} controls, ` +
      `${m.workflows.length} workflows, ${m.antiWorkflows.length} anti-workflows, ${cells.length} route compositions`);
    for (const p of problems) console.log(`FAIL ${p}`);
    console.log(problems.length ? `${problems.length} problem(s)` : 'all checks pass');
    return problems.length ? 1 : 0;
  }
  if (cmd === 'report' && SECTIONS[arg]) { console.log(SECTIONS[arg](m)); return 0; }
  if (cmd === 'sync-docs') {
    for (const p of DOC_PATHS) writeFileSync(p, syncText(m, readFileSync(p, 'utf8')));
    return 0;
  }
  console.error(`usage: relay-model.mjs check | sync-docs | report <${Object.keys(SECTIONS).join('|')}>`);
  return 2;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.exitCode = main(process.argv.slice(2));
}
