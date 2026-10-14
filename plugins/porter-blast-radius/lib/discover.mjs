// Builds a Porter coupling model from local repositories, deterministically and without
// running any program: git facts come from reading .git files (so a hostile repository
// cannot run code through git config such as core.fsmonitor), couplings come from
// dependency manifests and from each repo's optional porter.json.

import { createHash } from 'node:crypto';
import { readdirSync, realpathSync, statSync } from 'node:fs';
import { basename, join, resolve, sep } from 'node:path';
import { ToolError, confine, display, tryRead, walk } from './fsguard.mjs';

const SHA = /^[0-9a-f]{40}([0-9a-f]{24})?$/;
export const ID_RE = /^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}$/;
const TYPE_RE = /^[A-Za-z][A-Za-z0-9._:-]{0,63}$/;
const EFFECT_RE = /^[a-z][a-z0-9._-]{0,31}$/;

// Strips control and bidirectional-formatting characters and bounds length, so text
// copied from repository files cannot reshape the tool output it lands in.
export const cleanText = (s, max = 200) =>
  String(s ?? '').replace(/[\u0000-\u001f\u007f\u200b-\u200f\u202a-\u202e\u2066-\u2069]/g, ' ').slice(0, max);

// ---------------------------------------------------------------- git

// host/owner/repo, lowercased, with scheme, credentials, port and .git dropped.
// Local paths (file://, Windows drives, relative paths) have no remote identity.
export function normalizeRemote(url) {
  const s = String(url ?? '').trim();
  // Dot segments make identity ambiguous (URL parsing would silently resolve them), and no
  // real remote has them, so refuse rather than guess.
  if (/(^|[/:])\.\.?(\/|$)/.test(s.replace(/^[a-z][a-z0-9+.-]*:\/\//i, ''))) return undefined;
  let host;
  let path;
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(s)) {
    let u;
    try {
      u = new URL(s);
    } catch {
      return undefined;
    }
    if (u.protocol === 'file:') return undefined;
    host = u.hostname;
    path = decodeURIComponent(u.pathname);
  } else {
    const scp = /^(?:[^@/\s]+@)?([^:/\s]+):(?!\/)(\S+)$/.exec(s);
    if (!scp || scp[1].length === 1) return undefined; // "C:\repo" is a drive, not a host
    [, host, path] = scp;
  }
  if (path.includes('\\')) return undefined;
  host = host.toLowerCase().replace(/^www\./, '');
  path = path.replace(/^\/+|\/+$/g, '').replace(/\.git$/i, '').toLowerCase();
  if (!host || !path || path.split('/').some((seg) => seg === '..' || seg === '.')) return undefined;
  return `${host}/${path}`;
}

export function parseRemotes(configText) {
  const out = new Map();
  let current = null;
  for (const line of String(configText).split(/\r?\n/)) {
    const section = /^\s*\[\s*([A-Za-z0-9.-]+)(?:\s+"((?:[^"\\]|\\.)*)")?\s*\]/.exec(line);
    if (section) {
      current = section[1].toLowerCase() === 'remote' ? section[2] ?? null : null;
      continue;
    }
    if (current === null) continue;
    const kv = /^\s*url\s*=\s*(.*?)\s*$/i.exec(line);
    if (kv && !out.has(current)) out.set(current, kv[1].replace(/^"(.*)"$/, '$1'));
  }
  return out;
}

const insideAny = (roots, p) => {
  try {
    const r = realpathSync(p);
    return roots.some((root) => r === root || r.startsWith(root + sep));
  } catch {
    return false;
  }
};

// Reads HEAD, the ref it names (loose or packed) and the origin URL. Worktrees and
// submodules (a .git file with gitdir:) are followed only while they stay inside the roots.
export function gitFacts(dir, roots) {
  const dotgit = join(dir, '.git');
  let st;
  try {
    st = statSync(dotgit);
  } catch {
    return { warnings: [] };
  }
  const warnings = [];
  let gitdir = dotgit;
  if (st.isFile()) {
    const m = /^gitdir:\s*(.+?)\s*$/m.exec(tryRead(dotgit, 4096) ?? '');
    if (!m) return { warnings: ['unreadable .git file'] };
    gitdir = resolve(dir, m[1]);
  }
  const cd = tryRead(join(gitdir, 'commondir'), 4096)?.trim();
  const common = cd ? resolve(gitdir, cd) : gitdir;
  if (!insideAny(roots, gitdir) || !insideAny(roots, common)) {
    return { warnings: ['git directory is outside the allowed roots; git facts skipped'] };
  }
  const head = tryRead(join(gitdir, 'HEAD'), 4096)?.trim() ?? '';
  let branch;
  let commit;
  const ref = /^ref:\s*(refs\/[A-Za-z0-9._/-]+)$/.exec(head)?.[1];
  if (ref && !ref.split('/').includes('..')) {
    if (ref.startsWith('refs/heads/')) branch = ref.slice('refs/heads/'.length);
    commit = readRef(gitdir, common, ref);
  } else if (SHA.test(head)) {
    commit = head;
  }
  const remotes = parseRemotes(tryRead(join(common, 'config'), 256 * 1024) ?? '');
  const url = remotes.get('origin') ?? [...remotes.values()][0];
  const remote = url ? normalizeRemote(url) : undefined;
  if (url && !remote) warnings.push('remote URL is not a network remote; no remote identity');
  return { branch, commit, remote, warnings };
}

function readRef(gitdir, common, ref) {
  for (const d of new Set([gitdir, common])) {
    const t = tryRead(join(d, ref), 256)?.trim();
    if (t && SHA.test(t)) return t;
  }
  for (const line of (tryRead(join(common, 'packed-refs'), 4 * 1024 * 1024) ?? '').split('\n')) {
    const m = /^([0-9a-f]{40}(?:[0-9a-f]{24})?) (refs\/\S+)$/.exec(line);
    if (m && m[2] === ref) return m[1];
  }
  return undefined;
}

export const projectKey = (remote, dir) =>
  createHash('sha256').update(`porter-radius/1\n${remote ? `remote:${remote}` : `path:${dir}`}`).digest('hex').slice(0, 16);

// ---------------------------------------------------------------- manifests

const pep503 = (n) => n.toLowerCase().replace(/[-_.]+/g, '-');

// Git URL from a dependency spec, with the ref (#… or a trailing @ref after the path) removed.
export function gitUrlRemote(spec) {
  if (!spec) return undefined;
  let s = String(spec).trim().replace(/^git\+/, '').replace(/#.*$/, '');
  const scheme = s.indexOf('://');
  const pathStart = s.indexOf('/', scheme >= 0 ? scheme + 3 : 0);
  const at = s.lastIndexOf('@');
  if (scheme >= 0 && pathStart > 0 && at > pathStart) s = s.slice(0, at);
  return normalizeRemote(s);
}

function pep508(spec) {
  const s = String(spec).trim();
  const name = /^([A-Za-z0-9][A-Za-z0-9._-]*)/.exec(s)?.[1];
  const url = /@\s*((?:git\+)?(?:https?|ssh):\/\/\S+|git\+\S+)/.exec(s)?.[1];
  return { name: name ? pep503(name) : undefined, remote: gitUrlRemote(url) };
}

const strings = (text) => [...String(text).matchAll(/"((?:[^"\\]|\\.)*)"|'([^']*)'/g)].map((m) => m[1] ?? m[2]);

function npmSpecRemote(spec) {
  const s = String(spec).trim().replace(/#.*$/, '');
  const host = /^(github|gitlab|bitbucket):([\w.-]+\/[\w.-]+)$/.exec(s);
  if (host) return `${{ github: 'github.com', gitlab: 'gitlab.com', bitbucket: 'bitbucket.org' }[host[1]]}/${host[2].toLowerCase().replace(/\.git$/, '')}`;
  if (/^(git\+|git:\/\/)/.test(s) || /^https?:\/\/.+\.git$/.test(s)) return gitUrlRemote(s);
  if (/^[A-Za-z0-9][\w.-]*\/[\w.-]+$/.test(s)) return `github.com/${s.toLowerCase().replace(/\.git$/, '')}`; // npm's owner/repo shorthand
  return undefined;
}

export function parsePackageJson(text) {
  const j = JSON.parse(text);
  if (!j || typeof j !== 'object') return { provides: [], deps: [] };
  const deps = [];
  for (const [field, dev] of [['dependencies', false], ['peerDependencies', false], ['optionalDependencies', false], ['devDependencies', true]]) {
    for (const [name, spec] of Object.entries(j[field] ?? {})) deps.push({ key: `npm:${name.toLowerCase()}`, dev, remote: npmSpecRemote(spec) });
  }
  return { provides: typeof j.name === 'string' ? [`npm:${j.name.toLowerCase()}`] : [], deps };
}

export function parsePyproject(text) {
  const provides = [];
  const deps = [];
  let table = '';
  let pending = null; // { dev, buf } while a multi-line array is open
  const addSpec = (spec, dev) => {
    const { name, remote } = pep508(spec);
    if (name || remote) deps.push({ key: name ? `pypi:${name}` : undefined, dev, remote });
  };
  for (const raw of String(text).split(/\r?\n/)) {
    const line = raw.replace(/\s+#.*$/, '');
    if (pending) {
      pending.buf += `\n${line}`;
      if (/\]\s*$/.test(line)) { strings(pending.buf).forEach((s) => addSpec(s, pending.dev)); pending = null; }
      continue;
    }
    const header = /^\s*\[\[?\s*([^\]]+?)\s*\]\]?\s*$/.exec(line);
    if (header) { table = header[1].replace(/\s+/g, ''); continue; }
    const kv = /^\s*"?([A-Za-z0-9_.-]+)"?\s*=\s*(.*)$/.exec(line);
    if (!kv) continue;
    const [, key, val] = kv;
    const array = (dev) => { if (/\]\s*$/.test(val)) strings(val).forEach((s) => addSpec(s, dev)); else pending = { dev, buf: val }; };
    if ((table === 'project' || table === 'tool.poetry') && key === 'name') provides.push(`pypi:${pep503(strings(val)[0] ?? '')}`);
    else if (table === 'project' && key === 'dependencies') array(false);
    else if (table === 'project.optional-dependencies' || table === 'dependency-groups') array(true);
    else if (/^tool\.poetry\.(dependencies|dev-dependencies|group\.[^.]+\.dependencies)$/.test(table) && key !== 'python') {
      const git = /git\s*=\s*"([^"]+)"/.exec(val)?.[1];
      deps.push({ key: `pypi:${pep503(key)}`, dev: table !== 'tool.poetry.dependencies', remote: gitUrlRemote(git) });
    }
  }
  return { provides: provides.filter((p) => p !== 'pypi:'), deps };
}

export function parseRequirements(text, dev = false) {
  const deps = [];
  for (const raw of String(text).split(/\r?\n/)) {
    const line = raw.replace(/(^|\s)#.*$/, '').trim();
    if (!line || /^(-r|-c|--requirement|--constraint)\b/.test(line)) continue;
    const editable = /^(?:-e|--editable)\s+(\S+)/.exec(line);
    if (editable) {
      const remote = /^(git\+|https?:\/\/)/.test(editable[1]) ? gitUrlRemote(editable[1]) : undefined;
      const egg = /#egg=([\w.-]+)/.exec(editable[1])?.[1];
      if (remote || egg) deps.push({ key: egg ? `pypi:${pep503(egg)}` : undefined, dev, remote });
      continue;
    }
    if (line.startsWith('-')) continue;
    if (/^(git\+|https?:\/\/)/.test(line)) { deps.push({ key: undefined, dev, remote: gitUrlRemote(line) }); continue; }
    const { name, remote } = pep508(line);
    if (name) deps.push({ key: `pypi:${name}`, dev, remote });
  }
  return { provides: [], deps };
}

const goRemote = (path) => (/^[a-z0-9.-]+\.[a-z]{2,}\/[^/]+\/[^/]+/i.test(path) ? normalizeRemote(`https://${path.split('/').slice(0, 3).join('/')}`) : undefined);

export function parseGoMod(text) {
  const t = String(text);
  const mod = /^\s*module\s+(\S+)/m.exec(t)?.[1];
  const deps = [];
  for (const m of t.matchAll(/^\s*require\s+([^\s(]+)\s+v\S+/gm)) deps.push({ key: `go:${m[1]}`, dev: false, remote: goRemote(m[1]) });
  for (const block of t.matchAll(/^\s*require\s*\(([\s\S]*?)\)/gm)) {
    for (const m of block[1].matchAll(/^\s*([^\s/][^\s]*)\s+v\S+/gm)) deps.push({ key: `go:${m[1]}`, dev: false, remote: goRemote(m[1]) });
  }
  return { provides: mod ? [`go:${mod}`] : [], deps, providesRemote: mod ? goRemote(mod) : undefined };
}

export function parseCargo(text) {
  const crate = (n) => n.toLowerCase().replace(/_/g, '-');
  const provides = [];
  const deps = [];
  let table = '';
  let tableDep = null;
  for (const raw of String(text).split(/\r?\n/)) {
    const line = raw.replace(/\s+#.*$/, '');
    const header = /^\s*\[\s*([^\]]+?)\s*\]\s*$/.exec(line);
    if (header) {
      table = header[1].replace(/\s+/g, '');
      const dep = /^(?:target\..+\.)?(dependencies|dev-dependencies|build-dependencies)\.([A-Za-z0-9_-]+)$/.exec(table);
      tableDep = dep ? { key: `cargo:${crate(dep[2])}`, dev: dep[1] !== 'dependencies', remote: undefined } : null;
      if (tableDep) deps.push(tableDep);
      continue;
    }
    const kv = /^\s*([A-Za-z0-9_-]+)\s*=\s*(.*)$/.exec(line);
    if (!kv) continue;
    const [, key, val] = kv;
    if (table === 'package' && key === 'name') provides.push(`cargo:${crate(strings(val)[0] ?? '')}`);
    else if (tableDep && key === 'git') tableDep.remote = gitUrlRemote(strings(val)[0]);
    else if (/^(?:target\..+\.)?(dependencies|dev-dependencies|build-dependencies)$/.test(table)) {
      const git = /git\s*=\s*"([^"]+)"/.exec(val)?.[1];
      deps.push({ key: `cargo:${crate(key)}`, dev: !table.endsWith('.dependencies') && table !== 'dependencies', remote: gitUrlRemote(git) });
    }
  }
  return { provides: provides.filter((p) => p !== 'cargo:'), deps };
}

// ---------------------------------------------------------------- per repo

const MANIFESTS = [
  ['package.json', (t) => parsePackageJson(t)],
  ['pyproject.toml', (t) => parsePyproject(t)],
  ['go.mod', (t) => parseGoMod(t)],
  ['Cargo.toml', (t) => parseCargo(t)],
];

const slugOf = (s) => String(s).toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^[^a-z0-9]+/, '').slice(0, 64) || 'repo';

export function scanRepo(dir, roots) {
  const facts = gitFacts(dir, roots);
  const warnings = [...facts.warnings];
  const provides = [];
  const deps = [];
  const remotes = [facts.remote].filter(Boolean);
  const take = (file, parsed) => {
    provides.push(...parsed.provides);
    if (parsed.providesRemote) remotes.push(parsed.providesRemote);
    deps.push(...parsed.deps.map((d) => ({ ...d, file })));
  };
  for (const [file, parse] of MANIFESTS) {
    const text = tryRead(join(dir, file));
    if (text === undefined) continue;
    try {
      take(file, parse(text));
    } catch (e) {
      warnings.push(`${file}: ${cleanText(e.message, 120)}`);
    }
  }
  let names = [];
  try {
    names = readdirSync(dir).filter((n) => /^requirements[\w.-]*\.txt$/i.test(n));
  } catch { /* unreadable directory: manifests above already failed soft */ }
  for (const n of names) {
    const text = tryRead(join(dir, n));
    if (text !== undefined) take(n, parseRequirements(text, /dev|test|lint|doc/i.test(n)));
  }
  let porter;
  for (const file of ['porter.json', join('.porter', 'porter.json')]) {
    const text = tryRead(join(dir, file));
    if (text === undefined) continue;
    try {
      porter = JSON.parse(text);
      break;
    } catch (e) {
      warnings.push(`${file}: invalid JSON (${cleanText(e.message, 80)})`);
    }
  }
  const slug = slugOf((typeof porter?.name === 'string' && porter.name) || facts.remote?.split('/').at(-1) || basename(dir));
  return { dir, slug, facts, provides: [...new Set(provides)], deps, remotes: [...new Set(remotes)], porter, warnings };
}

// ---------------------------------------------------------------- model

// Merges repo scans into one coupling model. Edge A -> B means B depends on A, so a
// change or failure in A can reach B (the relay model's edge convention).
export function buildCouplingModel(scans, { includeDev = false, roots = [] } = {}) {
  const warnings = [];
  const evidence = [];
  const components = [];
  const used = new Set();
  for (const s of scans) {
    let slug = s.slug;
    for (let i = 2; used.has(slug); i++) slug = `${s.slug}-${i}`;
    used.add(slug);
    s.id = `repo:${slug}`;
    components.push({
      id: s.id, layer: 'repo', name: slug, path: display(roots, s.dir),
      remote: s.facts.remote, branch: s.facts.branch ? cleanText(s.facts.branch, 100) : undefined,
      commit: s.facts.commit?.slice(0, 12), projectKey: projectKey(s.facts.remote, s.dir), provides: s.provides,
    });
    for (const w of s.warnings) warnings.push(`${s.id}: ${w}`);
  }
  const providers = new Map();
  const byRemote = new Map();
  const bySlug = new Map(scans.map((s) => [s.id.slice(5), s.id]));
  for (const s of scans) {
    for (const p of s.provides) {
      if (providers.has(p) && providers.get(p) !== s.id) warnings.push(`${p} is provided by both ${providers.get(p)} and ${s.id}; using the first`);
      else providers.set(p, s.id);
    }
    for (const r of s.remotes) if (!byRemote.has(r)) byRemote.set(r, s.id);
  }

  const edges = [];
  const seen = new Set();
  const addEdge = (from, to, kind, controls, via) => {
    const key = `${from}>${to}>${kind}`;
    if (from === to || seen.has(key)) return;
    seen.add(key);
    edges.push(controls?.length ? [from, to, kind, controls] : [from, to, kind]);
    evidence.push({ from, to, kind, via });
  };
  for (const s of scans) {
    for (const d of s.deps) {
      if (d.dev && !includeDev) continue;
      const from = (d.key && providers.get(d.key)) || (d.remote && byRemote.get(d.remote));
      if (!from) continue;
      const via = `${d.key ?? d.remote} in ${d.file}${d.dev ? ' (dev)' : ''}`;
      addEdge(from, s.id, 'change', null, via);
      addEdge(from, s.id, 'runtime', null, via);
    }
  }

  const controls = {};
  for (const s of scans) {
    for (const [id, c] of Object.entries(s.porter?.controls ?? {})) {
      if (!ID_RE.test(id) || (c?.strength !== 'full' && c?.strength !== 'partial')) { warnings.push(`${s.id}: control ${cleanText(id, 40)} ignored (needs an id and strength full|partial)`); continue; }
      if (controls[id]) continue;
      controls[id] = {
        strength: c.strength, status: 'declared', mechanism: cleanText(c.mechanism, 300),
        ...(Array.isArray(c.threats) ? { threats: c.threats.filter((t) => t === 'integrity' || t === 'confidentiality') } : {}),
      };
    }
  }

  const capabilities = [];
  for (const s of scans) {
    const slug = s.id.slice(5);
    for (const dep of s.porter?.dependsOn ?? []) {
      const target = typeof dep?.target === 'string' ? dep.target : '';
      const from = byRemote.get(normalizeRemote(target) ?? '') ?? byRemote.get(target.toLowerCase()) ?? bySlug.get(target) ?? (target.startsWith('repo:') ? target : undefined);
      if (!from || !components.some((c) => c.id === from)) { warnings.push(`${s.id}: dependsOn target ${cleanText(target, 80)} is not among the scanned repos`); continue; }
      const kinds = (Array.isArray(dep.kinds) ? dep.kinds : ['change', 'runtime']).filter((k) => ['change', 'runtime', 'trust'].includes(k));
      const ctl = [];
      for (const c of Array.isArray(dep.controls) ? dep.controls : []) {
        if (controls[c]) ctl.push(c);
        else warnings.push(`${s.id}: unknown control ${cleanText(c, 40)} on dependsOn ${cleanText(target, 60)}`);
      }
      for (const k of kinds) addEdge(from, s.id, k, ctl, `porter.json dependsOn${dep.note ? `: ${cleanText(dep.note, 120)}` : ''}`);
    }
    for (const c of s.porter?.capabilities ?? []) {
      const id = typeof c?.id === 'string' && c.id.startsWith(`${slug}:`) ? c.id : `${slug}:${c?.id}`;
      if (!ID_RE.test(id) || !TYPE_RE.test(c?.in ?? '') || !TYPE_RE.test(c?.out ?? '')) { warnings.push(`${s.id}: capability ${cleanText(c?.id, 60)} ignored (id, in and out must be simple names)`); continue; }
      capabilities.push({
        id, repo: s.id, in: c.in, out: c.out,
        effects: (Array.isArray(c.effects) ? c.effects : []).filter((e) => EFFECT_RE.test(e)),
        ...(c.note ? { note: cleanText(c.note, 200) } : {}),
      });
    }
  }

  const model = {
    modelVersion: 1,
    about: 'Porter coupling model generated by porter_discover. Edge A -> B: B depends on A. Review, then save as porter.model.json.',
    components, edges, controls, capabilities, acceptedCycles: [], failureModes: [],
  };
  return { model, evidence, warnings };
}

// Repositories to scan: the given paths (confined), or every directory holding a .git
// entry within three levels of the roots.
export function discover(roots, { repos, includeDev = false, maxRepos = 50 } = {}) {
  let dirs;
  if (Array.isArray(repos) && repos.length) {
    dirs = repos.map((p) => {
      const real = confine(roots, p);
      if (!statSync(real).isDirectory()) throw new ToolError(`${p} is not a directory`);
      return real;
    });
  } else {
    dirs = [];
    walk(roots, { depth: 3, onDir: (dir, entries) => { if (dirs.length < maxRepos && entries.some((e) => e.name === '.git')) dirs.push(dir); } });
  }
  if (!dirs.length) throw new ToolError('no repositories found; pass repo paths inside the allowed roots');
  const scans = [...new Set(dirs)].slice(0, maxRepos).map((d) => scanRepo(d, roots));
  return buildCouplingModel(scans, { includeDev, roots });
}
