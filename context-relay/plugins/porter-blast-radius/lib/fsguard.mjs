// The connector's only filesystem boundary. Everything it reads goes through here:
// paths are resolved to real paths and must stay inside an allowed root, reads are
// size-bounded, and directory walks never follow symlinks. Nothing here writes.

import { readdirSync, readFileSync, realpathSync, statSync } from 'node:fs';
import { delimiter, isAbsolute, join, relative, resolve, sep } from 'node:path';

export const LIMITS = { modelBytes: 2 * 1024 * 1024, textBytes: 1024 * 1024, walkDepth: 4, walkFiles: 200 };

// A failure the caller should see as a tool error with an actionable message.
export class ToolError extends Error {}

const realDir = (p) => {
  try {
    const r = realpathSync(p);
    return statSync(r).isDirectory() ? r : null;
  } catch {
    return null;
  }
};

// Allowed roots: the project directory plus configured extras. Unexpanded
// placeholders such as "${CLAUDE_PROJECT_DIR}" (a client that does not substitute) are ignored.
export function rootsFromEnv(env = process.env, cwd = process.cwd()) {
  const raw = [env.PORTER_RADIUS_PROJECT, env.CLAUDE_PROJECT_DIR, ...String(env.PORTER_RADIUS_ROOTS ?? '').split(delimiter)];
  const roots = [...new Set(raw.filter((p) => p && !p.includes('${')).map((p) => realDir(p.trim())).filter(Boolean))];
  return roots.length ? roots : [realDir(cwd)].filter(Boolean);
}

const inside = (root, p) => p === root || p.startsWith(root.endsWith(sep) ? root : root + sep);

// Resolves a caller-supplied path (relative paths resolve against the first root)
// and refuses anything whose real path leaves every root.
export function confine(roots, p) {
  if (typeof p !== 'string' || !p || p.includes('\0')) throw new ToolError('path must be a non-empty string');
  const abs = isAbsolute(p) ? p : resolve(roots[0], p);
  let real;
  try {
    real = realpathSync(abs);
  } catch {
    throw new ToolError(`not found: ${p}${isAbsolute(p) ? '' : ` (relative paths resolve against ${roots[0]})`}`);
  }
  if (!roots.some((r) => inside(r, real))) throw new ToolError(`${p} is outside the allowed roots (${roots.join(', ')}); add its directory to the plugin's extra_roots setting`);
  return real;
}

export const display = (roots, real) => {
  const r = roots.find((x) => inside(x, real));
  return r ? relative(r, real) || '.' : real;
};

export function readBounded(path, max = LIMITS.textBytes) {
  const st = statSync(path);
  if (!st.isFile()) throw new ToolError(`${path} is not a regular file`);
  if (st.size > max) throw new ToolError(`${path} is ${st.size} bytes; the limit is ${max}`);
  return readFileSync(path, 'utf8');
}

// Optional read: undefined when missing, unreadable or too large.
export function tryRead(path, max = LIMITS.textBytes) {
  try {
    return readBounded(path, max);
  } catch {
    return undefined;
  }
}

const SKIP = new Set(['node_modules', '.git', 'dist', 'build', 'out', 'target', '.venv', 'venv', '__pycache__', '.next', '.cache', 'vendor']);

// Breadth-first walk from each root, depth- and count-bounded, never following symlinks.
// `visit(dir, entries)` returns nothing; `match(name)` selects files to collect.
export function walk(roots, { match, depth = LIMITS.walkDepth, max = LIMITS.walkFiles, onDir } = {}) {
  const found = [];
  for (const root of roots) {
    let frontier = [root];
    for (let d = 0; d <= depth && frontier.length && found.length < max; d++) {
      const next = [];
      for (const dir of frontier) {
        let entries;
        try {
          entries = readdirSync(dir, { withFileTypes: true });
        } catch {
          continue;
        }
        onDir?.(dir, entries);
        for (const e of entries) {
          if (e.isSymbolicLink()) continue;
          const full = join(dir, e.name);
          if (e.isDirectory()) {
            if (!SKIP.has(e.name) && !e.name.startsWith('.')) next.push(full);
          } else if (e.isFile() && match?.(e.name) && found.length < max) {
            found.push(full);
          }
        }
      }
      frontier = next;
    }
  }
  return [...new Set(found)];
}
