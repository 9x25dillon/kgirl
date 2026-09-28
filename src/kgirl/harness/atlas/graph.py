"""Graph algorithms over the Atlas: import resolution, blast radius, coupling, cards.

Edge semantics
--------------
    import   F imports module M (resolved to a file, possibly in another repo)
    ref      F imports M *and* uses symbol S defined in M   (symbol-level edge)
    clone    F contains a function/class whose normalized body hash equals S's
             (no runtime dependency, but the same defect lives in both places)

Blast radius of a target T is the reverse closure over import/ref edges up to
`max_depth`, plus clone siblings at depth 1. Weight of a node = sum over
discovered paths of 1/depth: cheap, monotone, and it ranks files by how
directly they depend on T.
"""

from __future__ import annotations

import sqlite3
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .store import Atlas

_STDLIB = frozenset(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
_JS_EXT = ("", ".ts", ".tsx", ".js", ".mjs", ".jsx", "/index.ts", "/index.tsx", "/index.js")
_GENERIC_NAMES = frozenset(
    "main run setup init start stop test update reset step forward call apply process execute handle "
    "load save get set build parse render create close open read write config helper utils demo".split())


# ============================================================ import resolution


def resolve_all(db: sqlite3.Connection) -> int:
    files = db.execute("SELECT id, repo_id, path, lang FROM files").fetchall()
    by_path: dict[tuple[int, str], int] = {}
    py_by_last: dict[str, list[tuple[int, int, tuple[str, ...]]]] = defaultdict(list)
    stem_any: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for f in files:
        by_path[(f["repo_id"], f["path"])] = f["id"]
        p = PurePosixPath(f["path"])
        stem_any[p.name].append((f["id"], f["repo_id"], f["path"]))
        if f["lang"] == "python":
            parts = p.with_suffix("").parts if p.suffix else p.parts
            if parts and parts[-1] == "__init__":
                parts = parts[:-1]
            if parts:
                py_by_last[parts[-1]].append((f["id"], f["repo_id"], tuple(parts)))
        elif f["lang"] == "julia":
            py_by_last[p.stem].append((f["id"], f["repo_id"], (p.stem,)))
    meta = {f["id"]: f for f in files}

    def py_suffix(parts: tuple[str, ...], repo_id: int, near: str) -> int | None:
        cands = [c for c in py_by_last.get(parts[-1], ()) if c[2][-len(parts):] == parts]
        if not cands:
            return None
        near_dir = PurePosixPath(near).parent.parts
        # prefer same repo, then longest shared directory prefix, then shortest path
        def key(c):
            shared = 0
            cp = PurePosixPath(meta[c[0]]["path"]).parent.parts
            for a, b in zip(cp, near_dir):
                if a != b:
                    break
                shared += 1
            return (c[1] != repo_id, -shared, len(c[2]))
        return min(cands, key=key)[0]

    updates: list[tuple[int | None, int]] = []
    for imp in db.execute("SELECT i.id, i.file_id, i.target, i.names, i.level, i.kind FROM imports i"):
        f = meta.get(imp["file_id"])
        if f is None:
            continue
        target, lang, repo_id, path = imp["target"] or "", f["lang"], f["repo_id"], f["path"]
        here = PurePosixPath(path).parent
        rid: int | None = None
        if lang == "python":
            names = [n for n in (imp["names"] or "").split(",") if n and n != "*"]
            if imp["level"]:
                base = here.parts[: max(0, len(here.parts) - (imp["level"] - 1))]
                parts = tuple(base) + (tuple(target.split(".")) if target else ())
                for cand in ([parts] if target else []) + [parts + (n,) for n in names]:
                    rel = "/".join(cand)
                    rid = by_path.get((repo_id, rel + ".py")) or by_path.get((repo_id, rel + "/__init__.py"))
                    if rid:
                        break
            elif target and target.split(".")[0] not in _STDLIB:
                parts = tuple(target.split("."))
                rid = py_suffix(parts, repo_id, path)
                if rid is None:
                    for n in names:
                        rid = py_suffix(parts + (n,), repo_id, path)
                        if rid:
                            break
        elif lang in ("javascript", "typescript") and target.startswith("."):
            base = (here / target).as_posix()
            norm = _normpath(base)
            for ext in _JS_EXT:
                rid = by_path.get((repo_id, norm + ext))
                if rid:
                    break
        elif lang == "julia":
            if imp["kind"] == "include":
                rid = by_path.get((repo_id, _normpath((here / target).as_posix())))
            else:
                mod = target.split(".")[0]
                if mod not in ("Base", "Core", "Main"):
                    rid = py_suffix((mod,), repo_id, path)
        elif lang == "rust" and "::" not in target and target.isidentifier():
            rid = by_path.get((repo_id, (here / f"{target}.rs").as_posix())) or \
                by_path.get((repo_id, (here / target / "mod.rs").as_posix()))
        elif lang in ("c", "cpp", "shell", "gdscript", "zig"):
            t = target.replace("res://", "")
            rid = by_path.get((repo_id, _normpath((here / t).as_posix()))) or by_path.get((repo_id, t))
            if rid is None and lang in ("c", "cpp"):
                same = [c for c in stem_any.get(PurePosixPath(t).name, ()) if c[1] == repo_id]
                rid = same[0][0] if len(same) == 1 else None
        if rid == imp["file_id"]:
            rid = None
        updates.append((rid, imp["id"]))
    db.executemany("UPDATE imports SET resolved_file_id=? WHERE id=?", updates)
    return sum(1 for r, _ in updates if r)


def _normpath(p: str) -> str:
    out: list[str] = []
    for part in p.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if out:
                out.pop()
        else:
            out.append(part)
    return "/".join(out)


# ============================================================ blast radius


@dataclass(frozen=True)
class Impact:
    repo: str
    path: str
    depth: int
    via: str        # import | ref | clone
    weight: float
    evidence: str   # e.g. "line 12: from numbskull_dual_orchestrator import X"


@dataclass
class BlastReport:
    target: str
    seeds: list[tuple[str, str]]
    impacts: list[Impact] = field(default_factory=list)

    @property
    def by_repo(self) -> dict[str, int]:
        return dict(Counter(i.repo for i in self.impacts).most_common())

    def to_dict(self) -> dict:
        return {"target": self.target, "seeds": [f"{r}:{p}" for r, p in self.seeds],
                "by_repo": self.by_repo,
                "impacts": [i.__dict__ for i in self.impacts]}

    def render(self, limit: int = 40) -> str:
        if not self.seeds:
            return f"blast radius: target {self.target!r} not found in the atlas"
        lines = [f"blast radius of {self.target}",
                 "  defined in: " + ", ".join(f"{r}:{p}" for r, p in self.seeds[:6]),
                 f"  {len(self.impacts)} dependents across {len(self.by_repo)} repo(s): "
                 + ", ".join(f"{k}={v}" for k, v in self.by_repo.items())]
        for i in self.impacts[:limit]:
            lines.append(f"  d{i.depth} {i.via:<6} w={i.weight:4.2f}  {i.repo}:{i.path}   {i.evidence}")
        if len(self.impacts) > limit:
            lines.append(f"  … {len(self.impacts) - limit} more")
        return "\n".join(lines)


def resolve_target(atlas: "Atlas", target: str) -> tuple[list[int], str | None, list[int]]:
    """Return (seed file ids, symbol name or None, symbol ids) for a user-facing target.

    Accepted forms: `repo:path`, `path`, `path::Qual.name`, `Qual.name`, `name`.
    """
    db = atlas.db
    repo = None
    if ":" in target and "::" not in target.split(":", 1)[0] and not target.startswith("::"):
        head, rest = target.split(":", 1)
        if not rest.startswith(":") and atlas.repo_id(head):
            repo, target = head, rest
    if "::" not in target:
        rows = db.execute(
            "SELECT f.id FROM files f JOIN repos r ON r.id=f.repo_id WHERE (f.path=? OR f.path LIKE ?)"
            + (" AND r.name=?" if repo else ""),
            (target, "%/" + target, repo) if repo else (target, "%/" + target)).fetchall()
        if rows:
            return [r["id"] for r in rows], None, []
    syms = atlas.find_symbol(target, repo)
    if not syms:
        return [], None, []
    name = syms[0]["name"]
    return sorted({s["file_id"] for s in syms}), name, [s["id"] for s in syms]


def blast_radius(atlas: "Atlas", target: str, max_depth: int = 3, include_clones: bool = True) -> BlastReport:
    db = atlas.db
    seeds, symbol, sym_ids = resolve_target(atlas, target)
    info = _file_info(db, seeds)
    report = BlastReport(target, [(info[f][0], info[f][1]) for f in seeds if f in info])
    if not seeds:
        return report
    best: dict[int, tuple[int, str, str]] = {}
    weight: dict[int, float] = defaultdict(float)
    frontier = deque((f, 0) for f in seeds)
    visited = set(seeds)
    while frontier:
        fid, d = frontier.popleft()
        if d >= max_depth:
            continue
        rows = db.execute(
            "SELECT i.file_id, i.line, i.target, i.names FROM imports i WHERE i.resolved_file_id=?",
            (fid,)).fetchall()
        for r in rows:
            src = r["file_id"]
            via = "import"
            if symbol and d == 0:
                names = (r["names"] or "").split(",")
                uses = symbol in names or db.execute(
                    "SELECT 1 FROM refs WHERE file_id=? AND name=?", (src, symbol)).fetchone()
                if not uses:
                    continue
                via = "ref"
            weight[src] += 1.0 / (d + 1)
            if src not in best or best[src][0] > d + 1:
                best[src] = (d + 1, via, f"L{r['line']}: {r['target'] or '.'}"
                             + (f" [{r['names']}]" if r["names"] else ""))
            if src not in visited:
                visited.add(src)
                frontier.append((src, d + 1))
    if include_clones:
        hashes = [r["body_hash"] for r in db.execute(
            "SELECT DISTINCT body_hash FROM symbols WHERE body_hash != '' AND "
            + (f"id IN ({','.join('?' * len(sym_ids))})" if sym_ids else
               f"file_id IN ({','.join('?' * len(seeds))})"),
            sym_ids or seeds)]
        for h in hashes:
            for r in db.execute("SELECT file_id, qualname, line FROM symbols WHERE body_hash=?", (h,)):
                if r["file_id"] in seeds or r["file_id"] in best:
                    continue
                weight[r["file_id"]] += 0.5
                best[r["file_id"]] = (1, "clone", f"L{r['line']}: {r['qualname']} (identical body)")
    info = _file_info(db, list(best))
    report.impacts = sorted(
        (Impact(info[f][0], info[f][1], d, via, round(weight[f], 3), ev) for f, (d, via, ev) in best.items()
         if f in info),
        key=lambda i: (i.depth, -i.weight, i.repo, i.path))
    return report


def _file_info(db: sqlite3.Connection, ids: list[int]) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        for r in db.execute(
                f"SELECT f.id, r.name, f.path FROM files f JOIN repos r ON r.id=f.repo_id"
                f" WHERE f.id IN ({','.join('?' * len(chunk))})", chunk):
            out[r["id"]] = (r["name"], r["path"])
    return out


# ============================================================ cross-repo coupling


@dataclass
class Coupling:
    repo: str
    imports_from: int = 0          # edges: hub-repo file -> this repo's file
    imports_into: int = 0          # edges: this repo's file -> hub-repo file
    clones: int = 0                # identical function/class bodies shared with hub
    shared_names: int = 0          # same non-generic top-level names defined in both
    hub_files_exposed: int = 0     # hub files transitively affected if this repo changes
    examples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return (3 * self.imports_from + 3 * self.imports_into + 2 * self.clones
                + 0.25 * self.shared_names + self.hub_files_exposed)


def coupling(atlas: "Atlas", hub: str = "kgirl", max_examples: int = 6) -> list[Coupling]:
    db = atlas.db
    hub_id = atlas.repo_id(hub)
    if hub_id is None:
        raise KeyError(f"repo {hub!r} is not indexed")
    out: dict[str, Coupling] = {}

    def get(name: str) -> Coupling:
        return out.setdefault(name, Coupling(name))

    for r in db.execute(
            "SELECT rb.name AS other, fa.path AS a, fb.path AS b, i.target FROM imports i"
            " JOIN files fa ON fa.id=i.file_id JOIN files fb ON fb.id=i.resolved_file_id"
            " JOIN repos rb ON rb.id=fb.repo_id WHERE fa.repo_id=? AND fb.repo_id!=?", (hub_id, hub_id)):
        c = get(r["other"])
        c.imports_from += 1
        c.examples.setdefault("hub imports", []).append(f"{r['a']} -> {r['b']}")
    for r in db.execute(
            "SELECT ra.name AS other, fa.path AS a, fb.path AS b FROM imports i"
            " JOIN files fa ON fa.id=i.file_id JOIN files fb ON fb.id=i.resolved_file_id"
            " JOIN repos ra ON ra.id=fa.repo_id WHERE fb.repo_id=? AND fa.repo_id!=?", (hub_id, hub_id)):
        c = get(r["other"])
        c.imports_into += 1
        c.examples.setdefault("imports hub", []).append(f"{r['a']} -> {r['b']}")
    clone_ids: dict[str, set[int]] = defaultdict(set)
    for r in db.execute(
            "SELECT rb.name AS other, sa.id AS sid, sa.qualname AS q, fa.path AS a, fb.path AS b FROM symbols sa"
            " JOIN files fa ON fa.id=sa.file_id JOIN symbols sb ON sb.body_hash=sa.body_hash AND sb.id!=sa.id"
            " JOIN files fb ON fb.id=sb.file_id JOIN repos rb ON rb.id=fb.repo_id"
            " WHERE sa.body_hash!='' AND fa.repo_id=? AND fb.repo_id!=?", (hub_id, hub_id)):
        c = get(r["other"])
        clone_ids[r["other"]].add(r["sid"])
        c.clones = len(clone_ids[r["other"]])
        c.examples.setdefault("clones", []).append(f"{r['q']}: {hub}:{r['a']} == {r['other']}:{r['b']}")
    for r in db.execute(
            "SELECT rb.name AS other, COUNT(DISTINCT sa.name) AS n, GROUP_CONCAT(DISTINCT sa.name) AS names"
            " FROM symbols sa JOIN files fa ON fa.id=sa.file_id"
            " JOIN symbols sb ON sb.name=sa.name JOIN files fb ON fb.id=sb.file_id JOIN repos rb ON rb.id=fb.repo_id"
            " WHERE fa.repo_id=? AND fb.repo_id!=? AND sa.kind IN ('class','function','struct')"
            " AND sb.kind IN ('class','function','struct') AND length(sa.name) > 5"
            " AND sa.name NOT LIKE '\\_%' ESCAPE '\\' GROUP BY rb.name", (hub_id, hub_id)):
        names = [n for n in (r["names"] or "").split(",") if n.lower() not in _GENERIC_NAMES]
        c = get(r["other"])
        c.shared_names = len(names)
        c.examples["shared names"] = sorted(names)[:max_examples]
    for c in out.values():
        other_id = atlas.repo_id(c.repo)
        if c.imports_from and other_id is not None:
            c.hub_files_exposed = _exposed(db, other_id, hub_id)
        for k in c.examples:
            c.examples[k] = sorted(set(c.examples[k]))[:max_examples]
    return sorted(out.values(), key=lambda c: -c.score)


def _exposed(db: sqlite3.Connection, other_id: int, hub_id: int, max_depth: int = 3) -> int:
    """How many hub files transitively depend on anything in `other_id`."""
    seeds = [r["id"] for r in db.execute("SELECT id FROM files WHERE repo_id=?", (other_id,))]
    seen: set[int] = set(seeds)
    frontier = list(seeds)
    hub_hits: set[int] = set()
    for _ in range(max_depth):
        nxt: list[int] = []
        for i in range(0, len(frontier), 500):
            chunk = frontier[i:i + 500]
            for r in db.execute(
                    f"SELECT i.file_id, f.repo_id FROM imports i JOIN files f ON f.id=i.file_id"
                    f" WHERE i.resolved_file_id IN ({','.join('?' * len(chunk))})", chunk):
                if r["file_id"] not in seen:
                    seen.add(r["file_id"])
                    nxt.append(r["file_id"])
                    if r["repo_id"] == hub_id:
                        hub_hits.add(r["file_id"])
        frontier = nxt
        if not frontier:
            break
    return len(hub_hits)


# ============================================================ architecture card


def architecture_card(atlas: "Atlas", repo: str, top: int = 8) -> dict:
    db = atlas.db
    rid = atlas.repo_id(repo)
    if rid is None:
        raise KeyError(f"repo {repo!r} is not indexed")
    row = db.execute("SELECT * FROM repos WHERE id=?", (rid,)).fetchone()
    langs = {r["lang"]: {"files": r["n"], "loc": r["loc"]} for r in db.execute(
        "SELECT lang, COUNT(*) AS n, SUM(loc) AS loc FROM files WHERE repo_id=? GROUP BY lang ORDER BY loc DESC",
        (rid,))}
    dirs = Counter()
    for r in db.execute("SELECT path, loc FROM files WHERE repo_id=? AND lang!='markdown'", (rid,)):
        parts = PurePosixPath(r["path"]).parts
        dirs["/".join(parts[:2]) if len(parts) > 2 else (parts[0] if len(parts) > 1 else ".")] += r["loc"] or 0
    hubs = [f"{r['path']} (<-{r['n']})" for r in db.execute(
        "SELECT f.path, COUNT(DISTINCT i.file_id) AS n FROM imports i JOIN files f ON f.id=i.resolved_file_id"
        " WHERE f.repo_id=? GROUP BY f.id ORDER BY n DESC, f.path LIMIT ?", (rid, top))]
    entries = [r["path"] for r in db.execute(
        "SELECT path FROM files WHERE repo_id=? AND (entry=1 OR path LIKE '%main.py' OR path LIKE '%/index.ts'"
        " OR path LIKE 'index.%' OR path LIKE '%server.%' OR path LIKE '%cli.py' OR path LIKE '%app.py')"
        " ORDER BY length(path), path LIMIT ?", (rid, top * 2))]
    external = Counter()
    for r in db.execute(
            "SELECT i.target FROM imports i JOIN files f ON f.id=i.file_id"
            " WHERE f.repo_id=? AND i.resolved_file_id IS NULL AND i.level=0 AND i.target!=''", (rid,)):
        t = r["target"]
        root = t.split("/")[0] if not t.startswith("@") else "/".join(t.split("/")[:2])
        root = root.split(".")[0].split("::")[0]
        if root and not root.startswith("."):
            external[root] += 1
    key_symbols = [f"{r['qualname']} — {r['path']}:{r['line']}" for r in db.execute(
        "SELECT s.qualname, f.path, s.line, (SELECT COUNT(*) FROM refs x WHERE x.name=s.name) AS uses"
        " FROM symbols s JOIN files f ON f.id=s.file_id WHERE f.repo_id=? AND s.kind IN ('class','struct')"
        " AND length(s.name) > 3 ORDER BY uses DESC, s.qualname LIMIT ?", (rid, top))]
    readme = db.execute(
        "SELECT f.path, f.doc FROM files f WHERE f.repo_id=? AND lower(f.path) IN ('readme.md','readme')",
        (rid,)).fetchone()
    broken = db.execute("SELECT COUNT(*) FROM files WHERE repo_id=? AND parse_error IS NOT NULL",
                        (rid,)).fetchone()[0]
    return {
        "repo": repo, "root": row["root"], "head": (row["head"] or "")[:12],
        "languages": langs, "top_dirs_by_loc": dict(dirs.most_common(top)),
        "entrypoints": entries, "hub_modules": hubs,
        "external_deps": dict(external.most_common(top * 2)), "key_types": key_symbols,
        "readme": readme["doc"] if readme else "", "unparseable_python_files": broken,
    }


def render_card(card: dict) -> str:
    langs = ", ".join(f"{k} {v['files']}f/{v['loc']}loc" for k, v in card["languages"].items())
    out = [f"## {card['repo']}  @{card['head']}", f"languages: {langs}"]
    if card.get("summary"):
        out.append(f"summary: {card['summary']}")
    elif card.get("readme"):
        out.append(f"readme: {card['readme']}")
    for key, label in (("top_dirs_by_loc", "layout"), ("entrypoints", "entrypoints"),
                       ("hub_modules", "hub modules"), ("key_types", "key types"), ("external_deps", "external")):
        val = card.get(key)
        if val:
            items = [f"{k} ({v})" for k, v in val.items()] if isinstance(val, dict) else val
            out.append(f"{label}: " + "; ".join(items))
    if card.get("unparseable_python_files"):
        out.append(f"warning: {card['unparseable_python_files']} python file(s) do not parse")
    return "\n".join(out)
