"""Markdown report: architecture of every indexed repo + blast radius into a hub repo."""

from __future__ import annotations

from datetime import datetime, timezone

from .atlas import Atlas, architecture_card, blast_radius, coupling


def atlas_report(atlas: Atlas, hub: str = "kgirl", top_couplings: int = 15, hub_blasts: int = 6,
                 exclude_prefixes: tuple[str, ...] = ("src/kgirl/harness/", "tests/harness/")) -> str:
    repos = atlas.repos()
    total_files = sum(r["nfiles"] for r in repos)
    total_syms = sum(r["nsymbols"] for r in repos)
    out = [f"# Atlas report — {hub} and its neighbours",
           "",
           f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `python -m kgirl.harness report`. "
           f"{len(repos)} repositories, {total_files} files, {total_syms} symbols. "
           "Structure only: signatures, docstrings, imports, clone hashes — no file bodies._",
           "",
           "Edges: **import** = resolved import (cross-repo when module names match); "
           "**clone** = identical normalized function/class body in both repos (same defect, no runtime link); "
           "**shared names** = same non-generic top-level class/function name defined in both.",
           ""]
    if atlas.repo_id(hub) is not None:
        out += [f"## Coupling into `{hub}` (blast radius across repos)", "",
                f"| repo | {hub}→repo imports | repo→{hub} imports | cloned symbols | shared names | "
                f"{hub} files exposed | score |", "|---|---:|---:|---:|---:|---:|---:|"]
        couplings = coupling(atlas, hub)
        for c in couplings[:top_couplings]:
            out.append(f"| {c.repo} | {c.imports_from} | {c.imports_into} | {c.clones} | {c.shared_names} | "
                       f"{c.hub_files_exposed} | {c.score:.0f} |")
        out.append("")
        for c in couplings[: min(8, len(couplings))]:
            ex = "; ".join(f"**{k}**: " + ", ".join(f"`{v}`" for v in vals[:4]) for k, vals in c.examples.items()
                           if vals)
            if ex:
                out.append(f"- **{c.repo}** — {ex}")
        out.append("")
        hubs = atlas.db.execute(
            "SELECT f.path, COUNT(DISTINCT i.file_id) AS n FROM imports i JOIN files f ON f.id=i.resolved_file_id"
            " JOIN repos r ON r.id=f.repo_id WHERE r.name=? GROUP BY f.id ORDER BY n DESC LIMIT ?",
            (hub, hub_blasts + 10)).fetchall()
        hubs = [h for h in hubs if not h["path"].startswith(exclude_prefixes)][:hub_blasts]
        if hubs:
            out += [f"## Highest-blast modules in `{hub}`", ""]
            for h in hubs:
                rep = blast_radius(atlas, f"{hub}:{h['path']}", max_depth=3)
                direct = [i for i in rep.impacts if i.depth == 1 and i.via != "clone"]
                clones = [i for i in rep.impacts if i.via == "clone"]
                out.append(f"### `{h['path']}` — {len(rep.impacts)} dependents "
                           f"({len(direct)} direct, {len(clones)} clone sites)")
                for i in rep.impacts[:8]:
                    out.append(f"- d{i.depth} {i.via} `{i.repo}:{i.path}` — {i.evidence}")
                out.append("")
    out += ["## Repository cards", ""]
    for r in repos:
        card = architecture_card(atlas, r["name"])
        langs = ", ".join(f"{k} {v['files']}f/{v['loc'] or 0}loc" for k, v in card["languages"].items())
        out.append(f"### {r['name']}  `{card['head'] or 'no-git'}`")
        out.append(f"- **languages**: {langs}")
        if card.get("readme"):
            out.append(f"- **readme**: {card['readme'][:240]}")
        if card["entrypoints"]:
            out.append("- **entrypoints**: " + ", ".join(f"`{e}`" for e in card["entrypoints"][:8]))
        if card["hub_modules"]:
            out.append("- **hub modules**: " + ", ".join(f"`{h}`" for h in card["hub_modules"][:6]))
        if card["key_types"]:
            out.append("- **key types**: " + ", ".join(f"`{k}`" for k in card["key_types"][:6]))
        ext = [k for k in card["external_deps"] if k not in _BORING][:10]
        if ext:
            out.append("- **external deps**: " + ", ".join(f"`{e}`" for e in ext))
        if card["unparseable_python_files"]:
            out.append(f"- **warning**: {card['unparseable_python_files']} python file(s) fail to parse")
        out.append("")
    return "\n".join(out)


_BORING = frozenset("typing os sys re json time math dataclasses logging pathlib collections itertools functools "
                    "enum abc asyncio hashlib random __future__ subprocess argparse datetime copy uuid warnings "
                    "threading traceback io string struct base64 shutil tempfile glob inspect contextlib".split())
