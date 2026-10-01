"""Model Context Protocol server (stdio, JSON-RPC 2.0, dependency-free).

Registers the harness as tools inside Claude Code (or any MCP client), so the
coding agent you already use gets repo-structural memory, blast radius and the
Soup pool. See `.mcp.json` at the repository root.

Transport: newline-delimited JSON-RPC messages on stdin/stdout. Logs go to
stderr only (stdout is the protocol channel).
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Callable

from . import __version__
from .assistant import Assistant
from .atlas import architecture_card, blast_radius, coupling, render_card

PROTOCOL_VERSION = "2025-06-18"


def _tools(a: Assistant) -> dict[str, tuple[dict, Callable[[dict], str]]]:
    def search(p):
        hits = a.atlas.search(p["query"], int(p.get("limit", 12)), p.get("repo"))
        return "\n".join(f"{h.cite()} [{h.kind}] {h.signature}" + (f" — {h.doc}" if h.doc else "") for h in hits) \
            or "no matches"

    def outline(p):
        rows = a.atlas.outline(p["repo"], p["path"])
        return "\n".join(f"{r['line']:>5} {r['kind']:<8} {r['signature']}" for r in rows) or "no symbols"

    def source(p):
        start = int(p.get("start", 1))
        return a.atlas.read_source(p["repo"], p["path"], start, int(p.get("end", start + 80)))

    def blast(p):
        return blast_radius(a.atlas, p["target"], int(p.get("depth", 3))).render(int(p.get("limit", 40)))

    def couple(p):
        rows = coupling(a.atlas, p.get("hub", "kgirl"))[: int(p.get("limit", 15))]
        return "\n".join(f"{c.repo}: score={c.score:.0f} imports_from={c.imports_from} imports_into={c.imports_into}"
                         f" clones={c.clones} shared_names={c.shared_names} exposed={c.hub_files_exposed}"
                         for c in rows) or "no coupling"

    def card(p):
        return render_card(architecture_card(a.atlas, p["repo"]))

    def recall(p):
        rec = a.soup.recall(p["query"], int(p.get("budget", 600)), scope=p.get("scope"))
        return rec.render() or "nothing recalled"

    def remember(p):
        fid = a.remember(p["text"], p.get("kind", "note"), p.get("tags", ""), p.get("scope", "*"))
        return f"stored fragment #{fid}"

    def ask(p):
        ans = a.ask(p["question"], p.get("repo"))
        return ans.text + ("\n\nsources: " + ", ".join(ans.citations[:12]) if ans.citations else "")

    def task(p):
        verify = p.get("verify")
        res = a.task(p["goal"], p["repo_path"], verify.split() if isinstance(verify, str) else verify,
                     size=int(p.get("size", 3)), apply=bool(p.get("apply", False)))
        return res.render()

    def ecl_quantify(p):
        from ..nihiline import calibration as cal
        mode = p.get("mode", "conc")
        rows = [cal.quantify(m, float(p[k]), mode) for m, k in (("CEA", "cea"), ("AFP", "afp")) if p.get(k) is not None]
        if not rows:
            raise ValueError("give cea and/or afp ECL intensities")
        return "\n".join(f"{q.marker}: {q.signal:.1f} a.u. → {q.value:.4g} {q.unit} [{q.flag}] (scar {q.scar} a.u.)"
                         for q in rows)

    def ecl_protocol(p):
        from ..nihiline.protocol import run_protocol
        return run_protocol({"CEA": float(p.get("cea_cells_per_ml", 1e4)), "AFP": float(p.get("afp_cells_per_ml", 1e4))},
                            tpa_mM=float(p.get("tpa_mM", 50.0)), f_nihil=p.get("f_nihil")).render()

    def ecl_gap(p):
        from ..nihiline.amplification import chain
        return "\n\n".join(chain(m).render() for m in ([p["marker"]] if p.get("marker") else ["CEA", "AFP"]))

    S = lambda props, req: {"type": "object", "properties": props, "required": req}  # noqa: E731
    s, i, n = {"type": "string"}, {"type": "integer"}, {"type": "number"}
    return {
        "atlas_search": (S({"query": s, "repo": s, "limit": i}, ["query"]), search),
        "atlas_outline": (S({"repo": s, "path": s}, ["repo", "path"]), outline),
        "atlas_source": (S({"repo": s, "path": s, "start": i, "end": i}, ["repo", "path"]), source),
        "atlas_blast_radius": (S({"target": s, "depth": i, "limit": i}, ["target"]), blast),
        "atlas_coupling": (S({"hub": s, "limit": i}, []), couple),
        "atlas_card": (S({"repo": s}, ["repo"]), card),
        "soup_recall": (S({"query": s, "budget": i, "scope": s}, ["query"]), recall),
        "soup_remember": (S({"text": s, "kind": s, "tags": s, "scope": s}, ["text"]), remember),
        "kgirl_ask": (S({"question": s, "repo": s}, ["question"]), ask),
        "jev_swarm_task": (S({"goal": s, "repo_path": s, "verify": s, "size": i, "apply": {"type": "boolean"}},
                             ["goal", "repo_path"]), task),
        "ecl_quantify": (S({"cea": n, "afp": n, "mode": {"type": "string", "enum": ["conc", "cells"]}}, []),
                         ecl_quantify),
        "ecl_protocol": (S({"cea_cells_per_ml": n, "afp_cells_per_ml": n, "tpa_mM": n, "f_nihil": n}, []),
                         ecl_protocol),
        "ecl_gap": (S({"marker": {"type": "string", "enum": ["CEA", "AFP"]}}, []), ecl_gap),
    }


_DESCRIPTIONS = {
    "atlas_search": "Search every indexed repo's symbols (functions/classes/sections) by words; returns repo:path:line.",
    "atlas_outline": "List the symbols (with signatures) defined in one file of an indexed repo.",
    "atlas_source": "Read numbered source lines of an indexed file.",
    "atlas_blast_radius": "Reverse-dependency closure of a file or symbol across all repos (imports, symbol refs, "
                          "clones). Target: 'repo:path', 'path', 'path::Qual.name' or 'Name'.",
    "atlas_coupling": "Rank repos by coupling to a hub repo (imports both ways, cloned code, shared names).",
    "atlas_card": "Architecture card of a repo: languages, layout, entrypoints, hub modules, key types, deps.",
    "soup_recall": "Just-in-time recall from the shared memory pool within a token budget.",
    "soup_remember": "Store a fact/preference/note in the shared memory pool.",
    "kgirl_ask": "Answer a question about the user's repos from a cited, token-bounded context pack.",
    "jev_swarm_task": "Run a swarm of sandboxed Jev agents on a coding goal; returns the verified diff (optionally "
                      "applies it). Requires local Ollama models.",
    "ecl_quantify": "CN121933729A c-BPE-ECL: convert CEA/AFP anode ECL intensities (a.u.) to ng/mL (mode=conc) or "
                    "MCF-7 cells/mL (mode=cells) with the patent's calibration curves; flags reads below the LOD scar.",
    "ecl_protocol": "Run the dual-marker c-BPE-ECL assay end to end (nine levels: probe synthesis → capture → "
                    "charge balance → pulsed drive → scar on zero → quantification) for given cell densities.",
    "ecl_gap": "Per-cell amplification chain (antigen → probes → MB → e⁻ → photons → counts) and predicted vs "
               "reported LOD for CEA/AFP.",
}


class MCPServer:
    def __init__(self, assistant: Assistant):
        self.a = assistant
        self.tools = _tools(assistant)

    def handle(self, msg: dict) -> dict | None:
        mid, method, params = msg.get("id"), msg.get("method", ""), msg.get("params") or {}
        if mid is None:  # notification (e.g. notifications/initialized)
            return None
        try:
            if method == "initialize":
                result: Any = {"protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                               "capabilities": {"tools": {"listChanged": False}},
                               "serverInfo": {"name": "kgirl-harness", "version": __version__}}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": [{"name": n, "description": _DESCRIPTIONS[n], "inputSchema": schema}
                                    for n, (schema, _) in self.tools.items()]}
            elif method == "tools/call":
                name = params.get("name")
                if name not in self.tools:
                    return _err(mid, -32602, f"unknown tool {name!r}")
                try:
                    text = self.tools[name][1](params.get("arguments") or {})
                    result = {"content": [{"type": "text", "text": text}], "isError": False}
                except Exception as exc:  # tool failures are results, not protocol errors
                    result = {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True}
            else:
                return _err(mid, -32601, f"method not found: {method}")
        except Exception as exc:  # pragma: no cover - defensive
            traceback.print_exc(file=sys.stderr)
            return _err(mid, -32603, str(exc))
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def serve(self, stdin=None, stdout=None) -> None:
        stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                resp = _err(None, -32700, "parse error")
            else:
                resp = self.handle(msg)
            if resp is not None:
                stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                stdout.flush()


def _err(mid, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}
