"""Supervised fine-tuning examples from Soup, one exporter per role, in the format that role is used in.

    python -m kgirl.harness soup export-sft out.jsonl [--role diff|jev|smith]

    diff    goal + files → the winner's diff              {"instruction", "context", "output", "meta"}
            (the original export; kgirl.tauls --sft reads this format)
    jev     the exact prompt of each model-made step → the one-line decision it led to
            {"system", "prompt", "completion", "meta"}; only steps that parsed and succeeded
    smith   the exact EDIT prompt → the SEARCH/REPLACE reply that applied
            {"system", "prompt", "completion", "meta"}; only replies the environment applied

Only active trajectories are exported: staged ones (for example from MCP-started swarms) never train until
a person promotes them. Examples whose prompt or reply was truncated when logged are skipped, because a
truncated prompt is not the input the model actually saw. Trajectories stored before the decision log
existed (no `log_version`) yield diff examples only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

ROLES = ("diff", "jev", "smith")


def _records(soup) -> Iterator[tuple[int, dict]]:
    for f in soup.all(status="active", kind="trajectory"):          # staged (e.g. MCP) runs never train
        if not f.data:
            continue
        try:
            yield f.id, json.loads(f.data)
        except ValueError:
            continue


def rows(soup, role: str = "diff") -> Iterator[dict]:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    for fid, rec in _records(soup):
        if role == "diff":
            if rec.get("diff"):
                yield {"instruction": rec["goal"], "context": ", ".join(rec.get("files", [])),
                       "output": rec["diff"], "meta": {"ops": rec.get("ops"), "fragment": fid}}
        elif role == "jev":
            for d in rec.get("decisions", []):
                if d.get("valid") and d.get("ok") and not d.get("truncated") and d.get("decision"):
                    yield {"system": rec["system"], "prompt": d["prompt"], "completion": d["decision"],
                           "meta": {"fragment": fid, "step": d["n"], "op": d["op"], "goal": rec["goal"],
                                    "temperature": rec.get("temperature"), "system_sha1": rec.get("system_sha1")}}
        else:
            for c in rec.get("smith", []):
                if c.get("applied") and not c.get("truncated") and not c.get("reply_truncated"):
                    yield {"system": rec["smith_system"], "prompt": c["prompt"], "completion": c["reply"],
                           "meta": {"fragment": fid, "step": c.get("step"), "path": c.get("path"),
                                    "goal": rec["goal"], "temperature": c.get("temperature"),
                                    "system_sha1": rec.get("smith_system_sha1")}}


def export(soup, path: str | Path, role: str = "diff") -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows(soup, role):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n
