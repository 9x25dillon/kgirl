"""Read-only health report for the local kgirl harness."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from .util import harness_home

Runner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


def _check(name: str, ok: bool, detail: str, fix: str, missing: str = "FAIL") -> Check:
    return Check(name, "OK" if ok else missing, detail, "" if ok else fix)


def _run(runner: Runner, argv: list[str], timeout: float = 3.0) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        return runner(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(argv, 127, "", str(exc))


def _ollama(checks: list[Check], opener: Callable, timeout: float) -> None:
    raw = os.environ.get("OLLAMA_HOST", "").strip() or "http://127.0.0.1:11434"
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urllib.parse.urlparse(raw)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.scheme not in {"http", "https"}:
        checks.append(Check("ollama", "FAIL", f"refusing non-loopback OLLAMA_HOST {raw!r}",
                            "set OLLAMA_HOST to http://127.0.0.1:11434"))
        return
    try:
        with opener(raw.rstrip("/") + "/api/tags", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        installed = {m.get("name", "") for m in payload.get("models", []) if isinstance(m, dict)}
    except (OSError, ValueError, urllib.error.URLError) as exc:
        checks.append(Check("ollama", "FAIL", f"not responding on loopback ({exc})", "start Ollama: ollama serve"))
        return
    checks.append(Check("ollama", "OK", f"up on {parsed.hostname}; {len(installed)} model(s) installed"))
    defaults = ("qwen2.5-coder:7b", "qwen2.5-coder:1.5b")
    wanted = {os.environ.get("KGIRL_SMITH_MODEL", defaults[0]), os.environ.get("KGIRL_JEV_MODEL", defaults[1])}
    wanted.add(os.environ.get("KGIRL_BUILD_MODEL", os.environ.get("KGIRL_SMITH_MODEL", defaults[0])))
    wanted.add(os.environ.get("KGIRL_CHAT_MODEL", os.environ.get("KGIRL_BUILD_MODEL",
                                                                  os.environ.get("KGIRL_SMITH_MODEL", defaults[0]))))
    absent = sorted(m for m in wanted if m not in installed and m + ":latest" not in installed)
    checks.append(_check("router models", not absent,
                         "all configured local models are present" if not absent else "missing " + ", ".join(absent),
                         "pull each missing model with ollama pull MODEL"))


def collect(home: str | Path | None = None, *, runner: Runner = subprocess.run,
            opener: Callable = urllib.request.urlopen, now: float | None = None) -> list[Check]:
    """Collect health checks without creating, changing or locking any user file."""
    checks: list[Check] = []
    root = Path(home).expanduser() if home else harness_home()
    current = time.time() if now is None else now
    _ollama(checks, opener, 2.0)

    gpu = _run(runner, ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
    try:
        temperatures = [int(line.strip()) for line in gpu.stdout.splitlines() if line.strip()]
    except ValueError:
        temperatures = []
    if temperatures:
        peak = max(temperatures)
        checks.append(Check("GPU temperature", "WARN" if peak >= 95 else "OK", f"peak {peak} °C",
                            "stop GPU work and let the card cool" if peak >= 95 else ""))
    else:
        checks.append(Check("GPU temperature", "WARN", "temperature unavailable", "check nvidia-smi"))

    index = Path(os.environ.get("KGIRL_CHUNKS_DB", root / "chunks" / "all" / "all-repos.db"))
    if index.is_file():
        age = max(0, current - index.stat().st_mtime)
        checks.append(Check("chunk index", "OK", f"{index} ({age / 86400:.1f} days old)"))
    else:
        checks.append(Check("chunk index", "FAIL", f"missing: {index}", "run kgirl chunks when you intend to rebuild"))
    marker = index.parent / "router-synced"
    synced = index.is_file() and marker.is_file() and marker.stat().st_mtime > index.stat().st_mtime
    checks.append(_check("router sync", synced,
                         "marker is newer than the chunk index" if synced else f"stale or missing: {marker}",
                         "after copying the index to the router, touch the router-synced marker", missing="WARN"))

    code_raw = os.environ.get("KGIRL_CODE", "").strip()
    code = Path(code_raw).expanduser() if code_raw else None
    if code and code.is_dir():
        head = _run(runner, ["git", "-C", str(code), "rev-parse", "--short", "HEAD"])
        ok = head.returncode == 0 and bool(head.stdout.strip())
        checks.append(_check("KGIRL_CODE", ok, f"{code} @ {head.stdout.strip()}" if ok else f"{code} (HEAD unreadable)",
                             "point KGIRL_CODE at a git worktree"))
    else:
        checks.append(Check("KGIRL_CODE", "FAIL", f"not set or not a directory: {code_raw or '(unset)'}",
                            "export KGIRL_CODE=/path/to/your/kgirl/worktree"))

    registered = False
    tried: list[str] = []
    for argv in (["codex", "mcp", "get", "kgirl"], ["claude", "mcp", "get", "kgirl"]):
        proc = _run(runner, argv)
        tried.append(argv[0])
        if proc.returncode == 0:
            registered = True
            break
    checks.append(_check("kgirl MCP", registered, "registered" if registered else "not found via " + "/".join(tried),
                         "register `python3 -m kgirl.harness mcp` in your coding client"))

    venv_raw = os.environ.get("KGIRL_VENV", "").strip()
    python = Path(venv_raw).expanduser() / "bin" / "python3" if venv_raw else None
    pytest = _run(runner, [str(python), "-c", "import pytest"]) if python and python.is_file() else None
    ok = pytest is not None and pytest.returncode == 0
    checks.append(_check("pytest", ok, f"importable in {venv_raw}" if ok else f"not importable in {venv_raw or '(unset)'}",
                         "set KGIRL_VENV to a venv that contains pytest"))
    return checks


def run(home: str | Path | None = None, json_output: bool = False) -> int:
    checks = collect(home)
    if json_output:
        print(json.dumps({"checks": [asdict(c) for c in checks]}, indent=2))
    else:
        for item in checks:
            suffix = f" — fix: {item.fix}" if item.fix else ""
            print(f"{item.status:<4} {item.name}: {item.detail}{suffix}")
    return 1 if any(c.status == "FAIL" for c in checks) else 0
