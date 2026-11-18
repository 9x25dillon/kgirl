#!/usr/bin/env python3
"""Run the offline engineering checks and retain logs with a compact final summary."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from _support import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended-console", action="store_true", help="require integrated Helix/Holo console routes")
    parser.add_argument("--timeout", type=float, default=180, help="maximum seconds per command (default: 180)")
    args = parser.parse_args()
    if not 0 < args.timeout <= 3600:
        parser.error("--timeout must be greater than zero and at most 3600")
    logs = Path(tempfile.mkdtemp(prefix="kgirl-check-logs-"))
    checks = [("engineering", ["engineering_tests.py"]), ("push-gate", ["hook_tests.py"]),
              ("merge-preview", ["merge_preview_tests.py"]),
              ("land-lanes", ["land_lanes_tests.py"]),
              ("harness+mind", ["suite_check.py", "harness"]),
              ("helix", ["suite_check.py", "helix"]), ("holo", ["suite_check.py", "holo"]),
              ("vessel", ["suite_check.py", "vessel"]),
              ("mind-smoke", ["mind_smoke.py"]),
              ("console-smoke", ["console_smoke.py", *(["--extended"] if args.extended_console else [])])]
    results = []
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    print(f"Python {sys.version.split()[0]}; logs: {logs}", flush=True)
    for label, command in checks:
        print(f"Running {label}...", flush=True)
        started = time.monotonic()
        with (logs / f"{label}.log").open("w", encoding="utf-8") as log:
            try:
                child = subprocess.Popen([sys.executable, str(ROOT / "scripts/smoke" / command[0]), *command[1:]],
                                         cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                try:
                    code = child.wait(timeout=args.timeout)
                    status = "PASS" if code == 0 else f"FAIL ({code})"
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                    # Terminate only the process group created for this check, including fixture children.
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        print(f"Check process {child.pid} did not stop; inspect it before rerunning.", file=sys.stderr)
                        return 2
                    if isinstance(exc, KeyboardInterrupt):
                        print(f"Interrupted; retained logs: {logs}", file=sys.stderr)
                        return 130
                    status = "TIMEOUT"
            except OSError as exc:
                log.write(f"Could not start check: {type(exc).__name__}: {exc}\n")
                status = "ERROR"
        results.append((label, status, time.monotonic() - started))
    print("\nCheck-day summary (mind tests are included in harness):")
    for label, status, elapsed in results:
        print(f"{label:<16} {status:<12} {elapsed:6.1f}s")
    print(f"Logs retained: {logs}")
    print("No install, model download, commit, push or merge was performed.")
    if any(status == "ERROR" for _, status, _ in results):
        return 2
    return 0 if all(status == "PASS" for _, status, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
