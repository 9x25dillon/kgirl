#!/usr/bin/env python3
"""Run one offline unittest family, failing on empty or incomplete discovery."""

from __future__ import annotations

import argparse
import os
import unittest

from _support import ROOT, isolated_home


def cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from cases(item)
        else:
            yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("harness", "mind", "helix", "holo", "vessel"))
    args = parser.parse_args()
    relative = "tests/harness/mind" if args.suite == "mind" else f"tests/{args.suite}"
    os.chdir(ROOT)
    with isolated_home():
        loader = unittest.TestLoader()
        suite = loader.discover(str(ROOT / relative), top_level_dir=str(ROOT))
        found = list(cases(suite))
        if loader.errors or not found:
            for error in loader.errors:
                print(error)
            print(f"FAIL {args.suite}: discovery errors or no tests found")
            return 1
        mind = [case for case in found if case.id().startswith("tests.harness.mind.")]
        if args.suite == "harness" and not mind:
            print("FAIL harness: tests/harness/mind was not discovered; check its __init__.py")
            return 1
        # unittest silently skips a folder without __init__.py, so a new lane's tests could vanish from CI.
        # Every folder holding test_*.py files must contribute at least one discovered case.
        ids = {case.id() for case in found}
        for folder in sorted({p.parent for p in (ROOT / relative).rglob("test_*.py")}):
            if "__pycache__" in folder.parts:
                continue
            dotted = ".".join(folder.relative_to(ROOT).parts) + "."
            if not any(i.startswith(dotted) for i in ids):
                print(f"FAIL {args.suite}: {folder.relative_to(ROOT)} has test files but none were discovered; "
                      "check its __init__.py")
                return 1
        print(f"Discovered {len(found)} {args.suite} tests ({len(mind)} mind tests)", flush=True)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
