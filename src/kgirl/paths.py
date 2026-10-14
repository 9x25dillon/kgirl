"""Where the user's sibling repositories live, so code never hardcodes one machine's home directory.

    find_repo("numbskull")  →  first existing of:
        $NUMBSKULL_PATH                      (env var: name upper-cased, non-alphanumerics → "_", + "_PATH")
        <kgirl>/numbskull                    (clones or symlinks in the repo root; .gitignore lists them)
        <kgirl>/../numbskull                 (siblings of the kgirl checkout)
        ~/numbskull
        /home/kill/numbskull                 (the original author's machine; kept so it keeps working there)
    If none exists, the <kgirl>/../<name> path is returned, so `.exists()` checks stay meaningful.

    ensure_on_path("numbskull") also makes the repo importable: it appends the repo root (and, for
    known repos, the directories their modules live in) to sys.path. Appending, not prepending, keeps
    kgirl's own package-qualified modules first.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]          # <kgirl>/src/kgirl/paths.py → <kgirl>
LEGACY_HOME = Path("/home/kill")

# Subdirectories that hold importable modules, per repository (relative to the repo root).
IMPORT_DIRS: dict[str, tuple[str, ...]] = {
    "numbskull": (".", "packages/wavecaster/src", "packages/emergentnet/src"),
    "NuRea_sim": (".", "entropy engine"),
    "Eopiez": (".", "refactored"),
    "kgirl": (".", "scripts/workflows"),
}
ALIASES = {"nurea_sim": "NuRea_sim", "eopiez": "Eopiez", "limp": "LiMp"}


def _env_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", name).upper() + "_PATH"


def candidates(name: str) -> list[Path]:
    name = ALIASES.get(name.lower(), name) if name.lower() in ALIASES else name
    out: list[Path] = []
    env = os.environ.get(_env_name(name))
    if env:
        out.append(Path(env).expanduser())
    if name == "kgirl":
        out.append(REPO_ROOT)
    out += [REPO_ROOT / name, REPO_ROOT.parent / name, Path.home() / name, LEGACY_HOME / name]
    lower = name.lower()                                  # GitHub names are case-insensitive; clones may differ
    if lower != name:
        out += [REPO_ROOT / lower, REPO_ROOT.parent / lower, Path.home() / lower]
    return out


def find_repo(name: str) -> Path:
    """The first existing location of a sibling repository (see module docstring)."""
    for p in candidates(name):
        if p.is_dir():
            return p
    return REPO_ROOT.parent / name


def ensure_on_path(*names: str) -> dict[str, Optional[Path]]:
    """Append each found repo's import directories to sys.path; returns {name: root or None}."""
    found: dict[str, Optional[Path]] = {}
    for name in names:
        root = find_repo(name)
        if not root.is_dir():
            found[name] = None
            continue
        found[name] = root
        key = ALIASES.get(name.lower(), name)
        for sub in IMPORT_DIRS.get(key, (".",)):
            d = str((root / sub).resolve())
            if Path(d).is_dir() and d not in sys.path:
                sys.path.append(d)
    return found


def describe(names: Iterable[str] = ("numbskull", "NuRea_sim", "Eopiez", "LiMp", "aipyapp")) -> dict[str, dict]:
    return {n: {"path": str(find_repo(n)), "exists": find_repo(n).is_dir(), "env": _env_name(n)} for n in names}
