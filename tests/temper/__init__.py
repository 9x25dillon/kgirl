"""Tests for TEMPER: the PreToolUse guard (.claude/hooks/temper_guard.py) and the model-swap gate (kgirl.temper)."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
