"""The console's _build() explain/summary feature detection (no server, no builder, no model).

On the base stack a BuildResult has neither explain() nor summary_dict(); the builder lane adds them.
_build() must pass them through when present and degrade to "" / None when absent, non-callable or raising.
Written, not run.
"""

import unittest
from dataclasses import dataclass, field

import tests.harness.console  # noqa: F401  (sys.path)
from kgirl.harness.console.server import _build


@dataclass
class OldResult:
    code: str = "def f():\n    return 1\n"
    chosen: int = 0
    confidence: float = 0.81234


@dataclass
class NewResult(OldResult):
    def explain(self) -> str:
        return "=== CANDIDATE VOTES ===\n  #0 passes 2"

    def summary_dict(self) -> dict:
        return {"chosen": self.chosen,
                "candidates": [{"idx": 0, "passed": ["test_gen0:T.test_a"], "failed": {}, "ok": True}]}


@dataclass
class BrokenExplain(OldResult):
    def explain(self) -> str:
        raise RuntimeError("old internal shape")

    def summary_dict(self) -> dict:
        return {"candidates": []}


@dataclass
class BrokenSummary(OldResult):
    def explain(self) -> str:
        return "why"

    def summary_dict(self) -> dict:
        raise RuntimeError("old internal shape")


@dataclass
class NotCallable(OldResult):
    explain: str = "a field, not a method"
    summary_dict: list = field(default_factory=list)


class BuildExplain(unittest.TestCase):
    def test_old_result_degrades_to_empty(self):
        out = _build(OldResult())
        self.assertEqual(out["explain"], "")
        self.assertIsNone(out["summary"])
        self.assertEqual(out["confidence"], 0.812)
        self.assertEqual(out["chosen"], 0)
        self.assertIn("def f():", out["code"])

    def test_new_result_passes_both_through(self):
        out = _build(NewResult())
        self.assertIn("CANDIDATE VOTES", out["explain"])
        self.assertEqual(out["summary"]["candidates"][0]["passed"], ["test_gen0:T.test_a"])
        self.assertEqual(out["summary"]["chosen"], 0)

    def test_explain_raising_keeps_summary(self):
        out = _build(BrokenExplain())
        self.assertEqual(out["explain"], "")
        self.assertEqual(out["summary"], {"candidates": []})

    def test_summary_raising_keeps_explain(self):
        out = _build(BrokenSummary())
        self.assertEqual(out["explain"], "why")
        self.assertIsNone(out["summary"])

    def test_non_callable_attributes_degrade(self):
        out = _build(NotCallable())
        self.assertEqual(out["explain"], "")
        self.assertIsNone(out["summary"])


if __name__ == "__main__":
    unittest.main()
