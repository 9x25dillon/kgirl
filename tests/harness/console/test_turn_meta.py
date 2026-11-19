"""Round-2 console core helpers, offline and without other lanes: the draft footer split, answer meta
(rules + budget + related) with malformed input ignored, the chip lines, and feature detection for drafts,
context and doctor (absent → available False; present → called in-process). Written, not run.
"""

import sys
import types
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import tests.harness.console  # noqa: F401  (sys.path)
from kgirl.harness.console.core import (Console, answer_meta, audit_line, budget_line, related_line,
                                        split_draft_footer)

FOOTER = "# Notes\n\nBody.\n(saved as draft #12; `kgirl write --resume 12` continues one)"


class DraftFooter(unittest.TestCase):
    def test_footer_is_split_off(self):
        body, draft = split_draft_footer(FOOTER)
        self.assertEqual(draft, 12)
        self.assertEqual(body, "# Notes\n\nBody.\n")

    def test_no_footer_leaves_text_alone(self):
        self.assertEqual(split_draft_footer("plain text"), ("plain text", None))

    def test_footer_text_inside_the_body_is_not_split(self):
        text = "see (saved as draft #3) above\nand more"
        self.assertEqual(split_draft_footer(text), (text, None))


class AnswerMeta(unittest.TestCase):
    def test_all_parts(self):
        turn = SimpleNamespace(meta={"rules_hits": ["push"], "rules_left": [], "rules_revised": True,
                                     "budget": {"limit": 10, "used": 9}, "related": [{"conv": 2, "via": "words"}, "junk"]})
        m = answer_meta(turn)
        self.assertEqual(m["rules_hits"], ["push"])
        self.assertEqual(m["budget"], {"limit": 10, "used": 9})
        self.assertEqual(m["related"], [{"conv": 2, "via": "words"}])

    def test_older_turns_have_none(self):
        m = answer_meta(SimpleNamespace(meta={}))
        self.assertEqual((m["budget"], m["related"]), (None, []))
        self.assertEqual(answer_meta(SimpleNamespace(meta={"budget": "x", "related": "y"}))["related"], [])

    def test_lines(self):
        self.assertEqual(budget_line(None), "")
        self.assertEqual(budget_line({"limit": 4000, "used": 3900, "dropped": {"memory": {}, "earlier": {}},
                                      "overflow": 5}), "context 3900/4000 · cut earlier, memory · over by 5")
        self.assertEqual(related_line({"conv": 3, "via": "tags", "tags": ["a", "b"]}), "conv #3 (same topic: a, b)")
        self.assertEqual(related_line({"conv": 4, "via": "words"}), "conv #4 (matching words)")
        self.assertEqual(audit_line({"action": "run", "target": "ls", "outcome": "ok"}), "run ls → ok")


class FeatureDetection(unittest.TestCase):
    def test_drafts_absent_and_present(self):
        bare = SimpleNamespace(ops=lambda: SimpleNamespace())
        self.assertEqual(Console.drafts(bare), {"available": False, "drafts": []})
        mind = SimpleNamespace(drafts=lambda conv=None: [(4, 7, "notes", "drafting: 1/3 sections")])
        d = Console.drafts(SimpleNamespace(ops=lambda: mind))
        self.assertEqual(d["drafts"], [{"id": 4, "conv_id": 7, "title": "notes", "progress": "drafting: 1/3 sections"}])

    def test_resume_without_drafts_is_refused(self):
        with self.assertRaises(ValueError):
            Console.resume_draft_job(SimpleNamespace(ops=lambda: SimpleNamespace()), 4)

    def test_doctor_absent(self):
        with mock.patch.dict(sys.modules, {"kgirl.harness.doctor": None}):
            self.assertEqual(Console.doctor(SimpleNamespace(a=SimpleNamespace(home="/nonexistent"))),
                             {"available": False, "checks": []})

    def test_doctor_present_is_called_in_process(self):
        @dataclass(frozen=True)
        class Check:
            name: str
            status: str
            detail: str
            fix: str = ""

        stub = types.ModuleType("kgirl.harness.doctor")
        seen = []
        stub.collect = lambda home: seen.append(home) or [Check("ollama", "OK", "up")]
        with mock.patch.dict(sys.modules, {"kgirl.harness.doctor": stub}):
            d = Console.doctor(SimpleNamespace(a=SimpleNamespace(home="/h")))
        self.assertEqual(seen, ["/h"])
        self.assertEqual(d, {"available": True, "checks": [{"name": "ollama", "status": "OK", "detail": "up", "fix": ""}]})


if __name__ == "__main__":
    unittest.main()
