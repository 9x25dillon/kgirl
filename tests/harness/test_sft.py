"""Decision log on trajectories and the role-specific SFT exporters (soup export-sft --role)."""
import contextlib
import io
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from tests.harness._fixtures import BUGGY_REPO, write
from tests.harness.test_jev import FIX, VERIFY, llm

from kgirl.harness import sft
from kgirl.harness.jev import CODE_OPS, JevConfig, Swarm, SwarmConfig, parse_decision
from kgirl.harness.jev.agent import PROMPT_CAP, SYSTEM, Jev
from kgirl.harness.jev.swarm import SMITH_SYSTEM
from kgirl.harness.soup import Soup
from kgirl.harness.util import sha1

GOAL = "make test_calc pass: mean() is wrong"


class DecisionLogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.repo = write(self.base / "calc", BUGGY_REPO)
        self.soup = Soup(self.base / "soup.db")

    def tearDown(self):
        self.soup.close()
        self.tmp.cleanup()

    def run_swarm(self, origin=None):
        cfg = SwarmConfig(size=3, temperatures=(0.1, 0.5, 0.9), jev=JevConfig(max_steps=6))
        res = Swarm(llm, self.soup, config=cfg).run(GOAL, self.repo, VERIFY, origin=origin)
        self.assertIsNotNone(res.winner, res.render())
        return res

    def record(self):
        (frag,) = self.soup.all(kind="trajectory")
        return frag, json.loads(frag.data)

    def test_winner_record_carries_prompts_decisions_and_smith_calls(self):
        res = self.run_swarm()
        w = res.winner
        frag, rec = self.record()
        self.assertEqual(rec["log_version"], 1)
        self.assertEqual(rec["temperature"], 0.1)
        self.assertEqual(rec["system"], w.system)
        self.assertEqual(rec["system_sha1"], sha1(w.system))
        self.assertTrue(rec["system"].startswith(SYSTEM.split("{grammar}")[0]))
        decs = rec["decisions"]
        self.assertEqual([d["op"] for d in decs], ["OPEN", "EDIT"])   # good_decider's path
        self.assertEqual(len(decs), w.llm_calls)
        for d in decs:
            self.assertTrue(d["prompt"].startswith(f"GOAL: {GOAL}"))
            self.assertTrue(d["prompt"].rstrip().endswith("Your one-line decision:"))
            self.assertTrue(d["valid"] and d["ok"] and not d["truncated"])
        self.assertIn("OPEN calc.py", decs[1]["prompt"])          # the prompt after a step shows its result
        (call,) = rec["smith"]
        self.assertEqual((call["step"], call["applied"], call["reply"]), (decs[1]["n"], True, FIX))
        self.assertEqual(call["temperature"], 0.1)
        self.assertIn("FILE: calc.py", call["prompt"])
        self.assertEqual(rec["smith_system"], SMITH_SYSTEM)

    def test_failed_and_invalid_steps_are_logged_but_not_exported(self):
        cfg = SwarmConfig(size=1, temperatures=(0.5,), jev=JevConfig(max_steps=4, max_invalid=2))
        res = Swarm(llm, self.soup, config=cfg).run(GOAL, self.repo, VERIFY)   # T 0.5 replies "FLY 1"
        self.assertIsNone(res.winner)
        t = res.trajectories[0]
        self.assertEqual([(d["op"], d["valid"]) for d in t.decisions], [("INVALID", False), ("INVALID", False)])
        self.assertEqual(t.decisions[0]["raw"], "FLY 1")
        self.assertEqual(list(sft.rows(self.soup, "jev")), [])           # no winner, no record, nothing to learn

    def test_roles_export_the_format_each_model_is_used_in(self):
        self.run_swarm()
        frag, rec = self.record()
        diff = list(sft.rows(self.soup, "diff"))
        self.assertEqual(len(diff), 1)
        self.assertEqual(set(diff[0]), {"instruction", "context", "output", "meta"})   # kgirl.tauls reads this
        self.assertEqual(diff[0]["output"], rec["diff"])
        jev = list(sft.rows(self.soup, "jev"))
        self.assertEqual([r["completion"].split()[0] for r in jev], ["OPEN", "EDIT"])
        for r in jev:
            self.assertEqual(r["system"], rec["system"])
            self.assertEqual(r["meta"]["fragment"], frag.id)
            n_el = sum(1 for line in r["prompt"].splitlines() if line.startswith("["))
            self.assertEqual(parse_decision(r["completion"], CODE_OPS, n_el).raw, r["completion"])   # a valid line
        (smith,) = sft.rows(self.soup, "smith")
        self.assertEqual((smith["system"], smith["completion"]), (SMITH_SYSTEM, FIX))
        with self.assertRaises(ValueError):
            list(sft.rows(self.soup, "everything"))

    def test_staged_mcp_runs_export_nothing_until_promoted(self):
        self.run_swarm(origin="mcp")
        frag, _ = self.record()
        self.assertEqual(frag.status, "staged")
        for role in sft.ROLES:
            self.assertEqual(list(sft.rows(self.soup, role)), [], role)
        self.soup.set_status(frag.id, "active", "promoted by user")
        self.assertEqual([len(list(sft.rows(self.soup, r))) for r in sft.ROLES], [1, 2, 1])

    def test_truncated_prompts_and_old_records_are_skipped(self):
        long_goal = "x" * (PROMPT_CAP + 10)
        d = Jev.log_decision(1, long_goal, "OPEN 1", "OPEN", True, True, "OPEN 1")
        self.assertTrue(d["truncated"])
        self.assertEqual(len(d["prompt"]), PROMPT_CAP)
        rec = {"goal": "g", "files": [], "diff": "d", "system": "s", "decisions": [d], "smith": [
            {"prompt": "p", "truncated": False, "reply": "r", "reply_truncated": True, "applied": True}]}
        self.soup.add("truncated run", kind="trajectory", source="cli", data=json.dumps(rec))
        self.soup.add("old run", kind="trajectory", source="cli",
                      data=json.dumps({"goal": "g2", "files": ["a.py"], "diff": "d2"}))   # before log_version 1
        self.assertEqual(len(list(sft.rows(self.soup, "diff"))), 2)
        self.assertEqual(list(sft.rows(self.soup, "jev")), [])
        self.assertEqual(list(sft.rows(self.soup, "smith")), [])

    def test_cli_role_flag(self):
        from kgirl.harness.cli import main
        home = self.base / "home"
        home.mkdir()
        self.soup.close()
        self.soup = Soup(home / "soup.db")
        self.run_swarm()
        out = self.base / "jev.jsonl"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(main(["--home", str(home), "soup", "export-sft", str(out), "--role", "jev"]), 0)
        self.assertIn("wrote 2 jev examples", buf.getvalue())
        self.assertEqual(len(out.read_text().splitlines()), 2)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--home", str(home), "soup", "export-sft", str(out)]), 0)
        self.assertEqual(set(json.loads(out.read_text().splitlines()[0])), {"instruction", "context", "output", "meta"})


class ExportFilterTests(unittest.TestCase):
    """Every combination of each filter's flags, on records complete enough to export.

    The records carry every key a row needs (notably `smith_system`), so a step can only stay out because the
    filter rejected it. A KeyError guard around a broken filter does not pass here.
    """

    def exported(self, role, **fields):
        rec = {"goal": "g", "files": [], "diff": "d", "system": "s", "smith_system": "ss",
               "decisions": [], "smith": [], **fields}
        with tempfile.TemporaryDirectory() as tmp:          # a fresh store per case, so no case sees another's record
            soup = Soup(Path(tmp) / "soup.db")
            try:
                soup.add("run", kind="trajectory", source="cli", data=json.dumps(rec))
                return list(sft.rows(soup, role))
            finally:
                soup.close()

    def test_smith_exports_only_applied_untruncated_replies(self):
        for applied, truncated, reply_truncated in itertools.product((True, False), repeat=3):
            flags = {"applied": applied, "truncated": truncated, "reply_truncated": reply_truncated}
            with self.subTest(**flags):
                call = {"prompt": "p", "reply": "r", "step": 1, "path": "a.py", **flags}
                rows = self.exported("smith", smith=[call])
                if applied and not truncated and not reply_truncated:
                    (row,) = rows
                    self.assertEqual((row["system"], row["prompt"], row["completion"]), ("ss", "p", "r"))
                else:
                    self.assertEqual(rows, [])

    def test_jev_exports_only_valid_successful_untruncated_decisions(self):
        for valid, ok, truncated, decision in itertools.product((True, False), (True, False), (True, False),
                                                                ("OPEN 1", "")):
            flags = {"valid": valid, "ok": ok, "truncated": truncated, "decision": decision}
            with self.subTest(**flags):
                step = {"n": 1, "op": "OPEN", "prompt": "p", **flags}
                rows = self.exported("jev", decisions=[step])
                if valid and ok and not truncated and decision:
                    (row,) = rows
                    self.assertEqual((row["system"], row["prompt"], row["completion"]), ("s", "p", "OPEN 1"))
                else:
                    self.assertEqual(rows, [])

    def test_diff_exports_only_records_with_a_diff(self):
        self.assertEqual(len(self.exported("diff")), 1)
        self.assertEqual(self.exported("diff", diff=""), [])


if __name__ == "__main__":
    unittest.main()
