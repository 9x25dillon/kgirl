import re
import sys
import tempfile
import unittest
from pathlib import Path

from tests.harness._fixtures import BUGGY_REPO, write

from kgirl.harness.jev import (CODE_OPS, CodeEnvironment, InvalidDecision, Jev, JevConfig, Swarm, SwarmConfig,
                               parse_decision)
from kgirl.harness.soup import Soup

VERIFY = [sys.executable, "-m", "unittest", "-q", "test_calc"]
FIX = "<<<<<<< SEARCH\n    return sum(xs) / (len(xs) + 1)\n=======\n    return sum(xs) / len(xs)\n>>>>>>> REPLACE"


def row(prompt: str, needle: str, kind: str | None = None, prefix: bool = False) -> int:
    for m in re.finditer(r"^\[(\d+)\] (\w+)\s+(.*)$", prompt, re.M):
        label = m.group(3)
        if (label.startswith(needle) if prefix else needle in label) and (kind is None or m.group(2) == kind):
            return int(m.group(1))
    raise AssertionError(f"{needle!r} not in table:\n{prompt}")


def good_decider(system: str, prompt: str) -> str:
    """A competent Jev: open -> edit -> run -> done, driven only by what it sees."""
    if "RUN exit=0" in prompt:
        return "DONE :: fixed the denominator in mean()"
    if "EDIT applied" in prompt:
        return "RUN"
    if "OPEN calc.py" in prompt:
        return f"EDIT {row(prompt, 'def mean', 'symbol')}"
    return f"OPEN {row(prompt, 'calc.py', prefix=True)}"


def llm(role, system, prompt, max_tokens, temperature):
    if role == "smith":
        return FIX
    if role == "scout":
        return "Fix averaging bugs in calc.py by checking the mean() denominator.\nVerify with `python -m unittest test_calc`."
    if temperature >= 0.9:
        return "BLOCKED :: this Meeseeks gives up"
    if 0.4 <= temperature < 0.9:
        return "FLY 1"
    return good_decider(system, prompt)


class DecisionTests(unittest.TestCase):
    def test_parse_valid_forms(self):
        self.assertEqual(parse_decision("OPEN 3", CODE_OPS, 5).target, 3)
        self.assertEqual(parse_decision("`EDIT [2]`", CODE_OPS, 5).target, 2)
        d = parse_decision("search :: tokenizer padding\nextra chatter", CODE_OPS, 0)
        self.assertEqual((d.op, d.arg), ("SEARCH", "tokenizer padding"))
        self.assertEqual(parse_decision("RUN", CODE_OPS, 0).op, "RUN")

    def test_parse_rejects_out_of_contract(self):
        for bad, n in [("OPEN 9", 3), ("OPEN", 3), ("rm -rf / 1", 3), ("DONE", 3), ("", 3), ("EXEC 1 :: ls", 3)]:
            with self.assertRaises(InvalidDecision, msg=bad):
                parse_decision(bad, CODE_OPS, n)


class CodeEnvTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = write(Path(self.tmp.name) / "calc", BUGGY_REPO)

    def tearDown(self):
        self.tmp.cleanup()

    def test_edit_must_match_exactly_once_and_stay_inside_workspace(self):
        (self.repo / "dup.py").write_text("x = 1\nx = 1\n")
        env = CodeEnvironment(self.repo, "change x in dup", smith=lambda *a: "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE")
        idx = next(i for i, e in enumerate(env.elements) if e.ref.get("path") == "dup.py")
        res = env.act(parse_decision(f"EDIT {idx}", CODE_OPS, len(env.elements)))
        self.assertFalse(res.ok)
        self.assertIn("2x", res.note)
        self.assertEqual((self.repo / "dup.py").read_text(), "x = 1\nx = 1\n")
        self.assertIsNone(env._safe("../outside.py"))

    def test_single_jev_fixes_bug(self):
        env = CodeEnvironment(self.repo, "make test_calc pass: mean() is wrong", smith=lambda *a: FIX,
                              verify_cmd=VERIFY)
        traj = Jev(env, good_decider, config=JevConfig(max_steps=6)).run(env.goal)
        self.assertEqual(traj.outcome, "done", [s.note[:80] for s in traj.steps])
        self.assertEqual([s.op for s in traj.steps], ["OPEN", "EDIT", "RUN", "DONE"])
        self.assertEqual(env.verify()[0], 0)
        self.assertIn("+    return sum(xs) / len(xs)", env.diff())

    def test_invalid_decisions_end_the_run(self):
        env = CodeEnvironment(self.repo, "x")
        traj = Jev(env, lambda s, p: "DANCE 1", config=JevConfig(max_invalid=2)).run("x")
        self.assertEqual(traj.outcome, "invalid")
        self.assertEqual(len(traj.steps), 2)


class SwarmTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.repo = write(self.base / "calc", BUGGY_REPO)
        self.soup = Soup(self.base / "soup.db")
        self.events = []

    def tearDown(self):
        self.soup.close()
        self.tmp.cleanup()

    def swarm(self, **kw):
        cfg = SwarmConfig(size=3, temperatures=(0.1, 0.5, 0.9), jev=JevConfig(max_steps=6), **kw)
        return Swarm(llm, self.soup, config=cfg, emit=lambda t, p: self.events.append(t))

    def test_accepts_verified_winner_learns_and_applies(self):
        goal = "make test_calc pass: mean() is wrong"
        res = self.swarm().run(goal, self.repo, VERIFY, apply=True)
        self.assertIsNotNone(res.winner, res.render())
        self.assertEqual(res.winner.variant["temperature"], 0.1)
        self.assertTrue(res.winner.verified)
        losers = {t.outcome for t in res.trajectories if t is not res.winner}
        self.assertTrue(losers <= {"blocked", "invalid", "cancelled"}, losers)
        self.assertEqual(res.applied, ["calc.py"])
        self.assertIn("/ len(xs)", (self.repo / "calc.py").read_text())
        self.assertIn("swarm.end", self.events)
        kinds = {f.kind: f.status for f in self.soup.all()}
        self.assertEqual(kinds, {"trajectory": "active", "abstraction": "staged"})
        # workspaces are cleaned up
        self.assertFalse(Path(res.winner.workspace).exists())

    def test_repeated_success_promotes_abstraction(self):
        goal = "make test_calc pass: mean() is wrong"
        for _ in range(2):
            write(self.base / "calc", BUGGY_REPO)
            self.swarm().run(goal, self.repo, VERIFY)
        active = self.soup.all(status="active", kind="abstraction")
        self.assertTrue(active, self.soup.stats())
        rec = self.soup.recall("mean() denominator in calc.py", scope="calc")
        self.assertTrue(any(f.kind == "abstraction" for f in rec.fragments))

    def test_unverified_done_is_rejected_by_default(self):
        res = self.swarm().run("make test_calc pass", self.repo, verify_cmd=None)
        self.assertIsNone(res.winner)
        self.assertIn("(len(xs) + 1)", (self.repo / "calc.py").read_text())

    def test_apply_refuses_when_repo_changed(self):
        sw = self.swarm(stop_on_first_success=True)
        res = sw.run("make test_calc pass", self.repo, VERIFY)
        (self.repo / "calc.py").write_text("# edited by the human meanwhile\n")
        sw._apply(res, self.repo)
        self.assertEqual(res.conflicts, ["calc.py"])
        self.assertEqual((self.repo / "calc.py").read_text(), "# edited by the human meanwhile\n")


if __name__ == "__main__":
    unittest.main()
