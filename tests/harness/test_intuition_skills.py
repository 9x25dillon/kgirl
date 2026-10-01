import json
import tempfile
import unittest
from pathlib import Path

from tests.harness._fixtures import BUGGY_REPO, write
from tests.harness.test_jev import VERIFY, good_decider, llm

from kgirl.harness.jev import CODE_OPS, CodeEnvironment, Jev, JevConfig, Swarm, SwarmConfig
from kgirl.harness.jev.env import Element
from kgirl.harness.jev.intuition import Intuition, Routine, RoutineStep, label_key
from kgirl.harness.skills import export, forge
from kgirl.harness.soup import Soup

GOAL = "make test_calc pass: mean() is wrong"
NEEDS = {k for k, v in CODE_OPS.items() if v.needs_target}


class LabelKeyTests(unittest.TestCase):
    def test_keys_survive_line_motion_and_metadata(self):
        self.assertEqual(label_key("calc.py:1-3  def mean(xs) — Arithmetic mean"), "calc.py def mean(xs)")
        self.assertEqual(label_key("calc.py:9-14  def mean(xs)"), "calc.py def mean(xs)")
        self.assertEqual(label_key("calc.py  (python, 3 lines) — doc"), "calc.py")
        self.assertEqual(label_key("calc.py  (4 lines)"), "calc.py")
        self.assertEqual(label_key("calc.py  (from memory)"), "calc.py")


class IntuitionTests(unittest.TestCase):
    def routine(self, support=2, first="calc.py"):
        return Routine(GOAL, [RoutineStep("OPEN", first), RoutineStep("EDIT", "calc.py def mean(xs)"),
                              RoutineStep("RUN"), RoutineStep("DONE", "", "fixed mean")], support, 0.5, 7)

    def test_propose_resolves_targets_in_the_current_table(self):
        intu = Intuition([self.routine()])
        table = [Element("file", "test_calc.py  (python, 9 lines)"), Element("file", "calc.py  (python, 3 lines)")]
        p = intu.propose(GOAL, [], table, NEEDS)
        self.assertEqual(p.line, "OPEN 1")
        self.assertGreater(p.confidence, 0.6)
        table2 = [Element("file", "calc.py  (4 lines)"), Element("symbol", "calc.py:1-3  def mean(xs)")]
        self.assertEqual(intu.propose(GOAL, [("OPEN", "calc.py")], table2, NEEDS).line, "EDIT 1")
        done = [("OPEN", "calc.py"), ("EDIT", "calc.py def mean(xs)"), ("RUN", "")]
        self.assertEqual(intu.propose(GOAL, done, table2, NEEDS).line, "DONE :: fixed mean")

    def test_no_proposal_for_unrelated_goal_or_missing_target(self):
        intu = Intuition([self.routine()])
        table = [Element("file", "calc.py  (python, 3 lines)")]
        self.assertIsNone(intu.propose("render the julia lattice", [], table, NEEDS))
        self.assertIsNone(intu.propose(GOAL, [], [Element("file", "other.py")], NEEDS))
        self.assertIsNone(intu.propose(GOAL, [("SEARCH", "")], table, NEEDS))   # prefix mismatch

    def test_jev_skips_the_model_and_falls_back_on_surprise(self):
        with tempfile.TemporaryDirectory() as d:
            repo = write(Path(d) / "calc", BUGGY_REPO)
            fix = "<<<<<<< SEARCH\n    return sum(xs) / (len(xs) + 1)\n=======\n    return sum(xs) / len(xs)\n>>>>>>> REPLACE"
            calls = []
            def decide(s, p):
                calls.append(1)
                return good_decider(s, p)
            env = CodeEnvironment(repo, GOAL, smith=lambda *a: fix, verify_cmd=VERIFY)
            t = Jev(env, decide, config=JevConfig(max_steps=6), intuition=Intuition([self.routine(first="calc.py def mean(xs)")])).run(GOAL)
            self.assertEqual(t.outcome, "done")
            self.assertEqual((t.llm_calls, t.intuition_steps, len(calls)), (0, 4, 0))
            self.assertEqual([s.source for s in t.steps], ["intuition"] * 4)
            # a misleading routine: EDIT with a smith that cannot apply → surprise → System 2 finishes
            write(Path(d) / "calc", BUGGY_REPO)
            env = CodeEnvironment(repo, GOAL, smith=lambda *a: "no blocks here", verify_cmd=VERIFY)
            t = Jev(env, decide, config=JevConfig(max_steps=6), intuition=Intuition([self.routine(first="calc.py def mean(xs)")])).run(GOAL)
            self.assertTrue(t.surprised)
            self.assertGreater(t.llm_calls, 0)


class SwarmLearningTests(unittest.TestCase):
    def test_repeat_task_is_solved_by_intuition_and_forged_into_a_skill(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            soup = Soup(base / "soup.db")
            try:
                jev_calls = []
                def counting(role, *a):
                    if role == "jev":
                        jev_calls.append(1)
                    return llm(role, *a)
                per_run = []
                for _ in range(3):
                    write(base / "calc", BUGGY_REPO)
                    jev_calls.clear()
                    cfg = SwarmConfig(size=3, temperatures=(0.1, 0.5, 0.9), jev=JevConfig(max_steps=6))
                    res = Swarm(counting, soup, config=cfg).run(GOAL, base / "calc", VERIFY)
                    self.assertIsNotNone(res.winner)
                    per_run.append(len(jev_calls))
                self.assertLess(per_run[1], per_run[0])
                self.assertTrue(any(t.intuition_steps for t in res.trajectories))
                rec = json.loads(soup.all(kind="trajectory")[0].data)
                self.assertEqual([s["op"] for s in rec["steps"]], ["OPEN", "EDIT", "RUN", "DONE"])

                skills = forge(soup, min_support=2)
                self.assertEqual(len(skills), 1)
                sk = skills[0]
                self.assertIn("calc", sk.name)
                self.assertEqual(sk.shape(), "OPEN(calc.py def mean(xs)) → EDIT(calc.py def mean(xs)) → RUN → DONE")
                self.assertTrue(soup.all(kind="skill"))
                out = export(skills, base / "skills")
                text = out[0].read_text()
                self.assertTrue(text.startswith("---\nname: "))
                self.assertIn("-m unittest -q test_calc", text)
                # hand-written skills are never clobbered
                (base / "skills" / "mine").mkdir()
                (base / "skills" / "mine" / "SKILL.md").write_text("hand written")
                sk.name = "mine"
                self.assertEqual(export([sk], base / "skills"), [])
            finally:
                soup.close()


if __name__ == "__main__":
    unittest.main()
