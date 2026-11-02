import json
import os
import re
import sys
import tempfile
import unittest

from kgirl.harness.jev.code_env import recover_blocks
from pathlib import Path

from tests.harness._fixtures import BUGGY_REPO, write

from kgirl.harness.jev import (CODE_OPS, CodeEnvironment, InvalidDecision, Jev, JevConfig, Swarm, SwarmConfig,
                               parse_decision)
from kgirl.harness.jev.intuition import Intuition
from kgirl.harness.skills import forge
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

    def test_parse_accepts_an_argument_without_the_separator(self):
        d = parse_decision('SEARCH "pop index" out of range', CODE_OPS, 0)
        self.assertEqual((d.op, d.arg), ("SEARCH", 'pop index" out of range'))
        with self.assertRaises(InvalidDecision):
            parse_decision("EDIT some words", CODE_OPS, 3)
        with self.assertRaises(InvalidDecision):
            parse_decision("REMOVE some words", CODE_OPS, 3)

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
        self.assertEqual([s.op for s in traj.steps], ["OPEN", "EDIT"])
        self.assertEqual(env.verify()[0], 0)
        self.assertIn("+    return sum(xs) / len(xs)", env.diff())

    def test_a_fix_ends_the_run_even_if_the_model_never_claims_it(self):
        env = CodeEnvironment(self.repo, "make test_calc pass", smith=lambda *a: FIX, verify_cmd=VERIFY)
        idx = next(i for i, e in enumerate(env.elements) if e.ref.get("path") == "calc.py")
        traj = Jev(env, lambda s, p: f"EDIT {idx}", config=JevConfig(max_steps=6)).run(env.goal)
        self.assertEqual((traj.outcome, len(traj.steps)), ("done", 1))

    def test_a_wrong_edit_reports_the_new_failure_and_the_run_continues(self):
        wrong = "<<<<<<< SEARCH\n    return sum(xs) / (len(xs) + 1)\n=======\n    return 0\n>>>>>>> REPLACE"
        env = CodeEnvironment(self.repo, "make test_calc pass", smith=lambda *a: wrong, verify_cmd=VERIFY)
        idx = next(i for i, e in enumerate(env.elements) if e.ref.get("path") == "calc.py")
        res = env.act(parse_decision(f"EDIT {idx}", CODE_OPS, len(env.elements)))
        self.assertTrue(res.ok)
        self.assertFalse(res.terminal)
        self.assertIn("still fails", res.note)

    def test_blocked_is_refused_until_a_source_edit_was_attempted(self):
        wrong = "<<<<<<< SEARCH\n    return sum(xs) / (len(xs) + 1)\n=======\n    return 0\n>>>>>>> REPLACE"
        env = CodeEnvironment(self.repo, "make test_calc pass", smith=lambda *a: wrong, verify_cmd=VERIFY)
        blocked = parse_decision("BLOCKED :: the test is wrong", CODE_OPS, len(env.elements))
        res = env.act(blocked)
        self.assertFalse(res.terminal)
        self.assertIn("BLOCKED refused", res.note)
        test_idx = next(i for i, e in enumerate(env.elements) if e.ref.get("path") == "test_calc.py")
        env.act(parse_decision(f"EDIT {test_idx}", CODE_OPS, len(env.elements)))
        self.assertFalse(env.act(blocked).terminal, "a refused edit on a test doesn't open the gate")
        idx = next(i for i, e in enumerate(env.elements) if e.ref.get("path") == "calc.py")
        env.act(parse_decision(f"EDIT {idx}", CODE_OPS, len(env.elements)))
        self.assertTrue(env.act(blocked).terminal)

    def test_blocked_ends_the_run_without_a_verifier_or_a_code_model(self):
        for kwargs in ({"smith": lambda *a: FIX}, {"verify_cmd": VERIFY}):
            env = CodeEnvironment(self.repo, "make test_calc pass", **kwargs)
            self.assertTrue(env.act(parse_decision("BLOCKED :: no way", CODE_OPS, len(env.elements))).terminal, kwargs)

    def test_failure_is_reproduced_first_and_points_at_the_source(self):
        (self.repo / "calc.py").write_text("def mean(xs):\n    raise ValueError('boom')\n")
        env = CodeEnvironment(self.repo, "make it work", verify_cmd=VERIFY)
        self.assertIn("boom", env.briefing)
        self.assertTrue(env.elements[0].label.startswith("calc.py:2"), env.elements[0].label)
        self.assertEqual(env.elements[0].ref["line"], 1)
        prompt = Jev(env, lambda s, p: "RUN").prompt("make it work", [])
        self.assertIn("CURRENT FAILURE", prompt)

    def test_tests_are_read_only_when_a_verifier_judges(self):
        seen = []
        env = CodeEnvironment(self.repo, "make test_calc pass", smith=lambda *a: seen.append(a) or FIX, verify_cmd=VERIFY)
        idx = next(i for i, e in enumerate(env.elements) if e.ref.get("path") == "test_calc.py")
        res = env.act(parse_decision(f"EDIT {idx}", CODE_OPS, len(env.elements)))
        self.assertFalse(res.ok)
        self.assertIn("tests judge", res.note)
        self.assertEqual(seen, [])

    def test_the_smith_is_told_why_its_last_attempt_failed(self):
        seen = []
        bad = "<<<<<<< SEARCH\nnot in the file\n=======\nx\n>>>>>>> REPLACE"
        env = CodeEnvironment(self.repo, "make test_calc pass", smith=lambda *a: seen.append(a[3]) or bad, verify_cmd=VERIFY)
        idx = next(i for i, e in enumerate(env.elements) if e.ref.get("path") == "calc.py")
        for _ in range(2):
            env.act(parse_decision(f"EDIT {idx}", CODE_OPS, len(env.elements)))
        self.assertIn("VERIFIER OUTPUT", seen[0])
        self.assertNotIn("LAST ATTEMPT", seen[0])
        self.assertIn("YOUR LAST ATTEMPT FAILED: EDIT: SEARCH text found 0x", seen[1])

    def test_search_text_may_differ_in_indentation_only(self):
        unindented = "<<<<<<< SEARCH\nreturn sum(xs) / (len(xs) + 1)\n=======\nreturn sum(xs) / len(xs)\n>>>>>>> REPLACE"
        env = CodeEnvironment(self.repo, "make test_calc pass", smith=lambda *a: unindented, verify_cmd=VERIFY)
        idx = next(i for i, e in enumerate(env.elements) if e.ref.get("path") == "calc.py")
        self.assertTrue(env.act(parse_decision(f"EDIT {idx}", CODE_OPS, len(env.elements))).terminal)
        self.assertIn("    return sum(xs) / len(xs)", (self.repo / "calc.py").read_text())

    def test_loose_matching_refuses_an_ambiguous_search(self):
        (self.repo / "dup.py").write_text("def a():\n    x = 1\ndef b():\n        x = 1\n")
        env = CodeEnvironment(self.repo, "change x in dup", smith=lambda *a: "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE")
        idx = next(i for i, e in enumerate(env.elements) if e.ref.get("path") == "dup.py")
        self.assertFalse(env.act(parse_decision(f"EDIT {idx}", CODE_OPS, len(env.elements))).ok)

    def test_verifier_runs_without_secret_environment(self):
        os.environ["FAKE_SERVICE_API_KEY"] = "do-not-leak"
        try:
            probe = [sys.executable, "-c", "import os; print(os.environ.get('FAKE_SERVICE_API_KEY', 'absent'))"]
            code, out = CodeEnvironment(self.repo, "x", verify_cmd=probe).verify()
        finally:
            os.environ.pop("FAKE_SERVICE_API_KEY")
        self.assertEqual((code, out.strip()), (0, "absent"))

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
        self.assertTrue(losers <= {"blocked", "exhausted", "invalid", "cancelled"}, losers)  # BLOCKED is refused until an edit
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

    def test_mcp_origin_stages_what_the_run_teaches(self):
        goal = "make test_calc pass: mean() is wrong"
        for _ in range(2):                                   # repetition must not promote it either
            write(self.base / "calc", BUGGY_REPO)
            res = self.swarm().run(goal, self.repo, VERIFY, origin="mcp")
            self.assertIsNotNone(res.winner)
        frags = {f.kind: f for f in self.soup.all()}
        self.assertEqual({k: (f.status, f.source) for k, f in frags.items()},
                         {"trajectory": ("staged", "mcp"), "abstraction": ("staged", "mcp")})
        self.assertEqual(frags["trajectory"].support, 2)
        self.assertIn(f"#{frags['trajectory'].id} staged trajectory", res.render())
        self.assertIn(f"soup promote {frags['trajectory'].id}", res.render())
        rec = json.loads(frags["trajectory"].data)
        self.assertEqual(rec["origin"], "mcp")
        self.assertTrue(rec["jev"].startswith("jev-"), rec["jev"])
        self.assertEqual(Intuition.from_soup(self.soup).routines, [])
        self.assertEqual(forge(self.soup), [])
        self.assertFalse(self.soup.recall(goal, scope="calc").fragments)
        # a person promotes the trajectory: now it replays and forges
        self.soup.set_status(frags["trajectory"].id, "active", "promoted by user")
        self.assertEqual(len(Intuition.from_soup(self.soup).routines), 1)
        self.assertEqual(len(forge(self.soup, store=False)), 1)

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



class RecoverBlocksTests(unittest.TestCase):
    REGION = ("import math\n\ndef mean(values):\n    \"\"\"Doc.\"\"\"\n    return sum(values) / (len(values) - 1)\n\n"
              "def other():\n    return 1")
    OLD = "def mean(values):\n    \"\"\"Doc.\"\"\"\n    return sum(values) / (len(values) - 1)"
    NEW = "def mean(values):\n    \"\"\"Doc.\"\"\"\n    return sum(values) / len(values)"

    def test_missing_markers(self):
        self.assertEqual(recover_blocks(f"{self.OLD}\n=======\n{self.NEW}", self.REGION), [(self.OLD, self.NEW)])
        self.assertEqual(recover_blocks(f"<<<<< SEARCH\n{self.OLD}\n=======\n{self.NEW}\n>>>>>>> REPLACE", self.REGION),
                         [(self.OLD, self.NEW)])

    def test_fenced_definition_replaces_only_that_definition(self):
        reply = f"<think>hmm</think>Explanation.\n\n```python\n{self.NEW}\n```\nDone."
        self.assertEqual(recover_blocks(reply, self.REGION), [(self.OLD, self.NEW)])

    def test_fenced_rewrite_adds_to_the_region(self):
        region = '"""Doc."""\n\nimport re\n\n\ndef a() -> int:\n    return 1\n'
        new = 'import re\n\n\ndef a() -> int:\n    return 1\n\n\ndef b() -> int:\n    return 2'
        blocks = recover_blocks(f"```python\n{new}\n```", region)
        self.assertEqual(blocks, [("import re\n\n\ndef a() -> int:\n    return 1", new)])
        self.assertEqual(region.replace(*blocks[0], 1), f'"""Doc."""\n\n{new}\n')

    def test_fenced_rewrite_sharing_too_little_gives_nothing(self):
        region = "import re\n\n\ndef a():\n    return 1\n\n\ndef b():\n    return 2\n\n\ndef c():\n    return 3"
        self.assertEqual(recover_blocks("```python\nimport re\nx = 1\n```", region), [])
        new = "import re\n" + "\n".join(f"y{i} = {i}" for i in range(6)) + "\n    return 3"
        self.assertEqual(recover_blocks(f"```python\n{new}\n```", region), [])

    def test_replacement_half_alone_gives_nothing(self):
        self.assertEqual(recover_blocks(f"=======\n{self.NEW}\n>>>>>>> REPLACE", self.REGION), [])

    def test_unrelated_fence_and_prose_give_nothing(self):
        self.assertEqual(recover_blocks("```python\ndef unknown():\n    pass\n```", self.REGION), [])
        self.assertEqual(recover_blocks("I would change the denominator.", self.REGION), [])
        self.assertEqual(recover_blocks("", self.REGION), [])
        self.assertEqual(recover_blocks(None, self.REGION), [])

if __name__ == "__main__":
    unittest.main()
