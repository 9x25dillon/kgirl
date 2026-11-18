"""Offline tests for mind.builder: agreement selection, repair, delivery and learning.

The model is a callable that answers from the system prompt and the temperature it is given, and the
sandbox is replaced by a table of which module passes which test: no Ollama, no network, and no generated
code is ever run.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tests.harness._fixtures  # noqa: F401  (sys.path)
from kgirl.harness.mind import builder as builder_mod
from kgirl.harness.mind import sandbox
from kgirl.harness.mind.build_spec import (atlas_context, build_summary_dict, candidate_prompt, donor_snippet,
                                           emit_files, expected_exports, explain_build, format_failures,
                                           fragment_utility, has_tests, missing_exports, normalize_suite,
                                           parse_multifile_reply, rank_exemplars, spec_checklist, spec_query,
                                           strip_numbered_lines, suite_prompt, suite_test_ids, test_method_source,
                                           test_weights, weighted_test_count)
from kgirl.harness.mind.builder import (EDGE_FIRST, IMPL_SYSTEM, REPAIR_SYSTEM, TEST_SYSTEM, BuildResult, Builder,
                                        Candidate, spec_from_text, suite_name)
from kgirl.harness.soup import Soup

SPEC = "Write mean(xs): the arithmetic mean of a list of numbers; raise ValueError when the list is empty."

SUITE = '''import unittest
from solution import mean


class TestMean(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(mean([1, 2, 3]), 2)

    def test_empty(self):
        with self.assertRaises(ValueError):
            mean([])
'''

BASIC, EMPTY = "test_gen0:TestMean.test_basic", "test_gen0:TestMean.test_empty"

SHORT = "def mean(xs):\n    return sum(xs) / len(xs)\n"                       # passes test_basic only
LONG = "def mean(xs):\n    total = sum(xs)\n    return total / len(xs)\n"     # passes test_basic only
ODD = "def mean(xs):\n    if not xs:\n        raise ValueError('empty')\n    return 0\n"   # passes test_empty only
FIXED = "def mean(xs):\n    if not xs:\n        raise ValueError('empty')\n    return sum(xs) / len(xs)\n"

PASSES = {SHORT: {BASIC}, LONG: {BASIC}, ODD: {EMPTY}, FIXED: {BASIC, EMPTY}}


def fenced(code: str) -> str:
    return f"Here is the module.\n```python\n{code}```\n"


def table_run_suite(code, suite, name, timeout=20.0, per_test=5.0):
    """Stands in for sandbox.run_suite: looks the module up instead of running it."""
    passed = set(PASSES.get(code, set()))
    return passed, {t: f"{t} failed" for t in (BASIC, EMPTY) if t not in passed}


class ModelByPrompt:
    """Answers from (system prompt, temperature), so parallel calls get the same reply every time."""

    def __init__(self, impls: dict, suite: str = fenced(SUITE), repair: str = fenced(FIXED)):
        self.impls, self.suite, self.repair = impls, suite, repair
        self.calls: list[tuple[str, str, int, float]] = []

    def __call__(self, system, prompt, max_tokens, temperature):
        self.calls.append((system, prompt, max_tokens, temperature))
        if system == TEST_SYSTEM:
            return self.suite
        if system == REPAIR_SYSTEM:
            return self.repair
        return self.impls[temperature]


class OfflineBuilderCase(unittest.TestCase):
    def setUp(self):
        for target, name, value in ((sandbox, "level", mock.Mock(return_value="none")),
                                    (sandbox, "imports_ok", mock.Mock(return_value=(True, ""))),
                                    (builder_mod, "run_suite", mock.Mock(side_effect=table_run_suite))):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.events: list[tuple[str, dict]] = []

    def builder(self, llm, **kw) -> Builder:
        kw.setdefault("n_impl", 3)
        kw.setdefault("n_suites", 1)
        return Builder(llm, emit=lambda topic, payload: self.events.append((topic, payload)), **kw)


class SmallPiecesTests(unittest.TestCase):
    def test_suite_name_and_spec_from_text(self):
        self.assertEqual(suite_name(2), "test_gen2")
        self.assertEqual(spec_from_text("planning the spec\n</think>\n  Write f(x).  "), "Write f(x).")

    def test_candidate_ok(self):
        self.assertTrue(Candidate(0, "x = 1\n", 0.2, "direct").ok)
        self.assertFalse(Candidate(0, "", 0.2, "direct").ok)
        self.assertFalse(Candidate(0, "x = 1\n", 0.2, "direct", dropped="imports socket (network)").ok)


class AgreementTests(unittest.TestCase):
    @staticmethod
    def cand(idx: int, size: int, *passed: str) -> Candidate:
        return Candidate(idx, "x" * size, 0.2, "direct", passed=frozenset(passed))

    def test_the_bigger_agreeing_group_wins_and_its_shortest_member_is_picked(self):
        a, b, lone = self.cand(0, 30, "t1", "t2"), self.cand(1, 20, "t1", "t2"), self.cand(2, 10, "t1", "t2", "t3")
        pick, group = Builder._agree([a, b, lone])                  # 2 members x 2 tests beats 1 x 3
        self.assertIs(pick, b)
        self.assertEqual(group, [a, b])

    def test_equal_scores_prefer_the_group_that_passes_more_tests(self):
        a, b, lone = self.cand(0, 30, "t1", "t2"), self.cand(1, 20, "t1", "t2"), \
            self.cand(2, 50, "t1", "t2", "t3", "t4")
        pick, group = Builder._agree([a, b, lone])                  # 2 x 2 ties 1 x 4
        self.assertIs(pick, lone)
        self.assertEqual(group, [lone])

    def test_a_full_tie_prefers_the_shorter_module(self):
        a, b = self.cand(0, 30, "t1"), self.cand(1, 10, "t2")
        pick, _ = Builder._agree([a, b])
        self.assertIs(pick, b)

    def test_equal_length_members_are_picked_by_lowest_index(self):
        a, b = self.cand(4, 10, "t1"), self.cand(2, 10, "t1")
        pick, group = Builder._agree([a, b])
        self.assertIs(pick, b)
        self.assertEqual(group, [a, b])

    def test_when_nothing_passes_everyone_agrees(self):
        a, b, c = self.cand(0, 30), self.cand(1, 10), self.cand(2, 20)
        pick, group = Builder._agree([a, b, c])
        self.assertIs(pick, b)
        self.assertEqual(len(group), 3)

    def test_failing_cluster_never_outscores_a_passing_candidate(self):
        fail_a, fail_b, fail_c, fail_d = (self.cand(i, 10) for i in range(4))
        passing = self.cand(4, 40, "test_gen0:T.test_a")
        pick, group = Builder._agree([fail_a, fail_b, fail_c, fail_d, passing])
        self.assertIs(pick, passing)
        self.assertEqual(group, [passing])

    def test_tie_prefers_tests_spanning_more_independent_suites(self):
        one_suite = self.cand(0, 10, "test_gen0:T.test_a", "test_gen0:T.test_b")
        two_suites = self.cand(1, 40, "test_gen0:T.test_a", "test_gen1:T.test_b")
        pick, group = Builder._agree([one_suite, two_suites])
        self.assertIs(pick, two_suites)
        self.assertEqual(group, [two_suites])

    def test_tie_prefers_higher_per_test_candidate_consensus(self):
        shared_a = self.cand(0, 40, "test_gen0:T.test_1", "test_gen0:T.test_2")
        helper = self.cand(1, 50, "test_gen0:T.test_1")
        isolated = self.cand(2, 10, "test_gen0:T.test_3", "test_gen0:T.test_4")
        pick, _ = Builder._agree([shared_a, helper, isolated])
        self.assertIs(pick, shared_a)

    def test_split_sibling_assertions_are_weighted_as_one_over_n(self):
        split_only = self.cand(0, 10, "test_gen0:T.test_cases_1", "test_gen0:T.test_cases_2", "test_gen0:T.test_cases_3")
        two_methods = self.cand(1, 40, "test_gen0:T.test_basic", "test_gen0:T.test_empty")
        pick, group = Builder._agree([split_only, two_methods])
        self.assertIs(pick, two_methods)
        self.assertEqual(group, [two_methods])
        weights = test_weights(["test_gen0:T.test_cases_1", "test_gen0:T.test_cases_2", "test_gen0:T.test_basic"])
        self.assertAlmostEqual(weights["test_gen0:T.test_cases_1"], 0.5)
        self.assertEqual(weights["test_gen0:T.test_basic"], 1.0)
        self.assertEqual(weighted_test_count(["test_gen0:T.test_cases_1", "test_gen0:T.test_cases_2"], weights), 1.0)

    def test_credible_is_every_test_someone_passes(self):
        self.assertEqual(Builder._credible([self.cand(0, 1, "t1"), self.cand(1, 1, "t2", "t3"), self.cand(2, 1)]),
                         {"t1", "t2", "t3"})
        self.assertEqual(Builder._credible([]), set())


class BuildResultTests(unittest.TestCase):
    @staticmethod
    def result(**kw) -> BuildResult:
        base = dict(spec="s", code="x = 1\n", tests="", chosen=0, group_size=2, passed=3, credible=4,
                    candidates=[], suites=[], exemplars=[])
        base.update(kw)
        return BuildResult(**base)

    @staticmethod
    def live(n: int) -> list[Candidate]:
        return [Candidate(i, "x = 1\n", 0.2, "direct") for i in range(n)]

    def test_confidence_is_zero_without_code_or_credible_tests(self):
        self.assertEqual(self.result(code="").confidence, 0.0)
        self.assertEqual(self.result(credible=0, passed=0).confidence, 0.0)

    def test_confidence_is_the_share_passed_when_enough_agree(self):
        self.assertEqual(self.result(candidates=self.live(4), group_size=2, passed=8, credible=10).confidence, 0.8)

    def test_confidence_is_discounted_when_few_agree(self):
        self.assertEqual(self.result(candidates=self.live(6), group_size=1, passed=4, credible=4).confidence, 0.667)

    def test_dropped_candidates_do_not_count_as_disagreement(self):
        dropped = [Candidate(9, "", 0.2, "direct", dropped="no Python module in the reply") for _ in range(4)]
        res = self.result(candidates=self.live(2) + dropped, group_size=2, passed=4, credible=4)
        self.assertEqual(res.confidence, 1.0)

    def test_render_of_a_build(self):
        both = frozenset({"a", "b", "c"})
        cands = [Candidate(0, "x = 1\n", 0.2, "direct", passed=both),
                 Candidate(1, "y = 1\n", 0.5, "edge-first", passed=both),
                 Candidate(2, "", 0.8, "direct", dropped="no Python module in the reply"),
                 Candidate(100, "z = 1\n", 0.2, "repair", passed=frozenset({"a"}), repaired=True)]
        res = self.result(candidates=cands, exemplars=[7, 9], notes=["repair round 1 produced no usable module"],
                          disputed={"test_gen0:T.test_b": "AssertionError"}, sandbox="none")
        self.assertEqual(res.render().split("\n"), [
            "built: candidate #0 (group of 2), passes 3/4 credible tests, confidence 0.75",
            "  ★ #0 direct     T=0.2 passed=3",
            "  · #1 edge-first T=0.5 passed=3",
            "  ✗ #2 direct     T=0.8 passed=0   dropped: no Python module in the reply",
            "  · #100 repair     T=0.2 passed=1   repaired",
            "  note: repair round 1 produced no usable module",
            "  disputed (the module fails these; the test or the module is wrong): test_gen0:T.test_b",
            "  recalled exemplars: #7, #9",
            "  sandbox: resource limits only (no isolation)"])

    def test_render_of_a_failed_build(self):
        res = self.result(code="", chosen=None, group_size=0, passed=0, credible=0,
                          candidates=[Candidate(0, "", 0.2, "direct", dropped="the model call failed: boom")])
        lines = res.render().split("\n")
        self.assertEqual(lines[0], "build failed: no usable module")
        self.assertEqual(lines[1], "  ✗ #0 direct     T=0.2 passed=0   dropped: the model call failed: boom")

    def test_render_lists_at_most_six_disputed_tests(self):
        res = self.result(disputed={f"t{i}": "out" for i in range(9)})
        line = [ln for ln in res.render().split("\n") if ln.startswith("  disputed")][0]
        self.assertTrue(line.endswith("t0, t1, t2, t3, t4, t5 (+3 more)"), line)


class GenerationTests(OfflineBuilderCase):
    def test_candidates_alternate_style_and_spread_temperatures(self):
        llm = ModelByPrompt({0.2: fenced(SHORT), 0.5: fenced(LONG), 0.8: fenced(ODD)})
        b = self.builder(llm)
        first, second, third = (b._candidate(SPEC, "", i) for i in range(3))
        self.assertEqual([(c.idx, c.temperature, c.style, c.ok) for c in (first, second, third)],
                         [(0, 0.2, "direct", True), (1, 0.5, "edge-first", True), (2, 0.8, "direct", True)])
        self.assertEqual(first.code, SHORT)
        self.assertEqual(llm.calls[0], (IMPL_SYSTEM, f"SPEC:\n{SPEC}", 2048, 0.2))
        self.assertEqual(llm.calls[1][1], f"SPEC:\n{SPEC}\n\n{EDGE_FIRST}")

    def test_exemplars_lead_the_prompt(self):
        llm = ModelByPrompt({0.2: fenced(SHORT)})
        self.builder(llm)._candidate(SPEC, "EXAMPLE 1 (accepted earlier):\nSPEC: median", 0)
        self.assertEqual(llm.calls[0][1], f"EXAMPLE 1 (accepted earlier):\nSPEC: median\n\nSPEC:\n{SPEC}")

    def test_a_candidate_is_dropped_with_the_reason(self):
        def failing(system, prompt, max_tokens, temperature):
            raise RuntimeError("ollama is not running")

        self.assertEqual(self.builder(failing)._candidate(SPEC, "", 0).dropped,
                         "the model call failed: ollama is not running")
        prose = ModelByPrompt({0.2: "I am not able to write that module."})
        self.assertEqual(self.builder(prose)._candidate(SPEC, "", 0).dropped, "no Python module in the reply")
        net = ModelByPrompt({0.2: fenced("import subprocess\n\ndef mean(xs):\n    return 0\n")})
        self.assertEqual(self.builder(net)._candidate(SPEC, "", 0).dropped,
                         "imports subprocess (processes or native code)")
        sandbox.imports_ok.assert_not_called()                 # none of the three got as far as an import

    def test_a_module_that_does_not_import_is_dropped_with_its_last_error_line(self):
        sandbox.imports_ok.return_value = (False, "Traceback (most recent call last):\n  ...\n"
                                                  "NameError: name 'total' is not defined\n")
        c = self.builder(ModelByPrompt({0.2: fenced(SHORT)}))._candidate(SPEC, "", 0)
        self.assertEqual(c.dropped, "does not import: NameError: name 'total' is not defined")
        self.assertFalse(c.ok)

    def test_file_access_is_only_allowed_when_the_builder_allows_io(self):
        code = "def load(path):\n    with open(path) as fh:\n        return fh.read()\n"
        llm = ModelByPrompt({0.2: fenced(code)})
        self.assertEqual(self.builder(llm)._candidate(SPEC, "", 0).dropped, "opens files")
        self.assertTrue(self.builder(llm, allow_io=True)._candidate(SPEC, "", 0).ok)

    def test_suites_are_kept_or_dropped_with_the_reason(self):
        b = self.builder(ModelByPrompt({}))
        self.assertEqual(b._suite(SPEC, 0), (SUITE, ""))
        for reply, why in (("I would test the mean of a few lists.", "no Python module in the reply"),
                           (fenced("import unittest\n"), "it does not import the solution module"),
                           (fenced("import socket\nfrom solution import mean\n"), "imports socket (network)")):
            self.assertEqual(self.builder(ModelByPrompt({}, suite=reply))._suite(SPEC, 0), ("", why))

    def test_suite_temperatures_cycle(self):
        llm = ModelByPrompt({})
        b = self.builder(llm)
        for i in range(5):
            b._suite(SPEC, i)
        self.assertEqual([call[3] for call in llm.calls], [0.2, 0.5, 0.8, 0.4, 0.2])
        self.assertEqual(llm.calls[0][:3], (TEST_SYSTEM, f"SPEC:\n{SPEC}", 2048))


class BuildTests(OfflineBuilderCase):
    def test_agreement_then_repair_to_a_strict_superset(self):
        llm = ModelByPrompt({0.2: fenced(LONG), 0.5: fenced(SHORT), 0.8: fenced(ODD)})
        res = self.builder(llm).build("\n    " + SPEC + "\n")
        self.assertEqual(res.spec, SPEC)
        self.assertEqual(res.code, FIXED)
        self.assertEqual((res.chosen, res.group_size, res.passed, res.credible), (100, 1, 2, 2))
        self.assertEqual(res.notes, ["repair round 1: 1 -> 2 tests"])
        self.assertEqual(res.disputed, {})
        self.assertEqual(res.suites, [SUITE])
        self.assertEqual(res.agreed, [SUITE])                   # the module passes every test: nothing is cut
        self.assertIn("generated suite 1", res.tests)
        self.assertEqual([(c.idx, c.style, c.repaired) for c in res.candidates],
                         [(0, "direct", False), (1, "edge-first", False), (2, "direct", False), (100, "repair", True)])
        self.assertEqual(res.confidence, 0.75)
        self.assertEqual(res.sandbox, "none")

        repair = [call for call in llm.calls if call[0] == REPAIR_SYSTEM]
        self.assertEqual(len(repair), 1)
        self.assertIn(f"CURRENT solution.py:\n```python\n{SHORT}```", repair[0][1])     # the shortest of the pair
        self.assertIn(f"--- {EMPTY}\n{EMPTY} failed", repair[0][1])
        self.assertNotIn(BASIC, repair[0][1])
        self.assertEqual(repair[0][3], 0.2)

        self.assertEqual(self.events[0], ("build.start", {"spec_chars": len(SPEC), "exemplars": [], "sandbox": "none"}))
        self.assertEqual(self.events[-1], ("build.end", {"chosen": 100, "group": 1, "passed": 2, "credible": 2,
                                                         "confidence": 0.75, "sandbox": "none"}))

    def test_a_repair_that_is_not_a_strict_superset_is_not_kept(self):
        llm = ModelByPrompt({0.2: fenced(LONG), 0.5: fenced(SHORT), 0.8: fenced(ODD)}, repair=fenced(ODD))
        res = self.builder(llm, repair_rounds=1).build(SPEC)
        self.assertEqual(res.code, SHORT)
        self.assertEqual((res.chosen, res.group_size, res.passed, res.credible), (1, 2, 1, 2))
        self.assertEqual(res.notes, ["repair round 1 did not pass a strict superset; kept the original"])
        self.assertEqual(res.disputed, {EMPTY: f"{EMPTY} failed"})
        self.assertEqual(res.candidates[-1].dropped, "did not pass a strict superset of the tests")
        self.assertEqual(len(res.agreed), 1)
        self.assertIn("def test_basic", res.agreed[0])
        self.assertNotIn("def test_empty", res.agreed[0])       # delivered tests pass as delivered

    def test_no_repair_when_nobody_passes_the_failing_test(self):
        llm = ModelByPrompt({0.2: fenced(LONG), 0.5: fenced(SHORT), 0.8: fenced(SHORT)})
        res = self.builder(llm).build(SPEC)
        self.assertEqual((res.code, res.group_size, res.passed, res.credible), (SHORT, 3, 1, 1))
        self.assertEqual([call for call in llm.calls if call[0] == REPAIR_SYSTEM], [])
        self.assertEqual(res.disputed, {})                      # test_empty is not credible: nobody passes it
        self.assertEqual(res.confidence, 1.0)

    def test_no_usable_candidate(self):
        llm = ModelByPrompt({0.2: "no code", 0.5: "still no code", 0.8: fenced("import socket\n")})
        res = self.builder(llm).build(SPEC)
        self.assertEqual((res.code, res.chosen, res.group_size, res.confidence), ("", None, 0, 0.0))
        self.assertEqual(res.notes, ["no candidate parsed, passed the static check and imported"])
        self.assertEqual([c.dropped for c in res.candidates],
                         ["no Python module in the reply", "no Python module in the reply", "imports socket (network)"])
        builder_mod.run_suite.assert_not_called()
        self.assertEqual(self.events[-1], ("build.end", {"chosen": None, "candidates": 3}))
        self.assertTrue(res.render().startswith("build failed: no usable module\n"))

    def test_no_suite_survives(self):
        llm = ModelByPrompt({0.2: fenced(LONG), 0.5: fenced(SHORT), 0.8: fenced(ODD)}, suite="I cannot write tests.")
        res = self.builder(llm).build(SPEC)
        self.assertEqual(res.notes, ["suite 1 dropped: no Python module in the reply",
                                     "no test suite survived, so nothing was checked: picked the shortest module "
                                     "that imports"])
        self.assertEqual((res.code, res.group_size, res.passed, res.credible, res.confidence), (SHORT, 3, 0, 0, 0.0))
        self.assertEqual((res.suites, res.agreed, res.tests), ([], [], ""))
        builder_mod.run_suite.assert_not_called()


class LearnTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.soup = Soup(Path(self.tmp.name) / "soup.db")
        self.builder = Builder(lambda *a: "", self.soup)

    def tearDown(self):
        self.soup.close()
        self.tmp.cleanup()

    @staticmethod
    def result(code: str = SHORT, exemplars: list | None = None) -> BuildResult:
        return BuildResult(spec="Write   mean(xs):\n the arithmetic mean.", code=code, tests="", chosen=0, group_size=1,
                           passed=1, credible=1, candidates=[], suites=[], exemplars=exemplars or [])

    def test_an_accepted_build_becomes_an_active_exemplar(self):
        earlier = self.soup.add("SPEC: median\n\nSOLUTION: ...", kind="exemplar")
        fid = self.builder.learn(self.result(exemplars=[earlier]), accepted=True, scope="kgirl", note="reviewed")
        frag = self.soup.get(fid)
        self.assertEqual((frag.kind, frag.status, frag.source, frag.scope, frag.tags),
                         ("exemplar", "active", "user", "kgirl", "build"))
        self.assertEqual(frag.text, "SPEC: Write mean(xs): the arithmetic mean.\n\nSOLUTION:\n```python\n"
                                    + SHORT.strip() + "\n```\n\nNOTE: reviewed")
        self.assertEqual(frag.wins, 1.0)
        self.assertEqual(self.soup.get(earlier).wins, 1.0)      # the worked example it was built from helped

    def test_an_unreviewed_build_is_only_staged(self):
        earlier = self.soup.add("SPEC: median\n\nSOLUTION: ...", kind="exemplar")
        fid = self.builder.learn(self.result(exemplars=[earlier]), accepted=False)
        frag = self.soup.get(fid)
        self.assertEqual((frag.status, frag.source), ("staged", "build"))
        self.assertEqual(self.soup.get(earlier).wins, 0.0)
        other = self.builder.learn(self.result(code=LONG), accepted=False, origin="mcp")
        self.assertEqual(self.soup.get(other).source, "mcp")    # a source the curator never promotes by itself

    def test_nothing_to_learn_without_code_or_memory(self):
        self.assertIsNone(self.builder.learn(self.result(code=""), accepted=True))
        self.assertIsNone(Builder(lambda *a: "").learn(self.result(), accepted=True))
        self.assertEqual(self.soup.all(), [])

    def test_only_active_exemplars_are_recalled_as_worked_examples(self):
        self.assertEqual(self.builder._exemplars(SPEC, "*"), ("", []))
        staged = self.builder.learn(self.result(), accepted=False)
        self.assertEqual(self.builder._exemplars("Write mean(xs): the arithmetic mean of numbers.", "*"), ("", []))
        self.soup.set_status(staged, "active", "accepted by the user")
        shots, ids = self.builder._exemplars("Write mean(xs): the arithmetic mean of numbers.", "*")
        self.assertEqual(ids, [staged])
        self.assertTrue(shots.startswith("EXAMPLE 1 (accepted earlier):\nSPEC: Write mean(xs)"), shots)
        self.assertEqual(Builder(lambda *a: "")._exemplars(SPEC, "*"), ("", []))

    def test_unrelated_exemplars_sharing_only_boilerplate_are_filtered_out(self):
        unrelated = self.soup.add(
            "SPEC: Write spiral(matrix: list[list[int]]) -> list[int]; raise ValueError on ragged rows.\n\n"
            "SOLUTION:\n```python\ndef spiral(matrix):\n    return []\n```",
            kind="exemplar",
            status="active",
        )
        shots, ids = self.builder._exemplars(SPEC, "*")
        self.assertNotIn(unrelated, ids)
        self.assertEqual((shots, ids), ("", []))

    def test_use_exemplars_false_skips_recall_while_allowing_learn(self):
        fid = self.builder.learn(self.result(), accepted=True)
        cold = Builder(lambda *a: "", self.soup, use_exemplars=False)
        self.assertEqual(cold._exemplars(SPEC, "*"), ("", []))
        self.assertIsNotNone(fid)

    def test_learn_stores_agreed_tests_in_fragment_data(self):
        res = self.result()
        res.tests = SUITE
        fid = self.builder.learn(res, accepted=True)
        self.assertEqual(self.soup.get(fid).data, SUITE.strip())

    def test_confirm_activates_user_accepted_winner_while_mcp_stays_staged(self):
        earlier = self.soup.add("SPEC: median\n\nSOLUTION: ...", kind="exemplar")
        fid = self.builder.confirm(self.result(exemplars=[earlier]), scope="kgirl", note="confirmed")
        frag = self.soup.get(fid)
        self.assertEqual((frag.kind, frag.status, frag.source, frag.scope),
                         ("exemplar", "active", "user", "kgirl"))
        self.assertEqual(frag.wins, 1.0)
        self.assertEqual(self.soup.get(earlier).wins, 1.0)
        mcp_id = self.builder.learn(self.result(code=LONG), accepted=True, origin="mcp")
        self.assertEqual((self.soup.get(mcp_id).status, self.soup.get(mcp_id).source), ("staged", "mcp"))
        self.assertAlmostEqual(fragment_utility(frag), 2.0 / 3.0)
        self.assertIsNone(self.builder.confirm(self.result(code="")))


class SpecAndSuiteNormalizationTests(unittest.TestCase):
    def test_spec_checklist_extracts_signatures_properties_exceptions_and_invariants(self):
        spec = (
            "Write class `MovingAverage(size: int)`. `size` < 1 raises ValueError.\n"
            "- `add(x: float) -> float` adds a value. The input must not be modified.\n"
            "- `count` property: how many values are in the window.\n"
            "`add` must run in O(1) time. `decode` inverts `encode`."
        )
        checklist = spec_checklist(spec)
        self.assertIn("`MovingAverage(size: int)`", checklist)
        self.assertIn("`add(x: float) -> float`", checklist)
        self.assertIn("`count`", checklist)
        self.assertIn("@property", checklist)
        self.assertIn("ValueError", checklist)
        self.assertIn("not mutate", checklist)
        self.assertIn("round-trip", checklist)
        self.assertIn("O(1)", checklist)

    def test_suite_prompt_keeps_first_direct_and_diversifies_later_suites(self):
        self.assertEqual(suite_prompt(SPEC, 0), f"SPEC:\n{SPEC}")
        p1 = suite_prompt(SPEC, 1)
        p2 = suite_prompt(SPEC, 2)
        self.assertIn("boundary", p1.lower())
        self.assertIn("contract", p2.lower())
        self.assertIn("ValueError", p1)

    def test_normalize_suite_leaves_single_assert_and_stateful_methods_unchanged(self):
        self.assertEqual(normalize_suite(SUITE), SUITE)
        stateful = (
            "import unittest\nfrom solution import MovingAverage\n\n"
            "class T(unittest.TestCase):\n"
            "    def test_window(self):\n"
            "        m = MovingAverage(3)\n"
            "        self.assertEqual(m.add(1), 1)\n"
            "        self.assertEqual(m.add(3), 2)\n"
        )
        self.assertEqual(normalize_suite(stateful), stateful)

    def test_normalize_suite_splits_stateless_multi_assert_methods(self):
        multi = (
            "import unittest\nfrom solution import encode, decode\n\n"
            "class TestRLE(unittest.TestCase):\n"
            "    def test_cases(self):\n"
            "        self.assertEqual(encode('aaab'), '3a1b')\n"
            "        self.assertEqual(decode('3a1b'), 'aaab')\n"
            "        with self.assertRaises(ValueError):\n"
            "            encode('a1')\n"
        )
        out = normalize_suite(multi)
        self.assertIn("def test_cases_1(self):", out)
        self.assertIn("def test_cases_2(self):", out)
        self.assertIn("def test_cases_3(self):", out)
        self.assertNotIn("def test_cases(self):", out)
        compile(out, "<suite>", "exec")

    def test_normalize_suite_wraps_bare_test_functions_and_strips_bare_main(self):
        bare = (
            "from solution import mean\n\n"
            "def test_one():\n"
            "    assert mean([2, 4]) == 3\n\n"
            "unittest.main()\n"
        )
        out = normalize_suite(bare)
        self.assertTrue(has_tests(out))
        self.assertIn("class TestSpec(unittest.TestCase):", out)
        self.assertIn("def test_one(self):", out)
        self.assertNotIn("unittest.main()", out)


class TargetedRepairTests(OfflineBuilderCase):
    def test_suite_without_test_methods_is_dropped(self):
        reply = fenced("import unittest\nfrom solution import mean\n\nclass Empty(unittest.TestCase):\n    pass\n")
        self.assertEqual(self.builder(ModelByPrompt({}, suite=reply))._suite(SPEC, 0),
                         ("", "it defines no test methods"))

    def test_repair_prompt_includes_failing_test_source_and_regression_warning(self):
        calls = []

        def llm(system, prompt, max_tokens, temperature):
            calls.append((system, prompt, temperature))
            if system == TEST_SYSTEM:
                return fenced(SUITE)
            if system == REPAIR_SYSTEM:
                return fenced(ODD if temperature < 0.3 else FIXED)
            return {0.2: fenced(LONG), 0.5: fenced(SHORT), 0.8: fenced(ODD)}[temperature]

        res = self.builder(llm, repair_rounds=2).build(SPEC)
        self.assertEqual(res.code, FIXED)
        repairs = [c for c in calls if c[0] == REPAIR_SYSTEM]
        self.assertEqual(len(repairs), 2)
        self.assertIn("def test_empty(self):", repairs[0][1])
        self.assertIn("REGRESSED IN PREVIOUS REPAIR", repairs[1][1])
        self.assertIn("def test_basic(self):", repairs[1][1])
        self.assertEqual(test_method_source([SUITE], EMPTY),
                         "def test_empty(self):\n    with self.assertRaises(ValueError):\n        mean([])")
        self.assertIn("def test_empty(self):", format_failures([SUITE], {EMPTY: "failed"}))
        self.assertIn("mean", spec_query(SPEC))
        self.assertEqual(rank_exemplars(SPEC, []), [])

    def test_explain_renders_suites_votes_repairs_and_winner_rationale(self):
        llm = ModelByPrompt({0.2: fenced(LONG), 0.5: fenced(SHORT), 0.8: fenced(ODD)})
        res = self.builder(llm).build(SPEC)
        explained = res.explain()
        self.assertEqual(explained, explain_build(res))
        self.assertIn("=== SUITES ===", explained)
        self.assertIn(BASIC, explained)
        self.assertIn(EMPTY, explained)
        self.assertIn("=== CANDIDATE VOTES & AGREEMENT GROUPS ===", explained)
        self.assertIn("group 1: members=[#0, #1]", explained)
        self.assertIn("=== REPAIR STEPS ===", explained)
        self.assertIn("round 1 (#100, T=0.20): 1 -> 2 tests", explained)
        self.assertIn("ACCEPTED (strict superset)", explained)
        self.assertIn("=== WHY THE WINNER WON ===", explained)
        self.assertIn("Initial dual-agreement pick", explained)
        self.assertIn("Targeted repair winner: candidate #100", explained)

    def test_explain_covers_failed_and_no_suite_builds(self):
        failed = BuildResult(spec=SPEC, code="", tests="", chosen=None, group_size=0, passed=0, credible=0,
                             candidates=[Candidate(0, "", 0.2, "direct", dropped="no Python module in the reply")],
                             suites=[], exemplars=[])
        self.assertIn("No candidate won", failed.explain())
        no_suite = BuildResult(spec=SPEC, code=SHORT, tests="", chosen=1, group_size=2, passed=0, credible=0,
                               candidates=[Candidate(0, LONG, 0.2, "direct"), Candidate(1, SHORT, 0.5, "edge-first")],
                               suites=[], exemplars=[])
        self.assertIn("won by fallback: no test suite survived", no_suite.explain())

    def test_multi_file_and_atlas_context_build(self):
        self.assertEqual(suite_test_ids(SUITE, "test_gen0"), [BASIC, EMPTY])
        self.assertIn("ValueError", candidate_prompt(SPEC, "", 2, EDGE_FIRST))
        numbered = "   1: def helper(x):\n   2:     return x + 1\n"
        self.assertEqual(strip_numbered_lines(numbered), "def helper(x):\n    return x + 1\n")

        fake_atlas = mock.Mock()
        fake_atlas.read_source.return_value = numbered
        ctx, extra = atlas_context(fake_atlas, "kgirl", ["src/pkg/helper.py"])
        self.assertIn("EXISTING REPO CONTEXT (from Atlas):", ctx)
        self.assertIn("def helper(x):", ctx)
        self.assertEqual(extra, {"helper.py": "def helper(x):\n    return x + 1\n"})

        with mock.patch("kgirl.harness.mind.build_spec.secret_reason", return_value="flagged"):
            with self.assertRaises(ValueError):
                atlas_context(fake_atlas, "kgirl", ["src/pkg/config.py"])

        sol, comp = parse_multifile_reply(
            "```python helpers.py\ndef add1(x):\n    return x + 1\n```\n"
            "```python\nfrom helpers import add1\ndef mean(xs):\n    return add1(0)\n```"
        )
        self.assertIn("def mean(xs):", sol)
        self.assertEqual(comp, {"helpers.py": "def add1(x):\n    return x + 1\n"})

        llm = ModelByPrompt({0.2: fenced(LONG), 0.5: fenced(SHORT), 0.8: fenced(ODD)})
        res = self.builder(llm).build_with_atlas(SPEC, fake_atlas, "kgirl", "src/pkg/helper.py")
        self.assertEqual(res.code, FIXED)
        self.assertEqual(res.extra_files, {"helper.py": "def helper(x):\n    return x + 1\n"})
        emitted = res.files("stats.py")
        self.assertEqual(set(emitted), {"stats.py", "helper.py", "test_stats.py"})
        self.assertIn("from stats import mean", emitted["test_stats.py"])
        self.assertEqual(emit_files("", [SUITE]), {})

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "stats.py"
            written = res.write(out_path)
            self.assertEqual(len(written), 3)
            self.assertTrue((Path(tmp) / "stats.py").is_file())
            self.assertTrue((Path(tmp) / "helper.py").is_file())
            self.assertTrue((Path(tmp) / "test_stats.py").is_file())

    def test_donor_snippet_and_summary_dict(self):
        llm = ModelByPrompt({0.2: fenced(LONG), 0.5: fenced(SHORT), 0.8: fenced(ODD)})
        res = self.builder(llm).build(SPEC)
        repairs = [c for c in llm.calls if c[0] == REPAIR_SYSTEM]
        self.assertIn("PASSING CANDIDATE REFERENCE (candidate #2 passes test_gen0:TestMean.test_empty):",
                      repairs[0][1])
        self.assertIn("raise ValueError('empty')", repairs[0][1])
        self.assertNotIn(BASIC, repairs[0][1])
        self.assertEqual(donor_snippet([], None, []), "")
        summary = res.summary_dict()
        self.assertEqual(summary, build_summary_dict(res))
        self.assertEqual((summary["chosen"], summary["passed"], summary["credible"]), (100, 2, 2))
        self.assertIn("=== WHY THE WINNER WON ===", summary["explain"])
        self.assertEqual(len(summary["candidates"]), 4)

    def test_expected_and_missing_exports_and_disputed_explanation(self):
        cls_spec = (
            "Write class `MovingAverage(size: int)`.\n"
            "- `add(x: float) -> float` adds a value.\n"
            "- `count` property: how many values are in the window."
        )
        self.assertEqual(expected_exports(cls_spec), (["MovingAverage"], ["add"], ["count"]))
        bad_cls = "class MovingAverage:\n    def count(self):\n        return 0\n"
        self.assertEqual(
            missing_exports(cls_spec, bad_cls),
            ["missing method `add`", "`count` is defined as a method without @property"],
        )
        good_cls = (
            "class MovingAverage:\n"
            "    def add(self, x):\n        return x\n"
            "    @property\n    def count(self):\n        return 0\n"
        )
        self.assertEqual(missing_exports(cls_spec, good_cls), [])
        self.assertEqual(expected_exports("Write `encode(s: str)` and `decode(s: str)`."),
                         (["encode", "decode"], [], []))
        self.assertEqual(missing_exports("Write `encode(s: str)` and `decode(s: str)`.", "def encode(s):\n    return s\n"),
                         ["missing top-level `decode`"])
        ann_and_imports = (
            "from codecs import encode as encode\n"
            "import math as _m\n"
            "try:\n"
            "    decode: object = lambda s: s\n"
            "except Exception:\n"
            "    def decode(s):\n"
            "        return s\n"
        )
        self.assertEqual(missing_exports("Write `encode(s: str)` and `decode(s: str)`.", ann_and_imports), [])
        if_try_cls = (
            "if True:\n"
            "    class MovingAverage:\n"
            "        try:\n"
            "            from operator import identity as add\n"
            "        except ImportError:\n"
            "            def add(self, x):\n"
            "                return x\n"
            "        count = property(lambda self: 0)\n"
        )
        self.assertEqual(missing_exports(cls_spec, if_try_cls), [])
        # Contract tie-breaking in _agree: complete exports beat shorter incomplete code on equal test score.
        incomplete = Candidate(0, "def encode(s):\n    return s\n", 0.2, "direct", passed=frozenset({"t1"}))
        complete = Candidate(
            1,
            "def encode(s):\n    return s\ndef decode(s):\n    return s\n",
            0.5,
            "edge-first",
            passed=frozenset({"t1"}),
        )
        pick_tie, _ = Builder._agree([incomplete, complete], "Write `encode(s: str)` and `decode(s: str)`.")
        self.assertIs(pick_tie, complete)

        llm = ModelByPrompt({0.2: fenced(LONG), 0.5: fenced(SHORT), 0.8: fenced(ODD)}, repair=fenced(ODD))
        res = self.builder(llm, repair_rounds=1).build(SPEC)
        explained = res.explain()
        self.assertIn("=== DISPUTED TESTS ===", explained)
        self.assertIn(f"{EMPTY} (passed by #2)", explained)


if __name__ == "__main__":
    unittest.main()
