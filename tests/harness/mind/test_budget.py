"""Offline tests for mind.budget: the one prompt budget, its priorities, and its record.

Pure functions on strings: no model, no stores, no network.
"""

import unittest

import tests.harness._fixtures  # noqa: F401  (sys.path)
from kgirl.harness.mind.budget import PRIORITY, Block, fit, render_report
from kgirl.harness.util import estimate_tokens


def lines(prefix: str, n: int, width: int = 80) -> list[str]:
    return [f"- {prefix}{i} " + "x" * width for i in range(n)]


class FitTests(unittest.TestCase):
    def blocks(self):
        return [Block("persona", ["P" * 400], fixed=True),
                Block("rules", ["RULES\n- never push"], fixed=True),
                Block("wants", lines("want", 4), "WANTS:", PRIORITY["wants"]),
                Block("memory", lines("mem", 6), "MEMORY:", PRIORITY["memory"]),
                Block("earlier", lines("old", 6), "EARLIER:", PRIORITY["earlier"]),
                Block("history", lines("turn", 6), priority=PRIORITY["history"], clip=False)]

    def test_everything_fits_so_nothing_is_cut(self):
        f = fit(self.blocks(), 10_000)
        self.assertEqual(f.cut, {})
        self.assertEqual(f.overflow, 0)
        self.assertEqual(f.kept["memory"], self.blocks()[3].items)
        self.assertEqual(f.text("wants").split("\n")[0], "WANTS:")
        self.assertEqual(f.meta()["dropped"], {})

    def test_the_lowest_priority_block_is_cut_first_and_from_its_end(self):
        full = fit(self.blocks(), 10_000).used
        f = fit(self.blocks(), full - 60)
        self.assertEqual(list(f.cut), ["earlier"])                 # one block gives enough
        self.assertEqual(f.kept["earlier"][:3], self.blocks()[4].items[:3])      # the best items stay
        self.assertEqual(f.kept["memory"], self.blocks()[3].items)
        self.assertLessEqual(f.used, full - 60)

    def test_blocks_go_in_priority_order_until_it_fits(self):
        b = self.blocks()
        full = fit(b, 10_000)
        budget = full.tokens("persona") + full.tokens("rules") + 60
        f = fit(b, budget)
        for name in ("earlier", "memory", "history"):              # all below WANTS: gone entirely, header too
            self.assertEqual(f.kept[name], [])
            self.assertEqual(f.text(name), "")
        self.assertEqual(f.cut["earlier"]["dropped"], 6)
        self.assertTrue(f.kept["wants"])                           # the highest-priority block survives, trimmed
        self.assertLessEqual(f.used, budget)
        self.assertEqual(f.overflow, 0)

    def test_rules_and_the_fixed_blocks_are_never_cut(self):
        f = fit(self.blocks(), 10)                                # absurdly small
        self.assertEqual(f.kept["persona"], ["P" * 400])
        self.assertEqual(f.kept["rules"], ["RULES\n- never push"])
        self.assertEqual(f.used, f.tokens("persona") + f.tokens("rules"))
        self.assertEqual(f.overflow, f.used - 10)                 # reported, not hidden
        self.assertNotIn("persona", f.cut)
        self.assertNotIn("rules", f.cut)

    def test_the_last_item_is_clipped_only_when_enough_of_it_stays(self):
        b = [Block("fixed", ["F" * 40], fixed=True),
             Block("memory", ["- short", "- " + "y" * 2000], "MEMORY:", PRIORITY["memory"])]
        total = fit(b, 10_000).used
        f = fit(b, total - 100)                                    # trims ~100 tokens off the long last item
        self.assertEqual(len(f.kept["memory"]), 2)
        self.assertTrue(f.kept["memory"][1].endswith("…"))
        self.assertTrue(f.cut["memory"]["clipped"])
        self.assertLessEqual(f.used, total - 100)
        f = fit(b, total - 490)                                    # not enough would stay: the item goes whole
        self.assertEqual(f.kept["memory"], ["- short"])
        self.assertFalse(f.cut["memory"]["clipped"])

    def test_history_is_never_clipped_only_dropped_oldest_first(self):
        b = [Block("fixed", ["F" * 40], fixed=True),
             Block("history", ["newest " + "n" * 300, "middle " + "m" * 300, "oldest " + "o" * 300],
                   priority=PRIORITY["history"], clip=False)]
        full = fit(b, 10_000).used
        f = fit(b, full - 60)
        self.assertEqual([t.split()[0] for t in f.kept["history"]], ["newest", "middle"])
        self.assertTrue(all(not t.endswith("…") for t in f.kept["history"]))

    def test_the_reserve_counts_against_the_budget_but_is_not_text(self):
        b = [Block("fixed", ["F" * 40], fixed=True), Block("tool_room", [], fixed=True, reserve=500),
             Block("memory", lines("m", 3), "MEMORY:", PRIORITY["memory"])]
        f = fit(b, 520)
        self.assertEqual(f.tokens("tool_room"), 500)
        self.assertEqual(f.text("tool_room"), "")
        self.assertEqual(f.kept["memory"], [])

    def test_it_is_deterministic(self):
        a = fit(self.blocks(), 300)
        b = fit(self.blocks(), 300)
        self.assertEqual((a.kept, a.cut, a.overflow), (b.kept, b.cut, b.overflow))

    def test_a_block_with_no_items_costs_nothing_and_has_no_header(self):
        b = Block("wants", [], "WANTS:", 5)
        self.assertEqual((b.render(), b.tokens()), ("", 0))


class ReportTests(unittest.TestCase):
    def test_meta_has_stable_keys_and_the_report_names_each_block(self):
        b = [Block("persona", ["P" * 400], fixed=True), Block("memory", lines("m", 8), "MEMORY:", PRIORITY["memory"])]
        meta = fit(b, 150).meta()
        self.assertEqual(set(meta), {"limit", "used", "overflow", "blocks", "dropped"})
        self.assertEqual(meta["limit"], 150)
        self.assertEqual(set(meta["dropped"]["memory"]), {"items", "tokens", "clipped"})
        text = render_report(meta, context=24576, answer=1600)
        self.assertIn("prompt budget 150 tokens", text)
        self.assertIn("context 24576, 1600 kept for the answer", text)
        self.assertRegex(text, r"PERSONA\s+\d+\s+never cut")
        self.assertRegex(text, r"MEMORY\s+\d+\s+cut \d+ tokens")

    def test_a_turn_that_cut_nothing_says_so_and_no_meta_says_so_too(self):
        meta = fit([Block("persona", ["hi"], fixed=True)], 100).meta()
        self.assertIn("nothing was cut", render_report(meta))
        self.assertEqual(render_report({}), "no prompt has been built in this conversation yet")

    def test_overflow_is_reported(self):
        meta = fit([Block("rules", ["R" * 800], fixed=True)], 50).meta()
        self.assertIn(f"exceed the budget by {meta['overflow']} tokens", render_report(meta))
        self.assertEqual(meta["overflow"], estimate_tokens("R" * 800) - 50)


if __name__ == "__main__":
    unittest.main()
