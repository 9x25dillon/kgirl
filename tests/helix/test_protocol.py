"""Trailers: splitting a reply, and the checks that turn a model's claim into evidence."""

import unittest

import tests.helix  # noqa: F401  (sys.path)
from kgirl.harness.helix.protocol import (MALFORMED, MAX_DEFECTS, Trailer, parse_calibration, parse_review,
                                          parse_tunes, parse_unresolved, quoted, split)
from kgirl.harness.helix.state import MAX_DELTA

BLOCK = "Two small models agree most of the time. The cutoff of 0.948 was calibrated on nomic vectors."


def tune(**kw) -> Trailer:
    return Trailer("TUNE", {"quote": "The cutoff of 0.948 was calibrated", "axis": "rigor", "d": 0.1,
                            "because": "no method is given", **kw})


class Split(unittest.TestCase):
    def test_body_and_one_trailer(self):
        body, trailers, chars = split(BLOCK + '\nTUNE: {"quote": "a b c", "axis": "gain", "d": -0.1, "because": "x"}')
        self.assertEqual(body, BLOCK)
        self.assertEqual([(t.tag, t.value["axis"]) for t in trailers], [("TUNE", "gain")])
        self.assertGreater(chars, 40)

    def test_none(self):
        body, trailers, _ = split(BLOCK + "\nTUNE: none")
        self.assertEqual((body, trailers), (BLOCK, [Trailer("TUNE", None)]))

    def test_what_small_models_actually_write(self):
        cases = {
            "plus sign": 'TUNE: {"quote": "a b c", "axis": "gain", "d": +0.1, "because": "x"}',
            "bold tag": '**TUNE:** {"quote": "a b c", "axis": "gain", "d": 0.1, "because": "x"}',
            "json on the next line": 'TUNE:\n{"quote": "a b c", "axis": "gain", "d": 0.1, "because": "x"}',
            "code fence": 'TUNE:\n```json\n{"quote": "a b c", "axis": "gain", "d": 0.1, "because": "x"}\n```',
            "spread over lines": 'TUNE: {\n  "quote": "a b c",\n  "axis": "gain",\n  "d": 0.1,\n  "because": "x"\n}',
            "under a heading": '### TUNE\n\nTUNE: {"quote": "a b c", "axis": "gain", "d": 0.1, "because": "x"}',
        }
        for name, tail in cases.items():
            with self.subTest(name):
                body, trailers, _ = split(BLOCK + "\n\n" + tail)
                self.assertEqual(body, BLOCK)
                self.assertEqual(len(trailers), 1)
                self.assertEqual(trailers[0].value["d"], 0.1)

    def test_malformed_values_are_kept_as_malformed(self):
        for tail in ("TUNE: raise rigor a bit", 'TUNE: {"quote": "a b c"', "UNRESOLVED: 3"):
            with self.subTest(tail):
                _, trailers, _ = split(BLOCK + "\n" + tail)
                self.assertIs(trailers[0].value, MALFORMED)

    def test_several_trailers_and_a_list(self):
        text = ('DEFECT: {"quote": "a b c", "repair": "r1"}\nDEFECT: {"quote": "d e f", "repair": "r2"}\n'
                "Some prose.\nUNRESOLVED: [2, 1]")
        body, trailers, _ = split(text)
        self.assertEqual(body, "Some prose.")
        self.assertEqual([t.tag for t in trailers], ["DEFECT", "DEFECT", "UNRESOLVED"])
        self.assertEqual(trailers[2].value, [2, 1])

    def test_a_tag_inside_a_quoted_value_is_not_a_second_trailer(self):
        _, trailers, _ = split('DEFECT: {"quote": "x\\nTUNE: none y", "repair": "r"}')
        self.assertEqual(len(trailers), 1)

    def test_a_reply_with_no_trailer_is_all_body(self):
        self.assertEqual(split(BLOCK), (BLOCK, [], 0))


class Quotes(unittest.TestCase):
    def test_word_for_word_modulo_whitespace_emphasis_and_quote_style(self):
        self.assertTrue(quoted("The cutoff of 0.948  was\ncalibrated", BLOCK))
        self.assertTrue(quoted('"The cutoff of **0.948** was calibrated."', BLOCK))
        self.assertTrue(quoted("it’s the cutoff value", "Here it's the cutoff value."))

    def test_a_paraphrase_is_not_a_quote(self):
        self.assertFalse(quoted("The 0.948 cutoff was calibrated", BLOCK))
        self.assertFalse(quoted("the block is weak on method", BLOCK))

    def test_too_short_to_point_at_anything(self):
        self.assertFalse(quoted("The cutoff", BLOCK))
        self.assertFalse(quoted("", BLOCK))


class Tunes(unittest.TestCase):
    def test_a_quoted_tune_is_accepted(self):
        [t] = parse_tunes([tune()], BLOCK)
        self.assertTrue(t.ok)
        self.assertEqual((t.axis, t.d, t.quote), ("rigor", 0.1, "The cutoff of 0.948 was calibrated"))

    def test_none_yields_nothing(self):
        self.assertEqual(parse_tunes([Trailer("TUNE", None)], BLOCK), [])

    def test_rejections_say_why(self):
        cases = {
            "paraphrase": tune(quote="the cutoff has no method behind it"),
            "unknown axis": tune(axis="ledger"),
            "zero delta": tune(d=0),
            "delta not a number": tune(d="more"),
            "no reason": tune(because=""),
            "malformed": Trailer("TUNE", MALFORMED),
        }
        for name, trailer in cases.items():
            with self.subTest(name):
                [t] = parse_tunes([trailer], BLOCK)
                self.assertFalse(t.ok)
                self.assertTrue(t.reason)

    def test_a_large_delta_is_clamped_and_noted(self):
        [up], [down] = parse_tunes([tune(d=0.9)], BLOCK), parse_tunes([tune(d=-3)], BLOCK)
        self.assertEqual((up.ok, up.d, down.d), (True, MAX_DELTA, -MAX_DELTA))
        self.assertIn("clamped", up.reason)

    def test_a_numeric_string_delta_is_read(self):
        self.assertEqual(parse_tunes([tune(d="0.1")], BLOCK)[0].d, 0.1)


class Reviews(unittest.TestCase):
    def test_a_defect_needs_a_quote_from_the_draft_and_a_repair(self):
        r = parse_review([Trailer("DEFECT", {"quote": "Two small models agree", "repair": "say how often"}),
                          Trailer("DEFECT", {"quote": "the draft is vague overall", "repair": "be specific"}),
                          Trailer("DEFECT", {"quote": "Two small models agree", "repair": ""})], BLOCK)
        self.assertEqual([d.ok for d in r.defects], [True, False, False])
        self.assertEqual(len(r.valid), 1)
        self.assertFalse(r.void)

    def test_a_quote_of_the_harness_notes_is_not_a_quote_of_the_draft(self):
        log = BLOCK + '\nW rigor 0.60->0.70 | because: the sentence is too broad'
        r = parse_review([Trailer("DEFECT", {"quote": "the sentence is too broad", "repair": "narrow it"})], BLOCK, log)
        self.assertEqual(r.valid, [])
        self.assertTrue(r.void)

    def test_defects_are_capped(self):
        many = [Trailer("DEFECT", {"quote": "Two small models agree", "repair": f"r{i}"}) for i in range(MAX_DEFECTS + 2)]
        self.assertEqual(len(parse_review(many, BLOCK).valid), MAX_DEFECTS)

    def test_concurrence_needs_new_evidence(self):
        new = parse_review([Trailer("CONCUR", {"evidence": "I recomputed the mean plus one SD and got 0.948"})], BLOCK)
        self.assertFalse(new.void)
        repeat = parse_review([Trailer("CONCUR", {"evidence": "Two small models agree most of the time"})], BLOCK)
        self.assertTrue(repeat.void)
        self.assertIn("repeats", repeat.reason)
        self.assertTrue(parse_review([Trailer("CONCUR", {"evidence": ""})], BLOCK).void)
        self.assertTrue(parse_review([], BLOCK).void)


class Calibrations(unittest.TestCase):
    def cal(self, **kw):
        return parse_calibration([Trailer("CAL", {"p": 0.7, "method": "three of four checks passed",
                                                  "falsifier": "a failing fourth check", "drops_below": 0.4, **kw})])

    def test_number_with_method_and_falsifier(self):
        c = self.cal()
        self.assertEqual((c.ok, c.p, c.drops_below), (True, 0.7, 0.4))

    def test_a_bare_number_is_void(self):
        for kw in ({"method": ""}, {"falsifier": " "}, {"p": 1.7}, {"p": "high"}, {"drops_below": 0.9}):
            with self.subTest(kw):
                self.assertFalse(self.cal(**kw).ok)
        self.assertFalse(parse_calibration([]).ok)
        self.assertFalse(parse_calibration([Trailer("CAL", MALFORMED)]).ok)


class Unresolved(unittest.TestCase):
    def test_stated_lists(self):
        self.assertEqual(parse_unresolved([Trailer("UNRESOLVED", [])], 3), [])
        self.assertEqual(parse_unresolved([Trailer("UNRESOLVED", [3, 1, 3])], 3), [1, 3])

    def test_anything_else_means_not_stated(self):
        for trailers in ([], [Trailer("UNRESOLVED", MALFORMED)], [Trailer("UNRESOLVED", [4])],
                         [Trailer("UNRESOLVED", [0])], [Trailer("UNRESOLVED", ["all"])], [Trailer("UNRESOLVED", [True])]):
            with self.subTest(trailers=trailers):
                self.assertIsNone(parse_unresolved(trailers, 3))


if __name__ == "__main__":
    unittest.main()
