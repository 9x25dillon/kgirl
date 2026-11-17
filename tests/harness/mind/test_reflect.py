"""Offline tests for the pure parts of mind.reflect: verdict reading, explicit memory, parsing, excerpt."""

import unittest

import tests.harness._fixtures  # noqa: F401  (sys.path)
from kgirl.harness.mind.conversations import Turn
from kgirl.harness.mind.reflect import (Reflection, episode_marker, explicit_memory, quick_feedback,
                                        render_excerpt)


def turn(tid, role, content):
    return Turn(tid, 1, role, content, 0.0)


class QuickFeedbackTests(unittest.TestCase):
    def test_positive_reactions(self):
        for text in ("thanks, that works", "Thank you!", "perfect", "great, now add tests",
                     "that works", "works now", "Correct.", "nice job", "exactly what I needed",
                     "ok thanks", "awesome that works"):
            self.assertEqual(quick_feedback(text), "positive", text)

    def test_negative_reactions(self):
        for text in ("no, that's wrong", "nope", "that's wrong", "it doesn't work", "didn't work",
                     "still broken", "you misunderstood", "try again", "not quite",
                     "sorry, that didn't work", "that is not right"):
            self.assertEqual(quick_feedback(text), "negative", text)

    def test_praise_then_same_sentence_complaint_is_negative(self):
        self.assertEqual(quick_feedback("great, but it still fails on empty input"), "negative")

    def test_praise_then_later_sentence_complaint_is_unjudged(self):
        self.assertEqual(quick_feedback("great. But it still fails on empty input."), "none")

    def test_problem_statement_is_not_a_verdict(self):
        self.assertEqual(quick_feedback("the test fails when the input is empty"), "none")

    def test_questions_are_never_verdicts(self):
        self.assertEqual(quick_feedback("does it work?"), "none")
        self.assertEqual(quick_feedback("great?"), "none")

    def test_greetings_are_not_praise(self):
        self.assertEqual(quick_feedback("good morning"), "none")
        self.assertEqual(quick_feedback("nice to meet you"), "none")

    def test_request_not_verdict(self):
        self.assertEqual(quick_feedback("correct this function please"), "none")

    def test_empty_and_none(self):
        self.assertEqual(quick_feedback(""), "none")
        self.assertEqual(quick_feedback(None), "none")


class ExplicitMemoryTests(unittest.TestCase):
    def test_remember_that_is_a_fact(self):
        self.assertEqual(explicit_memory("remember that the staging box is 10.0.0.7"),
                         ("the staging box is 10.0.0.7", "fact"))

    def test_standing_instruction_is_a_preference(self):
        self.assertEqual(explicit_memory("from now on, answer in British English"),
                         ("answer in British English", "preference"))

    def test_remember_to_becomes_preference(self):
        self.assertEqual(explicit_memory("remember to run the tests first"),
                         ("run the tests first", "preference"))

    def test_preference_words_make_it_a_preference(self):
        self.assertEqual(explicit_memory("remember that I prefer tabs over spaces"),
                         ("I prefer tabs over spaces", "preference"))

    def test_question_is_not_an_instruction(self):
        self.assertIsNone(explicit_memory("remember when we fixed the parser?"))

    def test_plain_message_is_none(self):
        self.assertIsNone(explicit_memory("the parser is fixed"))


class ReflectionParseTests(unittest.TestCase):
    def test_non_dict_is_empty(self):
        self.assertEqual(Reflection.parse("nope").summary, "")
        self.assertEqual(Reflection.parse(None).verdicts, {})

    def test_full_record(self):
        ref = Reflection.parse({"summary": "did a thing", "title": "A Title", "facts": ["a fact"],
                                "feedback": {"A1": "positive", "A2": "negative", "A3": "none"}})
        self.assertEqual(ref.summary, "did a thing")
        self.assertEqual(ref.facts, ["a fact"])
        self.assertEqual(ref.verdicts, {"A1": "positive", "A2": "negative"})
        self.assertEqual(ref.feedback, "negative")

    def test_legacy_single_verdict(self):
        ref = Reflection.parse({"summary": "s", "feedback": "positive"})
        self.assertEqual(ref.verdicts, {"LAST": "positive"})

    def test_items_capped_at_six(self):
        ref = Reflection.parse({"summary": "s", "facts": [f"f{i}" for i in range(10)]})
        self.assertEqual(len(ref.facts), 6)

    def test_scalar_item_wrapped(self):
        ref = Reflection.parse({"summary": "s", "facts": "just one"})
        self.assertEqual(ref.facts, ["just one"])


class RenderExcerptTests(unittest.TestCase):
    def test_labels_and_reacted(self):
        turns = [turn(1, "user", "hi"), turn(2, "assistant", "hello"), turn(3, "user", "thanks")]
        text, labels, reacted = render_excerpt(turns, 6000)
        self.assertIn("ASSISTANT [A1]", text)
        self.assertEqual(labels, {"A1": 2})
        self.assertEqual(reacted, ["A1"])

    def test_budget_keeps_newest(self):
        turns = [turn(i, "user", "x" * 100) for i in range(1, 6)]
        text, _, _ = render_excerpt(turns, 250)
        self.assertIn("x" * 100, text)
        self.assertLess(len(text), 400)

    def test_long_content_cut(self):
        text, _, _ = render_excerpt([turn(1, "assistant", "y" * 2000)], 6000)
        self.assertIn("…[cut]", text)


class EpisodeMarkerTests(unittest.TestCase):
    def test_marker_format(self):
        self.assertEqual(episode_marker(7), '{"conv": 7}')


if __name__ == "__main__":
    unittest.main()
