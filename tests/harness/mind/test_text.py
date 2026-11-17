"""Offline tests for mind.text: reading what local models write (no model, no network)."""

import unittest

import tests.harness._fixtures  # noqa: F401  (sys.path)
from kgirl.harness.mind.text import code_blocks, first_json, one_line, python_code, strip_reasoning


class StripReasoningTests(unittest.TestCase):
    def test_drops_think_block(self):
        self.assertEqual(strip_reasoning("thinking hard\n</think>\nthe answer"), "the answer")

    def test_drops_paired_think(self):
        self.assertEqual(strip_reasoning("<think>a\nb</think>\nanswer"), "answer")

    def test_drops_unopened_think(self):
        self.assertEqual(strip_reasoning("a\nb\n</think>answer"), "answer")

    def test_plain_text_unchanged(self):
        self.assertEqual(strip_reasoning("  hello world  "), "hello world")

    def test_none_is_empty(self):
        self.assertEqual(strip_reasoning(None), "")


class CodeBlockTests(unittest.TestCase):
    def test_python_filter_and_longest(self):
        text = "```python\nx = 1\n```\n```js\nlet y = 2\n```\n```python\na = 1\nb = 2\n```"
        blocks = code_blocks(text, "python")
        self.assertEqual(len(blocks), 2)
        self.assertIn("a = 1", python_code(text))
        self.assertNotIn("let y", python_code(text))

    def test_untagged_counts(self):
        self.assertEqual(code_blocks("```\nplain\n```", "python"), ["plain\n"])

    def test_python_code_whole_reply_when_it_parses(self):
        self.assertEqual(python_code("x = 1\n"), "x = 1\n")

    def test_python_code_empty_when_nothing_parses(self):
        self.assertEqual(python_code("this is not python ("), "")


class FirstJsonTests(unittest.TestCase):
    def test_bare_object(self):
        self.assertEqual(first_json('here you go: {"a": 1} thanks'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(first_json('```json\n{"a": [1, 2]}\n```'), {"a": [1, 2]})

    def test_array(self):
        self.assertEqual(first_json("[1, 2, 3]"), [1, 2, 3])

    def test_none_when_absent(self):
        self.assertIsNone(first_json("no json here"))

    def test_skips_reasoning(self):
        self.assertEqual(first_json('ignore\n</think>{"ok": true}'), {"ok": True})


class OneLineTests(unittest.TestCase):
    def test_collapses_whitespace(self):
        self.assertEqual(one_line("  a   b\n c  "), "a b c")

    def test_truncates_with_ellipsis(self):
        out = one_line("x" * 50, 10)
        self.assertEqual(len(out), 10)
        self.assertTrue(out.endswith("…"))

    def test_short_unchanged(self):
        self.assertEqual(one_line("short", 100), "short")


if __name__ == "__main__":
    unittest.main()
