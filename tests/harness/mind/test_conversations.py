"""Offline tests for mind.conversations: the turn store, full-text search, windows and export.

Everything runs on a temporary sqlite file; no model, no network, no live ~/.kgirl state.
"""

import tempfile
import unittest
from pathlib import Path

import tests.harness._fixtures  # noqa: F401  (sys.path)
from kgirl.harness.mind.conversations import ConversationStore


class ConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ConversationStore(Path(self.tmp.name) / "conversations.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_new_get_latest_list(self):
        a = self.store.new("first", scope="kgirl")
        b = self.store.new("second", scope="other")
        self.assertEqual(self.store.get(a).title, "first")
        self.assertEqual(self.store.get(a).scope, "kgirl")
        self.assertEqual(self.store.latest().id, b)
        self.assertEqual(self.store.latest("kgirl").id, a)
        self.assertEqual([c.id for c in self.store.list()], [b, a])
        self.assertIsNone(self.store.get(999))

    def test_update_and_threads_cap(self):
        c = self.store.new()
        self.store.update(c, summary="a summary", title="t", threads=[f"q{i}" for i in range(20)],
                          reflected_turn=5)
        got = self.store.get(c)
        self.assertEqual(got.summary, "a summary")
        self.assertEqual(got.title, "t")
        self.assertEqual(len(got.threads), 12)
        self.assertEqual(got.reflected_turn, 5)

    def test_append_roles_and_searchable(self):
        c = self.store.new()
        self.store.append(c, "user", "the staging box is at 10.0.0.7")
        self.store.append(c, "assistant", "partial thought", meta={"partial": True})
        self.store.append(c, "tool", "staging result text")
        a = self.store.append(c, "assistant", "the staging box is ready", recalled=[1, 2])
        self.assertEqual(self.store.count(c), 4)
        self.assertEqual(self.store.turn(a).recalled, [1, 2])
        with self.assertRaises(ValueError):
            self.store.append(c, "bogus", "x")
        hits = self.store.search("staging")
        self.assertTrue(hits)
        self.assertTrue(all(t.role in ("user", "assistant") and not t.partial for t in hits))
        self.assertFalse(any(t.role == "tool" for t in hits))

    def test_turn_properties_and_set_meta(self):
        c = self.store.new()
        p = self.store.append(c, "assistant", "thinking", meta={"partial": True})
        a = self.store.append(c, "assistant", "the answer")
        self.assertTrue(self.store.turn(p).partial)
        self.assertFalse(self.store.turn(p).is_answer)
        self.assertTrue(self.store.turn(a).is_answer)
        self.assertEqual(self.store.turn(a).verdict, "")
        self.store.set_meta(a, feedback={"verdict": "positive", "source": "command"})
        self.assertEqual(self.store.turn(a).verdict, "positive")

    def test_last_and_last_answer_skip_partial(self):
        c = self.store.new()
        self.store.append(c, "user", "q")
        a1 = self.store.append(c, "assistant", "first answer")
        self.store.append(c, "assistant", "half", meta={"partial": True})
        self.assertEqual(self.store.last_answer(c).id, a1)
        self.assertEqual(self.store.last(c, "user").content, "q")
        self.assertIsNone(self.store.last_answer(c, before_id=a1))

    def test_turns_since_and_roles(self):
        c = self.store.new()
        u = self.store.append(c, "user", "q")
        a = self.store.append(c, "assistant", "a")
        self.assertEqual([t.id for t in self.store.turns(c, since_id=u)], [a])
        self.assertEqual([t.id for t in self.store.turns(c, roles=("user",))], [u])

    def test_exchanges_group_steps_and_answer(self):
        c = self.store.new()
        self.store.append(c, "tool", "orphan slash output")
        u1 = self.store.append(c, "user", "q1")
        self.store.append(c, "assistant", "calling", meta={"partial": True})
        self.store.append(c, "tool", "result")
        a1 = self.store.append(c, "assistant", "answer1")
        u2 = self.store.append(c, "user", "q2")
        ex = self.store.exchanges(c)
        self.assertEqual(len(ex), 2)
        self.assertEqual(ex[0].user.id, u1)
        self.assertEqual(ex[0].answer.id, a1)
        self.assertEqual(len(ex[0].steps), 2)
        self.assertEqual(ex[1].user.id, u2)
        self.assertIsNone(ex[1].answer)

    def test_window_newest_always_included_and_ordered(self):
        c = self.store.new()
        ids = [self.store.append(c, "user", f"message number {i}") for i in range(5)]
        tiny = self.store.window(c, budget_tokens=1)
        self.assertEqual(tiny[-1].id, ids[-1])
        wide = self.store.window(c, budget_tokens=100000)
        self.assertEqual([t.id for t in wide], ids)

    def test_search_exclude_conv(self):
        a = self.store.new()
        b = self.store.new()
        self.store.append(a, "user", "the parser uses a recursive descent")
        self.store.append(b, "user", "the parser uses a recursive descent")
        hits = self.store.search("parser recursive", exclude_conv=a)
        self.assertTrue(hits)
        self.assertTrue(all(t.conv_id == b for t in hits))

    def test_context_save_and_read(self):
        c = self.store.new()
        a = self.store.append(c, "assistant", "answer")
        self.store.save_context(a, "SYSTEM PROMPT")
        self.assertEqual(self.store.context(a), "SYSTEM PROMPT")
        self.assertIsNone(self.store.context(999))

    def test_transcript_markdown(self):
        c = self.store.new("My Thread")
        self.store.update(c, summary="the summary", threads=["open question"])
        self.store.append(c, "user", "hello there")
        self.store.append(c, "tool", "tool output")
        text = self.store.transcript(c)
        self.assertIn("# My Thread", text)
        self.assertIn("**Summary:** the summary", text)
        self.assertIn("open question", text)
        self.assertIn("hello there", text)
        self.assertIn("## Tool result", text)
        self.assertEqual(self.store.transcript(999), "")

    def test_delete_cascades(self):
        c = self.store.new()
        self.store.append(c, "user", "searchable unique phrase zebra")
        self.store.append(c, "assistant", "answer")
        self.assertEqual(self.store.delete(c), 2)
        self.assertIsNone(self.store.get(c))
        self.assertEqual(self.store.count(c), 0)
        self.assertFalse(self.store.search("zebra"))


if __name__ == "__main__":
    unittest.main()
