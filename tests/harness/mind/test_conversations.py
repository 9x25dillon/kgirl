"""Offline tests for mind.conversations: the turn store, full-text search, windows and export.

Everything runs on a temporary sqlite file; no model, no network, no live ~/.kgirl state.
"""

import tempfile
import unittest
from pathlib import Path

import tests.harness._fixtures  # noqa: F401  (sys.path)
from kgirl.harness.mind.conversations import (ConversationStore, clean_tags, same_commitment, tags_in)


class ConversationStoreTestsBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ConversationStore(Path(self.tmp.name) / "conversations.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()


class ConversationStoreTests(ConversationStoreTestsBase):

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


class CommitmentTests(ConversationStoreTestsBase):
    def test_commitments_are_added_numbered_and_not_twinned(self):
        c = self.store.new()
        added = self.store.add_commitments(c, [("I'll check the migration script", "kgirl"),
                                               ("run the tests before merging", "user"), ("  ", "user")], turn=7)
        self.assertEqual([(x.label, x.who, x.status, x.turn) for x in added],
                         [("C1", "kgirl", "open", 7), ("C2", "user", "open", 7)])
        again = self.store.add_commitments(c, [("I will check the migration script.", "kgirl"), ("a new thing", "bob")])
        self.assertEqual([(x.label, x.who) for x in again], [("C3", "")])      # a restatement is not added; who is checked
        self.assertEqual([x.text for x in self.store.get(c).open_commitments],
                         ["I'll check the migration script", "run the tests before merging", "a new thing"])
        self.assertEqual(self.store.add_commitments(999, [("x y z w", "kgirl")]), [])

    def test_close_by_id_label_or_text_and_only_open_ones(self):
        c = self.store.new()
        self.store.add_commitments(c, [("write the changelog entry", "kgirl"), ("benchmark the parser", "user"),
                                       ("rename the helper module", "kgirl")])
        done = self.store.close_commitments(c, [1, "C2", "rename the helper"], by="user")
        self.assertEqual([(x.label, x.status, x.closed_by) for x in done],
                         [("C1", "done", "user"), ("C2", "done", "user"), ("C3", "done", "user")])
        self.assertEqual(self.store.close_commitments(c, ["C1"]), [])           # already done
        self.assertEqual(self.store.get(c).open_commitments, [])
        self.assertEqual(self.store.close_commitments(c, ["xy"]), [])           # too short to name a piece of text
        later = self.store.add_commitments(c, [("write the changelog entry", "kgirl")])
        self.assertEqual(later[0].label, "C4")                                  # a closed one may come back as new

    def test_the_record_is_bounded_and_keeps_open_ones(self):
        c = self.store.new()
        for i in range(25):                                                     # C1..C25; the oldest 5 fall off
            self.store.add_commitments(c, [(f"task number{i} alpha{i} beta{i} gamma{i}", "kgirl")])
        self.store.close_commitments(c, [f"C{i}" for i in range(6, 16)])
        for i in range(25, 40):
            self.store.add_commitments(c, [(f"task number{i} alpha{i} beta{i} gamma{i}", "kgirl")])
        got = self.store.get(c).commitments
        self.assertEqual([x.id for x in got if x.open], list(range(21, 41)))    # the newest 20 open ones
        self.assertEqual(len([x for x in got if not x.open]), 10)
        self.assertEqual([x.id for x in got], sorted(x.id for x in got))

    def test_same_commitment(self):
        self.assertTrue(same_commitment("I'll run the benchmarks", "I will run the benchmark."))
        self.assertFalse(same_commitment("run the benchmarks", "write the changelog"))
        self.assertFalse(same_commitment("", "run it"))

    def test_an_old_database_gains_the_new_columns(self):
        import sqlite3
        path = Path(self.tmp.name) / "old.db"
        db = sqlite3.connect(path)
        db.executescript("CREATE TABLE conversations (id INTEGER PRIMARY KEY, title TEXT DEFAULT '', "
                         "summary TEXT DEFAULT '', threads TEXT DEFAULT '[]', scope TEXT DEFAULT '*', started REAL, "
                         "updated REAL, reflected_turn INTEGER DEFAULT 0);"
                         "INSERT INTO conversations(title, started, updated) VALUES ('before', 1, 1);")
        db.commit()
        db.close()
        store = ConversationStore(path)
        try:
            got = store.get(1)
            self.assertEqual((got.title, got.commitments, got.tags), ("before", [], []))
            store.update(1, tags=["Rust"])
            self.assertEqual(store.get(1).tags, ["rust"])
        finally:
            store.close()

    def test_transcript_and_delete_include_the_new_state(self):
        c = self.store.new("Plan")
        self.store.update(c, summary="s", tags=["rust", "parser"])
        self.store.add_commitments(c, [("write the changelog entry", "kgirl"), ("benchmark the parser", "user")])
        self.store.close_commitments(c, ["C1"])
        art = self.store.save_artifact(c, "draft", "A note", {"stage": "drafting"})
        text = self.store.transcript(c)
        self.assertIn("**Topics:** rust, parser", text)
        self.assertIn("- [x] C1 write the changelog entry (kgirl)", text)
        self.assertIn("- [ ] C2 benchmark the parser (user)", text)
        self.assertIn(f"- #{art} A note (drafting)", text)
        self.store.delete(c)
        self.assertIsNone(self.store.artifact(art))


class TagTests(ConversationStoreTestsBase):
    def test_clean_tags(self):
        self.assertEqual(clean_tags(["Rust", "rust", "error handling", "error_handling", "x", "a" * 40, 7, "ok-1"]),
                         ["rust", "error-handling", "ok-1"])
        self.assertEqual(len(clean_tags([f"tag{i}" for i in range(20)])), 8)
        self.assertEqual(clean_tags(None), [])

    def test_tags_in_a_message_are_whole_words_and_plural_insensitive(self):
        known = {"rust": 2, "parser": 1, "error-handling": 1, "go": 1, "async": 1}
        self.assertEqual(tags_in("How should error handling work in the Rust parsers?", known),
                         ["rust", "parser", "error-handling"])
        self.assertEqual(tags_in("trustworthy, going, asynchronous", known), [])      # not substrings

    def test_related_ranks_by_shared_rare_tags_then_recency(self):
        a = self.store.new("a", scope="kgirl")
        b = self.store.new("b", scope="kgirl")
        c = self.store.new("c", scope="other")
        d = self.store.new("d", scope="kgirl")
        cur = self.store.new("now", scope="kgirl")
        self.store.update(a, tags=["rust", "parser"])
        self.store.update(b, tags=["rust"])
        self.store.update(c, tags=["rust", "parser"])
        self.store.update(d, tags=["docs"])
        self.store.update(cur, tags=["rust", "parser"])
        got = self.store.related(["rust", "parser"], exclude_conv=cur, scope="kgirl")
        self.assertEqual([(x.id, shared) for x, shared in got], [(a, ["rust", "parser"]), (b, ["rust"])])
        got = self.store.related(["rust", "parser"], exclude_conv=cur)           # no scope: every scope qualifies
        self.assertEqual([x.id for x, _ in got], [c, a, b])                      # c is newer than a: ties go to recency
        self.assertEqual(self.store.related(["unknown"]), [])
        self.assertEqual(self.store.related([]), [])
        self.assertEqual(self.store.tag_counts(), {"rust": 4, "parser": 3, "docs": 1})
        self.assertEqual(len(self.store.related(["rust"], limit=2)), 2)


class ArtifactTests(ConversationStoreTestsBase):
    def test_save_replace_list_and_no_effect_on_which_conversation_resumes(self):
        a = self.store.new("first")
        b = self.store.new("second")
        before = self.store.get(a).updated
        art = self.store.save_artifact(a, "draft", "Note", {"stage": "outline", "n": 1})
        self.assertEqual(self.store.save_artifact(a, "draft", "Note v2", {"stage": "drafting"}, artifact_id=art), art)
        got = self.store.artifact(art)
        self.assertEqual((got["conv_id"], got["kind"], got["title"], got["data"]), (a, "draft", "Note v2",
                                                                                      {"stage": "drafting"}))
        self.assertEqual(self.store.get(a).updated, before)                       # chat still resumes the newest
        self.assertEqual(self.store.latest().id, b)
        other = self.store.save_artifact(b, "draft", "Other", {})
        self.assertEqual([x["id"] for x in self.store.artifacts(a, "draft")], [art])
        self.assertEqual({x["id"] for x in self.store.artifacts()}, {art, other})
        self.assertIsNone(self.store.artifact(999))
        self.assertEqual(self.store.save_artifact(a, "draft", "gone", {}, artifact_id=999), other + 1)   # unknown id: new


if __name__ == "__main__":
    unittest.main()
