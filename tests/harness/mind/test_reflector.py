"""Offline tests for mind.reflect.Reflector: reflection, proposals and feedback credit, with a fake LLM.

The model is a callable that returns fixed text, the stores are temporary sqlite files: no Ollama, no
network, no live ~/.kgirl state.
"""

import json
import tempfile
import unittest
from pathlib import Path

import tests.harness._fixtures  # noqa: F401  (sys.path)
from kgirl.harness.mind.conversations import ConversationStore
from kgirl.harness.mind.reflect import REFLECT_SYSTEM, Reflection, Reflector, episode_marker
from kgirl.harness.soup import Curator, Soup


class FakeLLM:
    """(system, prompt, max_tokens, temperature) -> the next scripted reply; an Exception is raised instead."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[tuple[str, str, int, float]] = []

    def __call__(self, system: str, prompt: str, max_tokens: int, temperature: float) -> str:
        self.calls.append((system, prompt, max_tokens, temperature))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


class ReflectorCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        home = Path(self.tmp.name)
        self.soup = Soup(home / "soup.db")
        self.curator = Curator(self.soup)
        self.store = ConversationStore(home / "conversations.db")

    def tearDown(self):
        self.store.close()
        self.soup.close()
        self.tmp.cleanup()

    def reflector(self, *replies, every: int = 3) -> tuple[Reflector, FakeLLM]:
        llm = FakeLLM(*replies)
        return Reflector(llm, self.store, self.soup, self.curator, every=every), llm

    def exchange(self, conv: int, question: str, answer: str, recalled: list | None = None) -> tuple[int, int]:
        return (self.store.append(conv, "user", question),
                self.store.append(conv, "assistant", answer, recalled=recalled))


class DueTests(ReflectorCase):
    def test_due_after_enough_user_turns(self):
        ref, _ = self.reflector(every=3)
        conv = self.store.new()
        self.assertFalse(ref.due(conv))
        for i in range(2):
            self.exchange(conv, f"question {i}", f"answer {i}")
        self.assertFalse(ref.due(conv))
        _, last = self.exchange(conv, "question 2", "answer 2")
        self.assertTrue(ref.due(conv))
        self.store.update(conv, reflected_turn=last)
        self.assertFalse(ref.due(conv))

    def test_unknown_conversation_is_never_due(self):
        ref, _ = self.reflector()
        self.assertFalse(ref.due(999))


class CreditAnswerTests(ReflectorCase):
    def test_no_answer_or_no_verdict_records_nothing(self):
        ref, _ = self.reflector()
        conv = self.store.new()
        self.assertIsNone(ref.credit_answer(conv, "positive"))
        self.store.append(conv, "user", "hello")
        self.assertIsNone(ref.credit_answer(conv, "positive"))
        _, answer = self.exchange(conv, "q", "a")
        self.assertIsNone(ref.credit_answer(conv, "none"))
        self.assertIsNone(ref.credit_answer(conv, "meh"))
        self.assertEqual(self.store.turn(answer).verdict, "")

    def test_positive_credits_the_memory_the_answer_was_built_from(self):
        ref, _ = self.reflector()
        fid = self.soup.add("the staging box is 10.0.0.7", kind="fact")
        conv = self.store.new()
        _, answer = self.exchange(conv, "where is staging?", "at 10.0.0.7", recalled=[fid, "not-an-id"])
        self.assertEqual(ref.credit_answer(conv, "positive"), [fid])
        self.assertEqual(self.soup.get(fid).wins, 1.0)
        self.assertEqual(self.soup.get(fid).losses, 0.0)
        turn = self.store.turn(answer)
        self.assertEqual(turn.verdict, "positive")
        self.assertEqual(turn.meta["feedback"], {"verdict": "positive", "source": "reaction", "weight": 1.0})

    def test_an_answer_is_judged_once_unless_overridden(self):
        ref, _ = self.reflector()
        fid = self.soup.add("use tabs in Makefiles", kind="preference")
        conv = self.store.new()
        _, answer = self.exchange(conv, "q", "a", recalled=[fid])
        self.assertEqual(ref.credit_answer(conv, "positive"), [fid])
        self.assertIsNone(ref.credit_answer(conv, "negative"))
        self.assertEqual(self.soup.get(fid).losses, 0.0)
        self.assertEqual(ref.credit_answer(conv, "negative", source="command", override=True), [fid])
        self.assertEqual(self.soup.get(fid).losses, 1.0)
        self.assertEqual(self.store.turn(answer).meta["feedback"]["source"], "command")
        self.assertEqual(self.store.turn(answer).verdict, "negative")

    def test_an_answer_built_from_no_memory_still_takes_a_verdict(self):
        ref, _ = self.reflector()
        conv = self.store.new()
        _, answer = self.exchange(conv, "q", "a")
        self.assertEqual(ref.credit_answer(conv, "negative"), [])
        self.assertEqual(self.store.turn(answer).verdict, "negative")

    def test_the_newest_answer_is_the_default_and_a_tool_step_is_not_an_answer(self):
        ref, _ = self.reflector()
        conv = self.store.new()
        _, first = self.exchange(conv, "q1", "a1")
        self.store.append(conv, "user", "q2")
        step = self.store.append(conv, "assistant", "<tool>repos {}</tool>", meta={"partial": True})
        self.assertEqual(ref.credit_answer(conv, "positive"), [])              # skips the half-turn
        self.assertEqual(self.store.turn(first).verdict, "positive")
        self.assertIsNone(ref.credit_answer(conv, "positive", turn=self.store.turn(step)))

    def test_a_named_turn_is_credited_with_its_weight(self):
        ref, _ = self.reflector()
        fid = self.soup.add("prefer small functions", kind="preference")
        conv = self.store.new()
        _, first = self.exchange(conv, "q1", "a1", recalled=[fid])
        self.exchange(conv, "q2", "a2")
        self.assertEqual(ref.credit_answer(conv, "negative", source="reflection", weight=0.5,
                                           turn=self.store.turn(first)), [fid])
        self.assertEqual(self.soup.get(fid).losses, 0.5)
        self.assertEqual(self.store.turn(first).meta["feedback"]["weight"], 0.5)


class ReflectTests(ReflectorCase):
    RECORD = {"summary": "The user is porting the parser to Rust and asked about error handling.",
              "title": "Parser port",
              "threads": ["decide between anyhow and thiserror"],
              "facts": ["The user's parser project targets Rust 1.80."],
              "preferences": ["The user wants answers without filler."],
              "lessons": ["Showing a minimal failing example before the fix worked well for the parser port."],
              "mistakes": ["Do not suggest nightly-only features for the parser port."]}

    def test_nothing_to_reflect_on(self):
        ref, llm = self.reflector(every=3)
        self.assertIsNone(ref.reflect(999))
        conv = self.store.new()
        self.assertIsNone(ref.reflect(conv, force=True))                       # no turns at all
        self.exchange(conv, "q", "a")
        self.assertIsNone(ref.reflect(conv))                                   # fewer than `every` user turns
        self.assertEqual(llm.calls, [])

    def test_reflection_updates_the_conversation_and_proposes_memory(self):
        ref, llm = self.reflector(self.RECORD)
        conv = self.store.new(scope="parser")
        _, answer = self.exchange(conv, "how should errors work in the port?", "use a Result type")
        out = ref.reflect(conv, force=True)
        self.assertEqual(out.summary, self.RECORD["summary"])

        got = self.store.get(conv)
        self.assertEqual(got.summary, self.RECORD["summary"])
        self.assertEqual(got.title, "Parser port")
        self.assertEqual(got.threads, ["decide between anyhow and thiserror"])
        self.assertEqual(got.reflected_turn, answer)

        self.assertEqual([how for _, how in out.proposed], ["staged"] * 4)
        frags = [self.soup.get(fid) for fid, _ in out.proposed]
        self.assertEqual([f.kind for f in frags], ["fact", "preference", "abstraction", "antipattern"])
        self.assertEqual([f.scope for f in frags], ["*", "*", "parser", "parser"])
        for f in frags:
            self.assertEqual(f.status, "staged")              # one conversation only suggests a habit
            self.assertEqual(f.source, f"reflection:conv{conv}")
            self.assertEqual(f.tags, "conversation")
        self.assertEqual(frags[0].text, self.RECORD["facts"][0])

    def test_the_model_sees_the_previous_summary_and_a_labelled_excerpt(self):
        ref, llm = self.reflector(self.RECORD, self.RECORD)
        conv = self.store.new()
        self.store.append(conv, "user", "list the repos")
        self.store.append(conv, "assistant", "checking\n<tool>repos {}</tool>", meta={"partial": True})
        self.store.append(conv, "tool", "[repos] kgirl (120 files)")
        self.store.append(conv, "assistant", "there is one repo: kgirl")
        ref.reflect(conv, force=True)
        system, prompt, max_tokens, temperature = llm.calls[0]
        self.assertEqual(system, REFLECT_SYSTEM)
        self.assertEqual((max_tokens, temperature), (900, 0.1))
        self.assertEqual(prompt, "PREVIOUS SUMMARY:\n(none: new conversation)\n\nCONVERSATION EXCERPT:\n"
                                 "USER: list the repos\n\nASSISTANT [A1]: there is one repo: kgirl")

        self.exchange(conv, "and how big is it?", "120 files")
        ref.reflect(conv, force=True)
        prompt = llm.calls[1][1]
        self.assertTrue(prompt.startswith(f"PREVIOUS SUMMARY:\n{self.RECORD['summary']}\n\n"), prompt)
        self.assertNotIn("list the repos", prompt)             # only the turns since the last reflection
        self.assertIn("USER: and how big is it?", prompt)

    def test_an_existing_title_is_kept(self):
        ref, _ = self.reflector(self.RECORD)
        conv = self.store.new("Named by the user")
        self.exchange(conv, "q", "a")
        ref.reflect(conv, force=True)
        self.assertEqual(self.store.get(conv).title, "Named by the user")

    def test_a_failing_or_unusable_model_changes_nothing(self):
        ref, llm = self.reflector(RuntimeError("ollama is down"), "I could not produce JSON, sorry.",
                                  {"title": "no summary here", "facts": ["a fact without a summary"]})
        conv = self.store.new()
        self.exchange(conv, "q", "a")
        for _ in range(3):
            self.assertIsNone(ref.reflect(conv, force=True))
        self.assertEqual(len(llm.calls), 3)
        got = self.store.get(conv)
        self.assertEqual((got.summary, got.title, got.reflected_turn), ("", "", 0))
        self.assertEqual(self.soup.all(), [])

    def test_reasoning_and_fences_around_the_json_are_tolerated(self):
        reply = "the user greeted me\n</think>\n```json\n" + json.dumps({"summary": "A greeting."}) + "\n```"
        ref, _ = self.reflector(reply)
        conv = self.store.new()
        self.exchange(conv, "hi", "hello")
        self.assertEqual(ref.reflect(conv, force=True).summary, "A greeting.")


class EpisodeTests(ReflectorCase):
    def episodes(self, status: str = "active"):
        return self.soup.all(status=status, kind="episode")

    def test_one_active_episode_per_conversation_replaced_as_the_summary_grows(self):
        ref, _ = self.reflector({"summary": "First summary.", "title": "Thread"},
                                {"summary": "Second, longer summary.", "title": "Thread"},
                                {"summary": "Second, longer summary.", "title": "Thread"})
        conv = self.store.new(scope="kgirl")
        self.exchange(conv, "q1", "a1")
        ref.reflect(conv, force=True)
        first = self.episodes()
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].text, f"conversation #{conv} Thread: First summary.")
        self.assertEqual(first[0].data, episode_marker(conv))
        self.assertEqual((first[0].scope, first[0].source, first[0].tags), ("kgirl", "conversation", "conversation"))

        self.exchange(conv, "q2", "a2")
        ref.reflect(conv, force=True)
        second = self.episodes()
        self.assertEqual([f.text for f in second], [f"conversation #{conv} Thread: Second, longer summary."])
        self.assertEqual([f.id for f in self.episodes("retired")], [first[0].id])

        self.exchange(conv, "q3", "a3")
        ref.reflect(conv, force=True)                                           # the same summary again
        self.assertEqual([f.id for f in self.episodes()], [second[0].id])
        self.assertEqual(len(self.episodes("retired")), 1)

    def test_episodes_of_different_conversations_do_not_replace_each_other(self):
        ref, _ = self.reflector({"summary": "About sorting."}, {"summary": "About parsing."})
        a, b = self.store.new(), self.store.new()
        self.exchange(a, "q", "a")
        self.exchange(b, "q", "a")
        ref.reflect(a, force=True)
        ref.reflect(b, force=True)
        self.assertEqual(sorted(f.data for f in self.episodes()), [episode_marker(a), episode_marker(b)])


class ReflectionFeedbackTests(ReflectorCase):
    """The model's reading of the excerpt credits only answers the user reacted to, at half weight, once."""

    def test_only_an_answer_a_user_message_follows_is_credited(self):
        ref, _ = self.reflector({"summary": "s", "feedback": {"A1": "positive", "A2": "positive", "A9": "negative"}})
        fid = self.soup.add("sorted() returns a new list", kind="fact")
        conv = self.store.new()
        _, a1 = self.exchange(conv, "how do I sort?", "use sorted()", recalled=[fid])
        _, a2 = self.exchange(conv, "and in place?", "use list.sort()", recalled=[fid])
        out = ref.reflect(conv, force=True)
        self.assertEqual(out.credited, [a1])                   # A2 has no reaction yet; A9 does not exist
        self.assertEqual(self.store.turn(a1).meta["feedback"],
                         {"verdict": "positive", "source": "reflection", "weight": 0.5})
        self.assertEqual(self.store.turn(a2).verdict, "")
        self.assertEqual(self.soup.get(fid).wins, 0.5)

    def test_the_answer_before_the_excerpt_is_shown_so_its_reaction_can_be_read(self):
        ref, llm = self.reflector({"summary": "first pass"}, {"summary": "second pass", "feedback": {"A1": "negative"}})
        fid = self.soup.add("the deploy script lives in ops/deploy.sh", kind="fact")
        conv = self.store.new()
        _, a1 = self.exchange(conv, "where is the deploy script?", "in scripts/deploy.sh", recalled=[fid])
        ref.reflect(conv, force=True)
        _, a2 = self.exchange(conv, "no, that's wrong, it moved", "sorry: it is in ops/deploy.sh")
        out = ref.reflect(conv, force=True)
        self.assertIn("ASSISTANT [A1]: in scripts/deploy.sh\n\nUSER: no, that's wrong, it moved", llm.calls[1][1])
        self.assertEqual(out.credited, [a1])
        self.assertEqual(self.store.turn(a1).verdict, "negative")
        self.assertEqual(self.soup.get(fid).losses, 0.5)
        self.assertEqual(self.store.get(conv).reflected_turn, a2)

    def test_the_users_own_verdict_is_not_overwritten(self):
        ref, _ = self.reflector({"summary": "s", "feedback": {"A1": "negative"}})
        fid = self.soup.add("the build uses make", kind="fact")
        conv = self.store.new()
        _, a1 = self.exchange(conv, "how do I build?", "run make", recalled=[fid])
        self.store.append(conv, "user", "thanks, that works")
        self.assertEqual(ref.credit_answer(conv, "positive"), [fid])            # the reaction, read by the chat loop
        out = ref.reflect(conv, force=True)
        self.assertEqual(out.credited, [])
        self.assertEqual(self.store.turn(a1).verdict, "positive")
        self.assertEqual((self.soup.get(fid).wins, self.soup.get(fid).losses), (1.0, 0.0))

    def test_an_already_judged_earlier_answer_is_not_shown_again(self):
        ref, llm = self.reflector({"summary": "first"}, {"summary": "second", "feedback": {"A1": "negative"}})
        conv = self.store.new()
        _, a1 = self.exchange(conv, "q1", "first answer")
        ref.reflect(conv, force=True)
        ref.credit_answer(conv, "positive")
        _, a2 = self.exchange(conv, "thanks! next question", "second answer")
        out = ref.reflect(conv, force=True)
        self.assertNotIn("first answer", llm.calls[1][1])
        self.assertEqual(out.credited, [])                     # A1 is now the second answer: no reaction yet
        self.assertEqual(self.store.turn(a1).verdict, "positive")
        self.assertEqual(self.store.turn(a2).verdict, "")

    def test_the_older_single_verdict_form_means_the_last_answer_reacted_to(self):
        ref, _ = self.reflector({"summary": "s", "feedback": "positive"})
        conv = self.store.new()
        _, a1 = self.exchange(conv, "q1", "a1")
        _, a2 = self.exchange(conv, "q2", "a2")
        self.store.append(conv, "user", "q3")
        out = ref.reflect(conv, force=True)
        self.assertEqual(out.credited, [a2])
        self.assertEqual(self.store.turn(a1).verdict, "")
        self.assertEqual(self.store.turn(a2).verdict, "positive")


class CommitmentReflectionTests(ReflectorCase):
    def test_parse_reads_commitments_closed_labels_and_tags(self):
        r = Reflection.parse({"summary": "s", "commitments": [{"text": "I'll run the benchmark", "who": "KGIRL"},
                                                               "next time, check the lockfile",
                                                               {"text": "x", "who": "somebody"}, {"who": "user"}, 5],
                              "closed": ["C1", " c2 ", 3, {"x": 1}], "tags": ["Rust", "error handling", "a"]})
        self.assertEqual(r.commitments, [("I'll run the benchmark", "kgirl"), ("next time, check the lockfile", ""),
                                         ("x", "")])
        self.assertEqual(r.closed, ["C1", "c2", "3"])
        self.assertEqual(r.tags, ["rust", "error-handling"])
        self.assertEqual((Reflection.parse({"summary": "s"}).commitments, Reflection.parse({"summary": "s"}).tags),
                         ([], []))
        self.assertEqual(len(Reflection.parse({"commitments": [f"do thing {i}" for i in range(9)]}).commitments), 6)

    def test_reflection_stores_commitments_and_tags_and_later_closes_them(self):
        first = {"summary": "Porting the parser.", "title": "Parser port", "tags": ["rust", "parser"],
                 "threads": ["pick an error type"],
                 "commitments": [{"text": "I'll benchmark the new tokenizer", "who": "kgirl"},
                                 {"text": "next time I will read the changelog first", "who": "user"}]}
        second = {"summary": "Benchmarked.", "tags": ["rust", "benchmark"], "threads": [],
                  "commitments": [{"text": "I will benchmark the new tokenizer.", "who": "kgirl"}],
                  "closed": ["C1"]}
        ref, llm = self.reflector(first, second)
        conv = self.store.new()
        _, last = self.exchange(conv, "port the parser", "ok, I'll benchmark the tokenizer next")
        ref.reflect(conv, force=True)
        got = self.store.get(conv)
        self.assertEqual([(c.label, c.who, c.status, c.turn) for c in got.commitments],
                         [("C1", "kgirl", "open", last), ("C2", "user", "open", last)])
        self.assertEqual(got.tags, ["rust", "parser"])

        self.exchange(conv, "how did the benchmark go?", "done: 12% faster")
        ref.reflect(conv, force=True)
        prompt = llm.calls[1][1]                                           # the model is shown what is still open
        self.assertIn("OPEN THREADS SO FAR:\n- pick an error type", prompt)
        self.assertIn("C1 (kgirl): I'll benchmark the new tokenizer\nC2 (user): next time I will read the changelog first",
                      prompt)
        self.assertLess(prompt.index("OPEN COMMITMENTS"), prompt.index("CONVERSATION EXCERPT"))
        got = self.store.get(conv)
        self.assertEqual([(c.label, c.status, c.closed_by) for c in got.commitments],
                         [("C1", "done", "reflection"), ("C2", "open", "")])    # the restated C1 was not added twice
        self.assertEqual(got.tags, ["rust", "benchmark", "parser"])
        self.assertEqual(got.threads, [])

    def test_the_prompt_has_no_open_state_block_when_nothing_is_open(self):
        ref, llm = self.reflector({"summary": "s"})
        conv = self.store.new()
        self.exchange(conv, "hello", "hi")
        ref.reflect(conv, force=True)
        self.assertNotIn("OPEN", llm.calls[0][1])
        self.assertIn('"commitments"', REFLECT_SYSTEM)
        self.assertIn('"closed"', REFLECT_SYSTEM)
        self.assertIn('"tags"', REFLECT_SYSTEM)

    def test_a_failed_reflection_leaves_commitments_alone(self):
        ref, _ = self.reflector(RuntimeError("model down"))
        conv = self.store.new()
        self.store.add_commitments(conv, [("write the changelog entry", "kgirl")])
        self.exchange(conv, "hello", "hi")
        self.assertIsNone(ref.reflect(conv, force=True))
        self.assertEqual(len(self.store.get(conv).open_commitments), 1)


class ProposalDedupeTests(ReflectorCase):
    FACT = "The user's parser project targets Rust 1.80."

    def propose_in(self, ref, conv, fact):
        ref.llm.replies.append({"summary": "s", "facts": [fact]})
        self.exchange(conv, "q", "a")
        return ref.reflect(conv, force=True).proposed[0]

    def test_a_restatement_in_another_conversation_reinforces_instead_of_staging_a_twin(self):
        ref, _ = self.reflector()
        a, b = self.store.new(), self.store.new()
        fid, how = self.propose_in(ref, a, self.FACT)
        self.assertEqual(how, "staged")
        self.assertEqual(self.soup.get(fid).support, 1)
        # a paraphrase that is not a near-copy, but says the same thing
        fid2, how2 = self.propose_in(ref, b, "The parser project the user is building targets Rust 1.80.")
        self.assertEqual((fid2, how2), (fid, "reinforced"))
        self.assertEqual(self.soup.get(fid).support, 2)
        self.assertEqual(len(self.soup.all(kind="fact")), 1)
        self.assertEqual(self.soup.get(fid).status, "staged")             # support 2 alone promotes nothing: the curator decides

    def test_the_same_conversation_saying_it_again_is_not_independent_evidence(self):
        ref, _ = self.reflector()
        a = self.store.new()
        fid, _ = self.propose_in(ref, a, self.FACT)
        again = self.propose_in(ref, a, self.FACT)
        self.assertEqual(again, (fid, "seen"))
        self.assertEqual(self.soup.get(fid).support, 1)
        self.assertEqual(self.soup.get(fid).wins, 1.0)                    # no second win either

    def test_a_third_conversation_counts_once_and_a_second_pass_of_the_second_does_not(self):
        ref, _ = self.reflector()
        a, b, c = self.store.new(), self.store.new(), self.store.new()
        fid, _ = self.propose_in(ref, a, self.FACT)
        self.propose_in(ref, b, self.FACT)
        self.assertEqual(self.propose_in(ref, b, self.FACT), (fid, "seen"))
        self.assertEqual(self.soup.get(fid).support, 2)
        self.assertEqual(self.propose_in(ref, c, self.FACT), (fid, "reinforced"))
        self.assertEqual(self.soup.get(fid).support, 3)

    def test_a_different_fact_is_staged_not_folded_into_an_unrelated_one(self):
        ref, _ = self.reflector()
        a, b = self.store.new(), self.store.new()
        fid, _ = self.propose_in(ref, a, self.FACT)
        other, how = self.propose_in(ref, b, "The user's staging box is called box7 and runs Debian 12.")
        self.assertNotEqual(other, fid)
        self.assertEqual(how, "staged")
        self.assertEqual(self.soup.get(fid).support, 1)


if __name__ == "__main__":
    unittest.main()
