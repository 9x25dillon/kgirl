"""Offline tests for mind.evolve: the consolidation pass, the user profile and the fine-tune export.

Temporary sqlite stores and a fake LLM that answers by system prompt: no Ollama, no network, no live
~/.kgirl state.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tests.harness._fixtures  # noqa: F401  (sys.path)
from kgirl.harness.mind.conversations import ConversationStore
from kgirl.harness.mind.evolve import (PROFILE_SYSTEM, EvolveReport, evolve, export_chat_sft, iter_chat_sft, preview,
                                       rewrite_profile, verdict_on)
from kgirl.harness.mind.reflect import REFLECT_SYSTEM
from kgirl.harness.soup import Curator, Soup
from kgirl.harness.soup.curator import CurationPolicy
from kgirl.harness.util import now


class ModelByPrompt:
    """Reflection requests get `reflection` as JSON, profile requests get `profile` (or raise it)."""

    def __init__(self, profile="- Works on kgirl.", reflection=None):
        self.profile, self.reflection = profile, reflection or {"summary": "A short conversation."}
        self.calls: list[tuple[str, str, int, float]] = []

    def __call__(self, system, prompt, max_tokens, temperature):
        self.calls.append((system, prompt, max_tokens, temperature))
        if system == REFLECT_SYSTEM:
            return json.dumps(self.reflection)
        if system != PROFILE_SYSTEM:
            raise AssertionError(f"unexpected system prompt: {system[:60]}")
        if isinstance(self.profile, Exception):
            raise self.profile
        return self.profile

    def asked(self, system) -> list[str]:
        return [prompt for s, prompt, _, _ in self.calls if s == system]


class EvolveCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.soup = Soup(self.home / "soup.db")
        self.curator = Curator(self.soup)
        self.store = ConversationStore(self.home / "conversations.db")

    def tearDown(self):
        self.store.close()
        self.soup.close()
        self.tmp.cleanup()

    def exchange(self, conv: int, question: str, answer: str) -> tuple[int, int]:
        return self.store.append(conv, "user", question), self.store.append(conv, "assistant", answer)

    def approve(self, turn_id: int, verdict: str = "positive") -> None:
        self.store.set_meta(turn_id, feedback={"verdict": verdict, "source": "command", "weight": 1.0})

    def notes(self, status: str = "active"):
        return [f for f in self.soup.all(status=status, kind="note") if f.tags == "profile"]


class VerdictTests(EvolveCase):
    def test_read_from_the_next_message_when_none_is_recorded(self):
        conv = self.store.new()
        self.exchange(conv, "how do I sort a list?", "use sorted(xs)")
        self.exchange(conv, "thanks, that works", "glad it helped")
        self.exchange(conv, "no, that's wrong", "sorry, let me look again")
        self.store.append(conv, "user", "still waiting")
        done = self.store.exchanges(conv)
        self.assertEqual([verdict_on(done, i) for i in range(4)], ["positive", "negative", "none", "none"])

    def test_a_recorded_verdict_wins_over_the_next_message(self):
        conv = self.store.new()
        _, answer = self.exchange(conv, "how do I sort a list?", "use xs.sort() and print the result")
        self.exchange(conv, "thanks, that works", "glad it helped")
        self.approve(answer, "negative")
        self.assertEqual(verdict_on(self.store.exchanges(conv), 0), "negative")

    def test_the_last_answer_has_no_reaction_yet(self):
        conv = self.store.new()
        self.exchange(conv, "q", "a")
        self.assertEqual(verdict_on(self.store.exchanges(conv), 0), "none")


class ExportTests(EvolveCase):
    def rows(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def conversation(self) -> tuple[int, int, int]:
        conv = self.store.new("Sorting")
        _, a1 = self.exchange(conv, "how do I sort a list?", "use sorted(xs)")
        self.store.append(conv, "user", "thanks, that works. which files are here?")
        self.store.append(conv, "assistant", "<tool>ls {}</tool>", meta={"partial": True, "tool": "ls"})
        self.store.append(conv, "tool", "[ls] a.py  (10 B)")
        a2 = self.store.append(conv, "assistant", "there is one file: a.py")
        self.approve(a2)
        self.store.save_context(a2, "SYSTEM PROMPT")
        self.exchange(conv, "what does a.py do?", "it defines main()")       # nobody reacted to this one
        return conv, a1, a2

    def test_one_row_per_approved_answer_in_the_form_the_model_is_used_in(self):
        conv, a1, a2 = self.conversation()
        out = self.home / "export" / "deep" / "chat.jsonl"
        self.assertEqual(export_chat_sft(self.store, out), 2)
        first, second = self.rows(out)
        self.assertEqual(first, {"messages": [{"role": "user", "content": "how do I sort a list?"},
                                              {"role": "assistant", "content": "use sorted(xs)"}],
                                 "conv": conv, "turn": a1})
        self.assertEqual(second, {
            "messages": [{"role": "user", "content": "how do I sort a list?"},
                         {"role": "assistant", "content": "use sorted(xs)"},
                         {"role": "user", "content": "thanks, that works. which files are here?"},
                         {"role": "assistant", "content": "<tool>ls {}</tool>"},
                         {"role": "user", "content": "TOOL RESULT [ls] a.py  (10 B)"},
                         {"role": "assistant", "content": "there is one file: a.py"}],
            "conv": conv, "turn": a2, "system": "SYSTEM PROMPT"})

    def test_context_exchanges_bounds_the_history(self):
        self.conversation()
        out = self.home / "chat.jsonl"
        self.assertEqual(export_chat_sft(self.store, out, context_exchanges=0), 2)
        self.assertEqual(self.rows(out)[1]["messages"][0],
                         {"role": "user", "content": "thanks, that works. which files are here?"})

    def test_unapproved_and_corrected_answers_are_left_out(self):
        conv = self.store.new()
        _, wrong = self.exchange(conv, "where is the deploy script?", "in scripts/deploy.sh")
        self.approve(wrong, "negative")
        self.exchange(conv, "thanks anyway", "sorry about that")
        out = self.home / "chat.jsonl"
        self.assertEqual(export_chat_sft(self.store, out), 0)
        self.assertEqual(out.read_text(encoding="utf-8"), "")

    def test_an_unanswered_message_is_not_used_as_context(self):
        conv = self.store.new()
        self.store.append(conv, "user", "are you there?")
        _, answer = self.exchange(conv, "hello again", "yes, I am here")
        self.approve(answer)
        out = self.home / "chat.jsonl"
        self.assertEqual(export_chat_sft(self.store, out), 1)
        self.assertEqual(self.rows(out)[0]["messages"], [{"role": "user", "content": "hello again"},
                                                         {"role": "assistant", "content": "yes, I am here"}])

    def test_the_file_is_rewritten_each_time(self):
        self.conversation()
        out = self.home / "chat.jsonl"
        export_chat_sft(self.store, out)
        export_chat_sft(self.store, out)
        self.assertEqual(len(self.rows(out)), 2)


class ProfileTests(EvolveCase):
    def test_nothing_known_means_no_profile_and_no_model_call(self):
        llm = ModelByPrompt()
        self.assertIsNone(rewrite_profile(llm, self.soup))
        self.soup.add("repo kgirl: languages python; hub modules harness", kind="fact", source="atlas")
        self.soup.add("The user may prefer tabs.", kind="preference", status="staged")
        self.assertIsNone(rewrite_profile(llm, self.soup))                    # index facts and proposals don't count
        self.assertEqual(llm.calls, [])

    def test_the_profile_is_written_from_active_memory(self):
        pref = self.soup.add("The user wants answers without filler.", kind="preference")
        fact = self.soup.add("The user's GPU is an RTX 2080 SUPER with 8 GB.", kind="fact", source="user")
        self.soup.add("repo kgirl: languages python; hub modules harness", kind="fact", source="atlas")
        self.soup.add("The user may prefer tabs.", kind="preference", status="staged")
        self.soup.add("conversation #1 Parser port: the parser was ported to Rust.", kind="episode", data='{"conv": 1}')
        llm = ModelByPrompt("listing what I know\n</think>\n- Wants no filler.\n- Has an 8 GB GPU.")
        fid = rewrite_profile(llm, self.soup)
        self.assertEqual(llm.calls, [(PROFILE_SYSTEM,
                                      f"PREFERENCE (#{pref}): The user wants answers without filler.\n"
                                      f"FACT (#{fact}): The user's GPU is an RTX 2080 SUPER with 8 GB.\n"
                                      "EPISODE: conversation #1 Parser port: the parser was ported to Rust.",
                                      700, 0.2)])
        note = self.soup.get(fid)
        self.assertEqual((note.kind, note.tags, note.source, note.status), ("note", "profile", "evolve", "active"))
        self.assertEqual(note.text, "- Wants no filler. - Has an 8 GB GPU.")
        self.assertEqual([f.id for f in self.notes()], [fid])

    def test_a_new_profile_supersedes_the_old_one(self):
        self.soup.add("The user wants answers without filler.", kind="preference")
        first = rewrite_profile(ModelByPrompt("- Wants no filler."), self.soup)
        second = rewrite_profile(ModelByPrompt("- Wants no filler.\n- Builds kgirl."), self.soup)
        self.assertNotEqual(first, second)
        self.assertEqual([f.id for f in self.notes()], [second])
        self.assertEqual([f.id for f in self.notes("retired")], [first])

    def test_an_unchanged_profile_stays_the_one_active_note(self):
        self.soup.add("The user wants answers without filler.", kind="preference")
        first = rewrite_profile(ModelByPrompt("- Wants no filler."), self.soup)
        again = rewrite_profile(ModelByPrompt("- Wants no filler."), self.soup)
        self.assertEqual(again, first)
        self.assertEqual([f.id for f in self.notes()], [first])
        self.assertEqual(self.notes("retired"), [])

    def test_a_failing_or_silent_model_keeps_the_existing_profile(self):
        self.soup.add("The user wants answers without filler.", kind="preference")
        first = rewrite_profile(ModelByPrompt("- Wants no filler."), self.soup)
        self.assertIsNone(rewrite_profile(ModelByPrompt(RuntimeError("ollama is down")), self.soup))
        self.assertIsNone(rewrite_profile(ModelByPrompt("only reasoning, no answer\n</think>\n  "), self.soup))
        self.assertEqual([f.id for f in self.notes()], [first])


class EvolveTests(EvolveCase):
    REFLECTION = {"summary": "The user set up their editor.", "title": "Editor setup",
                  "facts": ["The user's editor is Helix."],
                  "preferences": ["The user wants terse commit messages."]}

    def test_one_consolidation_pass(self):
        fresh = self.store.new()
        _, answer = self.exchange(fresh, "which editor should kgirl assume?", "Helix, as you said")
        self.approve(answer)
        done = self.store.new("Already reflected")
        _, last = self.exchange(done, "old question", "old answer")
        self.store.update(done, summary="An old conversation.", reflected_turn=last)
        llm = ModelByPrompt("- Uses Helix.", self.REFLECTION)
        out = self.home / "sft" / "chat.jsonl"

        rep = evolve(llm, self.soup, self.curator, self.store, dataset=out, forge_skills=False)

        self.assertEqual(rep.reflected, [fresh])                 # the finished conversation is not reflected again
        self.assertEqual(len(llm.asked(REFLECT_SYSTEM)), 1)
        self.assertEqual(self.store.get(fresh).summary, "The user set up their editor.")
        self.assertEqual(self.store.get(done).summary, "An old conversation.")
        self.assertEqual((rep.promoted, rep.retired, rep.expired), ([], [], []))
        considering = [self.soup.get(i) for i in rep.considering]
        self.assertEqual({f.text for f in considering},
                         {"The user wants terse commit messages.", "The user's editor is Helix."})
        self.assertEqual(len(considering), 2)
        self.assertTrue(all(f.status == "staged" for f in considering))    # seen once: a suggestion, not a habit
        self.assertEqual(self.soup.get(rep.profile_id).text, "- Uses Helix.")
        self.assertIn("EPISODE: conversation #1 Editor setup: The user set up their editor.", llm.asked(PROFILE_SYSTEM)[0])
        self.assertEqual(rep.skills, [])
        self.assertEqual(rep.dataset, (str(out), 1))
        self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 1)

        lines = rep.render(self.soup).split("\n")
        self.assertEqual(lines[0], f"reflected on conversations: [{fresh}]")
        self.assertEqual(lines[1], "still considering (seen once; promoted when seen again, or `kgirl soup promote ID`):")
        self.assertEqual(sorted(lines[2:4]), sorted(f"  #{f.id} {f.kind}: {f.text}" for f in considering))
        self.assertEqual(lines[4:], [f"profile rewritten as #{rep.profile_id}",
                                     f"wrote 1 approved exchanges to {out}"])

    def test_a_second_pass_has_nothing_left_to_reflect_on(self):
        conv = self.store.new()
        self.exchange(conv, "which editor should kgirl assume?", "Helix, as you said")
        llm = ModelByPrompt("- Uses Helix.", self.REFLECTION)
        evolve(llm, self.soup, self.curator, self.store, forge_skills=False)
        rep = evolve(llm, self.soup, self.curator, self.store, forge_skills=False)
        self.assertEqual(rep.reflected, [])
        self.assertEqual(len(llm.asked(REFLECT_SYSTEM)), 1)
        self.assertIsNone(rep.dataset)
        self.assertTrue(rep.render(self.soup).startswith("reflected on conversations: none pending\n"))

    def test_a_skills_failure_never_fails_the_consolidation(self):
        self.soup.add("The user wants answers without filler.", kind="preference")
        with mock.patch("kgirl.harness.skills.forge", side_effect=RuntimeError("forge is broken")):
            rep = evolve(ModelByPrompt(), self.soup, self.curator, self.store)
        self.assertEqual(rep.skills, [])
        self.assertIsNotNone(rep.profile_id)

    def test_an_empty_report(self):
        self.assertEqual(EvolveReport().render(self.soup), "reflected on conversations: none pending")


class PreviewTests(EvolveCase):
    def staged(self, text: str, support: int, wins: float, source: str = "reflection:conv1") -> int:
        fid = self.soup.add(text, kind="fact", source=source, status="staged")
        for _ in range(support - 1):
            self.soup.reinforce(fid)
        self.soup.credit([fid], True, wins)
        return fid

    def test_a_dry_run_says_what_a_pass_would_do_and_changes_nothing(self):
        ready = self.staged("The user's staging box is box7.", 2, 3)
        lonely = self.staged("The user likes green tea.", 1, 3)
        manual = self.staged("A claim that arrived over a tool.", 5, 5, source="mcp")     # only a person promotes it
        stale = self.staged("The user once mentioned a ferry timetable.", 1, 0)
        self.soup.db.execute("UPDATE fragments SET created=? WHERE id=?", (now() - 60 * 86400, stale))
        failing = self.soup.add("Prefer verbose logging in every function.", kind="preference")
        self.soup.credit([failing], False, 8)
        fine = self.soup.add("The user wants short answers.", kind="preference")
        conv, other = self.store.new(), self.store.new()
        self.exchange(conv, "hello", "hi")
        _, answer = self.exchange(other, "how do I sort?", "use sorted")
        self.approve(answer)
        out = self.home / "approved.jsonl"
        ledger = len(self.soup.ledger(limit=100000))
        stats = self.soup.stats()

        rep = preview(self.soup, self.curator, self.store, out)

        self.assertTrue(rep.dry_run)
        self.assertEqual(sorted(rep.reflected), sorted([conv, other]))
        self.assertEqual((rep.promoted, rep.expired, rep.retired), ([ready], [stale], [failing]))
        self.assertEqual(rep.considering, [lonely])
        self.assertNotIn(manual, rep.promoted + rep.considering)
        self.assertEqual(rep.profile_from, (2, 0, 0))
        self.assertEqual(rep.dataset, (str(out), 1))
        # nothing was written: no ledger entry, no status change, no reflection, no file
        self.assertEqual(len(self.soup.ledger(limit=100000)), ledger)
        self.assertEqual(self.soup.stats(), stats)
        self.assertEqual(self.soup.get(ready).status, "staged")
        self.assertEqual(self.store.get(conv).reflected_turn, 0)
        self.assertEqual(self.notes(), [])
        self.assertFalse(out.exists())
        self.assertIsNotNone(self.soup.get(fine))

        text = rep.render(self.soup)
        self.assertTrue(text.startswith("dry run: nothing was changed, no model was called"))
        self.assertIn(f"would be promoted to habit:\n  #{ready} fact (support 2, utility 0.80): The user's staging box", text)
        self.assertIn(f"would be retired:\n  #{failing} preference", text)
        self.assertIn(f"would expire (proposals nobody repeated):\n  #{stale}", text)
        self.assertIn("would rewrite the profile from 2 preferences, 0 facts and 0 episodes", text)
        self.assertIn(f"would write 1 approved exchanges to {out}", text)
        self.assertIn("skills: not previewed", text)

    def test_the_preview_agrees_with_a_real_curator_step_that_has_no_decay(self):
        ready = self.staged("The user's staging box is box7.", 2, 3)
        self.staged("The user likes green tea.", 1, 3)
        stale = self.staged("A forgotten remark about ferries.", 1, 0)
        self.soup.db.execute("UPDATE fragments SET created=? WHERE id=?", (now() - 60 * 86400, stale))
        failing = self.soup.add("Prefer verbose logging in every function.", kind="preference")
        self.soup.credit([failing], False, 8)
        rep = preview(self.soup, self.curator, self.store)
        real = Curator(self.soup, CurationPolicy(gamma=1.0)).step()
        self.assertEqual((rep.promoted, rep.retired, rep.expired), (real.promoted, real.retired, real.expired))
        self.assertEqual(real.promoted, [ready])

    def test_capacity_overflow_is_previewed_too_and_an_empty_soup_says_so(self):
        curator = Curator(self.soup, CurationPolicy(capacity=2))
        ids = [self.soup.add(f"fact number {i} about the zebra", kind="fact") for i in range(4)]
        rep = preview(self.soup, curator, self.store)
        self.assertEqual(len(rep.retired), 2)
        self.assertTrue(set(rep.retired) <= set(ids))

    def test_an_empty_memory_has_nothing_to_preview(self):
        soup = Soup(self.home / "empty.db")
        store = ConversationStore(self.home / "empty-conversations.db")
        self.addCleanup(soup.close)
        self.addCleanup(store.close)
        text = preview(soup, Curator(soup), store).render(soup)
        self.assertEqual(text.splitlines()[-3:], ["would leave the profile alone (nothing to write it from)",
                                                  "skills: not previewed",
                                                  "nothing to reflect on, promote or retire"])

    def test_iter_chat_sft_is_what_export_writes(self):
        conv = self.store.new()
        _, a = self.exchange(conv, "q1", "a1")
        self.approve(a)
        rows = list(iter_chat_sft(self.store))
        self.assertEqual(len(rows), 1)
        out = self.home / "x.jsonl"
        self.assertEqual(export_chat_sft(self.store, out), 1)
        self.assertEqual([json.loads(ln) for ln in out.read_text().splitlines()], rows)


if __name__ == "__main__":
    unittest.main()
