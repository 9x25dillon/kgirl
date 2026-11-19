"""Offline tests for mind.writer: plan, draft by section, critique, revise, with a fake LLM.

The model is a callable that answers by system prompt (plan, draft, critic, revise) from fixed text: no
Ollama, no network.
"""

import json
import unittest

import tests.harness._fixtures  # noqa: F401  (sys.path)
from kgirl.harness.mind.writer import (CRITIC_SYSTEM, DRAFT_SYSTEM, PLAN_SYSTEM, REVISE_SYSTEM, Draft, Writer,
                                       _length_issues, _render_outline, _set_targets, style_guide)

OUTLINE = {"title": "Why small tasks win", "audience": "engineers", "thesis": "Small tasks are checkable.",
           "sections": [{"heading": "Intro", "points": ["state the claim", "one example"], "words": 120},
                        {"heading": "Evidence", "points": ["drill results"]},
                        {"heading": "Close", "points": ["what to do next"], "words": 90}]}


class ScriptedWriterLLM:
    """Replies by system prompt; drafts and revisions are taken in order."""

    def __init__(self, plan, drafts=(), critique=None, revisions=()):
        self.plan = plan
        self.drafts, self.revisions = list(drafts), list(revisions)
        self.critique = critique
        self.calls: list[tuple[str, str, int, float]] = []

    def __call__(self, system, prompt, max_tokens, temperature):
        self.calls.append((system, prompt, max_tokens, temperature))
        if system == PLAN_SYSTEM:
            return self.plan if isinstance(self.plan, str) else json.dumps(self.plan)
        if system == DRAFT_SYSTEM:
            return self.drafts.pop(0)
        if system == CRITIC_SYSTEM:
            return self.critique if isinstance(self.critique, str) else json.dumps(self.critique)
        return self.revisions.pop(0)

    def prompts(self, system) -> list[str]:
        return [prompt for s, prompt, _, _ in self.calls if s == system]


class DraftTests(unittest.TestCase):
    def test_markdown(self):
        d = Draft("A Title", {}, sections=[("One", "  first body\n"), ("Two", "second body")])
        self.assertEqual(d.markdown(), "# A Title\n\n## One\n\nfirst body\n\n## Two\n\nsecond body\n")

    def test_render_outline(self):
        text = _render_outline(OUTLINE, OUTLINE["sections"])
        self.assertEqual(text.split("\n"), [
            "Title: Why small tasks win", "Audience: engineers", "Thesis: Small tasks are checkable.",
            "- Intro (~120 words): state the claim; one example",
            "- Evidence (~200 words): drill results",
            "- Close (~90 words): what to do next"])


class WriterTests(unittest.TestCase):
    def test_plan_draft_critique_revise(self):
        critique = {"issues": [{"section": " intro ", "problem": "no example", "fix": "add the drill example"},
                               {"section": "Nowhere", "problem": "x", "fix": "y"},
                               "not an issue record"]}
        llm = ScriptedWriterLLM(OUTLINE, drafts=["intro body", "evidence body", "close body"], critique=critique,
                                revisions=["intro body, now with the drill example"])
        said: list[str] = []
        events: list[tuple[str, dict]] = []
        writer = Writer(llm, emit=lambda topic, payload: events.append((topic, payload)))
        draft = writer.write("Argue for small tasks.", progress=said.append)

        self.assertEqual(draft.title, "Why small tasks win")
        self.assertEqual(draft.sections, [("Intro", "intro body, now with the drill example"),
                                          ("Evidence", "evidence body"), ("Close", "close body")])
        self.assertEqual(draft.issues, critique["issues"][:2])           # only issue records are kept
        self.assertEqual(said, ["planning", "drafting 1/3: Intro", "drafting 2/3: Evidence", "drafting 3/3: Close",
                                "critiquing", "revising: Intro"])
        self.assertEqual(events, [("write.done", {"title": "Why small tasks win", "sections": 3, "issues": 2})])
        self.assertEqual([call[0] for call in llm.calls],
                         [PLAN_SYSTEM, DRAFT_SYSTEM, DRAFT_SYSTEM, DRAFT_SYSTEM, CRITIC_SYSTEM, REVISE_SYSTEM])
        self.assertEqual([call[2:] for call in llm.calls],
                         [(1200, 0.3), (1600, 0.6), (1600, 0.6), (1600, 0.6), (1000, 0.2), (1600, 0.4)])

    def test_each_draft_prompt_carries_the_outline_and_the_previous_section(self):
        llm = ScriptedWriterLLM(OUTLINE, drafts=["intro body", "evidence body", "close body"], critique={"issues": []})
        Writer(llm).write("Argue for small tasks.")
        first, second, last = llm.prompts(DRAFT_SYSTEM)
        self.assertTrue(first.startswith("BRIEF:\nArgue for small tasks.\n\nOUTLINE:\nTitle: Why small tasks win\n"))
        self.assertIn("PREVIOUS SECTION (end):\n(none)\n\nWRITE SECTION: Intro\n"
                      "Points: state the claim; one example\nLength: about 120 words. ", first)
        self.assertIn("PREVIOUS SECTION (end):\nintro body\n\nWRITE SECTION: Evidence\n"
                      "Points: drill results\nLength: about 200 words. ", second)
        self.assertNotIn("LAST section", first + second)
        self.assertTrue(last.endswith("Length: about 90 words. This is the LAST section: close the piece."), last)

    def test_the_critic_sees_the_whole_draft_and_revision_sees_its_issues(self):
        critique = {"issues": [{"section": "Close", "problem": "ends abruptly", "fix": "name the next step"}]}
        llm = ScriptedWriterLLM(OUTLINE, drafts=["intro body", "evidence body", "close body"], critique=critique,
                                revisions=["close body with a next step"])
        Writer(llm).write("Argue for small tasks.")
        critic = llm.prompts(CRITIC_SYSTEM)[0]
        self.assertTrue(critic.endswith("DRAFT:\n# Why small tasks win\n\n## Intro\n\nintro body\n\n## Evidence\n\n"
                                        "evidence body\n\n## Close\n\nclose body\n"), critic)
        self.assertEqual(llm.prompts(REVISE_SYSTEM), ["BRIEF:\nArgue for small tasks.\n\nSECTION: Close\nclose body"
                                                      "\n\nISSUES:\n- ends abruptly -> name the next step"])

    def test_context_and_target_length(self):
        outline = {"title": "T", "sections": [{"heading": "A", "points": ["p"]}, {"heading": "B", "points": ["q"]}]}
        llm = ScriptedWriterLLM(outline, drafts=["a", "b"])
        Writer(llm).write("The brief.", context="- (preference#3) no filler", words=1000, revise=False)
        self.assertEqual(llm.prompts(PLAN_SYSTEM), [
            "BRIEF:\nThe brief.\nTarget length: about 1000 words in total."
            "\n\nCONTEXT (what kgirl knows; use, don't contradict):\n- (preference#3) no filler"])
        for prompt in llm.prompts(DRAFT_SYSTEM):
            self.assertIn("CONTEXT (what kgirl knows; use, don't contradict):\n- (preference#3) no filler", prompt)
            self.assertIn("Length: about 500 words.", prompt)       # 1000 words shared by two sections

    def test_short_targets_still_give_each_section_room(self):
        outline = {"title": "T", "sections": [{"heading": h, "points": []} for h in "ABCDE"]}
        llm = ScriptedWriterLLM(outline, drafts=list("abcde"))
        Writer(llm).write("The brief.", words=200, revise=False)
        self.assertIn("Length: about 80 words.", llm.prompts(DRAFT_SYSTEM)[0])

    def test_no_revision_pass(self):
        llm = ScriptedWriterLLM(OUTLINE, drafts=["a", "b", "c"])
        said: list[str] = []
        draft = Writer(llm).write("The brief.", revise=False, progress=said.append)
        self.assertEqual([call[0] for call in llm.calls], [PLAN_SYSTEM] + [DRAFT_SYSTEM] * 3)
        self.assertEqual(draft.issues, [])
        self.assertNotIn("critiquing", said)

    def test_an_unusable_plan_falls_back_to_one_section(self):
        brief = "Explain, in plain words, why the reviewer experiment was stopped after stage A."
        for plan in ("I would start with an introduction.", {"title": "No sections"}, [1, 2, 3]):
            llm = ScriptedWriterLLM(plan, drafts=["the whole piece"])
            draft = Writer(llm).write(brief, words=300, revise=False)
            self.assertEqual(draft.title, brief[:60])
            self.assertEqual(draft.sections, [("Draft", "the whole piece")])
            self.assertIn(f"WRITE SECTION: Draft\nPoints: {brief}\nLength: about 300 words.",
                          llm.prompts(DRAFT_SYSTEM)[0])

    def test_sections_without_a_heading_are_skipped_and_seven_is_the_most(self):
        sections = [{"heading": f"S{i}", "points": []} for i in range(9)]
        sections.insert(1, {"points": ["no heading"]})
        sections.insert(2, "not a section")
        llm = ScriptedWriterLLM({"title": "Long", "sections": sections}, drafts=[f"body {i}" for i in range(7)])
        draft = Writer(llm).write("The brief.", revise=False)
        self.assertEqual([heading for heading, _ in draft.sections], [f"S{i}" for i in range(7)])

    def test_an_untitled_outline(self):
        llm = ScriptedWriterLLM({"sections": [{"heading": "Only", "points": []}]}, drafts=["body"])
        self.assertEqual(Writer(llm).write("The brief.", revise=False).title, "Untitled")

    def test_reasoning_is_stripped_from_drafts_and_revisions(self):
        critique = {"issues": [{"section": "Only", "problem": "p", "fix": "f"}]}
        llm = ScriptedWriterLLM({"title": "T", "sections": [{"heading": "Only", "points": []}]},
                                drafts=["<think>how to open</think>The draft."], critique=critique,
                                revisions=["weighing the fix\n</think>\nThe revision."])
        draft = Writer(llm).write("The brief.")
        self.assertEqual(draft.sections, [("Only", "The revision.")])
        self.assertIn("SECTION: Only\nThe draft.\n", llm.prompts(REVISE_SYSTEM)[0])

    def test_a_critique_that_is_not_usable_json_revises_nothing(self):
        for critique in ("Looks fine to me.", [{"section": "Only", "problem": "p"}], {"issues": []}, {"notes": "none"}):
            llm = ScriptedWriterLLM({"title": "T", "sections": [{"heading": "Only", "points": []}]},
                                    drafts=["body"], critique=critique)
            draft = Writer(llm).write("The brief.")
            self.assertEqual(draft.issues, [], critique)
            self.assertEqual(draft.sections, [("Only", "body")])
            self.assertEqual(llm.prompts(REVISE_SYSTEM), [])

    def test_at_most_eight_issues_are_kept(self):
        critique = {"issues": [{"section": "Elsewhere", "problem": f"p{i}", "fix": "f"} for i in range(12)]}
        llm = ScriptedWriterLLM({"title": "T", "sections": [{"heading": "Only", "points": []}]},
                                drafts=["body"], critique=critique)
        self.assertEqual(len(Writer(llm).write("The brief.").issues), 8)


class StyleAndCoverageTests(unittest.TestCase):
    def test_style_goes_into_every_call_apart_from_the_context(self):
        critique = {"issues": [{"section": "Intro", "problem": "p", "fix": "f"}]}
        llm = ScriptedWriterLLM(OUTLINE, drafts=["a", "b", "c"], critique=critique, revisions=["a2"])
        Writer(llm).write("The brief.", context="- a fact", style="Their preferences:\n- no filler")
        block = "STYLE (the user's; write in it):\nTheir preferences:\n- no filler"
        for system in (PLAN_SYSTEM, DRAFT_SYSTEM, CRITIC_SYSTEM, REVISE_SYSTEM):
            for prompt in llm.prompts(system):
                self.assertIn(block, prompt, system[:20])
        self.assertIn("CONTEXT (what kgirl knows; use, don't contradict):\n- a fact\n\n" + block,
                      llm.prompts(PLAN_SYSTEM)[0])

    def test_without_style_the_prompts_carry_none(self):
        llm = ScriptedWriterLLM(OUTLINE, drafts=["a", "b", "c"], critique={"issues": []})
        Writer(llm).write("The brief.")
        self.assertFalse(any("STYLE (" in call[1] for call in llm.calls))

    def test_later_sections_are_told_what_earlier_ones_covered(self):
        llm = ScriptedWriterLLM(OUTLINE, drafts=["We claim small tasks win. More text follows.", "Drills agree. x", "end"],
                                critique={"issues": []})
        Writer(llm).write("The brief.")
        first, second, third = llm.prompts(DRAFT_SYSTEM)
        self.assertNotIn("ALREADY COVERED", first)
        self.assertIn("ALREADY COVERED (do not repeat):\n- Intro: We claim small tasks win\n\nPREVIOUS SECTION", second)
        self.assertIn("- Intro: We claim small tasks win\n- Evidence: Drills agree", third)

    def test_style_guide(self):
        self.assertEqual(style_guide(), "")
        self.assertEqual(style_guide("- Builds kgirl.", ["no filler", " ", "short answers"]),
                         "Who the user is and how they like to work:\n- Builds kgirl.\n\n"
                         "Their preferences:\n- no filler\n- short answers")
        self.assertEqual(style_guide("", ["only a preference"]), "Their preferences:\n- only a preference")
        long = style_guide("x" * 3000, [], limit=100)
        self.assertEqual((len(long), long[-1]), (100, "…"))


class LengthTargetTests(unittest.TestCase):
    def test_targets_are_shared_or_scaled_to_the_total(self):
        secs = [{"heading": "A"}, {"heading": "B", "words": 300}, {"heading": "C", "words": "junk"}]
        _set_targets(secs, 900)
        self.assertEqual([s["words"] for s in secs], [300, 300, 300])           # the unplanned ones share the total
        full = [{"heading": "A", "words": 100}, {"heading": "B", "words": 300}]
        _set_targets(full, 800)                                                   # a complete plan 50% off: scaled
        self.assertEqual([s["words"] for s in full], [200, 600])
        near = [{"heading": "A", "words": 100}, {"heading": "B", "words": 300}]
        _set_targets(near, 450)                                                   # within a quarter: the plan stands
        self.assertEqual([s["words"] for s in near], [100, 300])
        none = [{"heading": "A"}]
        _set_targets(none, None)
        self.assertEqual(none, [{"heading": "A"}])

    def test_sections_far_off_target_become_issues_found_by_counting(self):
        plan = [{"heading": "Long", "words": 100}, {"heading": "Short", "words": 200},
                {"heading": "Fine", "words": 100}, {"heading": "Tiny target", "words": 40}]
        secs = [("Long", " ".join(["w"] * 180)), ("Short", " ".join(["w"] * 50)), ("Fine", " ".join(["w"] * 120)),
                ("Tiny target", " ".join(["w"] * 400))]
        got = _length_issues(secs, plan)
        self.assertEqual([(x["section"], x["problem"].split(":")[0]) for x in got],
                         [("Long", "too long"), ("Short", "too short")])
        self.assertTrue(all(x["auto"] for x in got))
        self.assertIn("cut it to about 100 words", got[0]["fix"])
        self.assertIn("expand it to about 200 words", got[1]["fix"])

    def test_a_length_target_adds_length_issues_to_the_critique_and_one_revision_fixes_both(self):
        outline = {"title": "T", "sections": [{"heading": "One", "points": ["p"], "words": 100},
                                              {"heading": "Two", "points": ["q"], "words": 100}]}
        critique = {"issues": [{"section": "One", "problem": "vague", "fix": "name the number"}]}
        llm = ScriptedWriterLLM(outline, drafts=[" ".join(["a"] * 300), " ".join(["b"] * 100)], critique=critique,
                                revisions=["one, shorter and exact"])
        d = Writer(llm).write("The brief.", words=200)
        self.assertEqual(len(llm.prompts(REVISE_SYSTEM)), 1)                      # Two is fine; One has two issues
        self.assertIn("- vague -> name the number\n- too long: 300 words against a target of about 100 -> cut it to",
                      llm.prompts(REVISE_SYSTEM)[0])
        self.assertEqual([x.get("auto", False) for x in d.issues], [False, True])
        self.assertEqual(d.sections[0], ("One", "one, shorter and exact"))

    def test_without_a_total_target_length_is_not_enforced(self):
        outline = {"title": "T", "sections": [{"heading": "One", "points": ["p"], "words": 100}]}
        llm = ScriptedWriterLLM(outline, drafts=[" ".join(["a"] * 900)], critique={"issues": []})
        self.assertEqual(Writer(llm).write("The brief.").issues, [])


class ResumeTests(unittest.TestCase):
    def test_a_checkpoint_follows_every_step_and_the_record_round_trips(self):
        critique = {"issues": [{"section": "Close", "problem": "abrupt", "fix": "name the next step"}]}
        llm = ScriptedWriterLLM(OUTLINE, drafts=["a", "b", "c"], critique=critique, revisions=["c2"])
        seen: list[tuple[str, int, bool, list[str]]] = []
        Writer(llm).write("The brief.", style="s",
                          checkpoint=lambda d: seen.append((d.stage, len(d.sections), d.critiqued, list(d.revised))))
        self.assertEqual(seen, [("drafting", 0, False, []), ("drafting", 1, False, []), ("drafting", 2, False, []),
                                ("drafting", 3, False, []), ("critique", 3, True, []), ("revising", 3, True, ["Close"]),
                                ("done", 3, True, ["Close"])])
        d = Writer(ScriptedWriterLLM(OUTLINE, drafts=["a", "b", "c"], critique=critique, revisions=["c2"])).write(
            "The brief.", style="s")
        d.words = 300                                                             # (a length target survives too)
        back = Draft.from_dict(json.loads(json.dumps(d.to_dict())))
        self.assertEqual(back, d)
        self.assertEqual((back.stage, back.done, back.progress(), back.words, back.style), ("done", True,
                                                                                            "done: 3 sections", 300, "s"))

    def test_resume_drafts_only_what_is_missing(self):
        first = ScriptedWriterLLM(OUTLINE, drafts=["intro body", "evidence body", "close body"], critique={"issues": []})
        snapshots: list[dict] = []

        class Stop(Exception):
            pass

        def stop_after_two_sections(d: Draft) -> None:
            snapshots.append(json.loads(json.dumps(d.to_dict())))
            if len(d.sections) == 2:
                raise Stop

        with self.assertRaises(Stop):
            Writer(first).write("Argue for small tasks.", context="- a fact", style="the voice",
                                checkpoint=stop_after_two_sections)
        saved = Draft.from_dict(snapshots[-1])
        self.assertEqual((saved.stage, len(saved.sections), saved.progress()), ("drafting", 2, "drafting: 2/3 sections"))

        second = ScriptedWriterLLM({}, drafts=["close body"], critique={"issues": []})
        said: list[str] = []
        d = Writer(second).write("ignored brief", context="ignored", style="ignored", resume=saved, progress=said.append)
        self.assertEqual([c[0] for c in second.calls], [DRAFT_SYSTEM, CRITIC_SYSTEM])          # no new plan, one draft
        self.assertEqual(said, ["drafting 3/3: Close", "critiquing"])
        self.assertEqual([t for _, t in d.sections], ["intro body", "evidence body", "close body"])
        self.assertIn("BRIEF:\nArgue for small tasks.", second.prompts(DRAFT_SYSTEM)[0])       # the draft's own brief,
        self.assertIn("- a fact", second.prompts(DRAFT_SYSTEM)[0])                              # context
        self.assertIn("the voice", second.prompts(DRAFT_SYSTEM)[0])                             # and style
        self.assertNotIn("ignored", second.prompts(DRAFT_SYSTEM)[0])
        self.assertEqual((d.stage, d.title), ("done", "Why small tasks win"))

    def test_resume_does_not_revise_twice_and_a_finished_draft_is_returned_as_is(self):
        critique = {"issues": [{"section": "Intro", "problem": "p", "fix": "f"}, {"section": "Close", "problem": "q",
                                                                                   "fix": "g"}]}
        llm = ScriptedWriterLLM(OUTLINE, drafts=["a", "b", "c"], critique=critique, revisions=["a2"])
        snaps: list[dict] = []

        def stop_after_first_revision(d: Draft) -> None:
            snaps.append(json.loads(json.dumps(d.to_dict())))
            if d.revised:
                raise KeyError

        with self.assertRaises(KeyError):
            Writer(llm).write("The brief.", checkpoint=stop_after_first_revision)
        again = ScriptedWriterLLM({}, revisions=["c2"])
        d = Writer(again).write("", resume=Draft.from_dict(snaps[-1]))
        self.assertEqual([c[0] for c in again.calls], [REVISE_SYSTEM])                          # only Close is left
        self.assertEqual(d.sections, [("Intro", "a2"), ("Evidence", "b"), ("Close", "c2")])
        idle = ScriptedWriterLLM({})
        self.assertIs(Writer(idle).write("", resume=d), d)
        self.assertEqual(idle.calls, [])

    def test_a_damaged_record_loads_as_far_as_it_can(self):
        d = Draft.from_dict({"title": 5, "sections": [["A", "body"], "junk", ["only one"]], "stage": "weird",
                             "words": -3, "plan": [1, {"heading": "A"}], "issues": ["x"], "recalled": ["a", 3]})
        self.assertEqual((d.title, d.sections, d.stage, d.words, d.plan, d.issues, d.recalled),
                         ("5", [("A", "body")], "outline", None, [{"heading": "A"}], [], [3]))
        self.assertEqual(Draft.from_dict({}).title, "Untitled")


if __name__ == "__main__":
    unittest.main()
