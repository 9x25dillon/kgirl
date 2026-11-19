"""Offline tests for mind.mind: one chat turn end to end, with a scripted backend that streams.

The assistant is the real one on a temporary home (empty Atlas, empty Soup); the chat role is served by a
backend that replays fixed replies piece by piece. No Ollama, no network, no live ~/.kgirl state, and the
swarm is never started.
"""

import json
import os
import sys
import types
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tests.harness._fixtures  # noqa: F401  (sys.path)
from kgirl.harness.assistant import Assistant
from kgirl.harness.hermes import Router
from kgirl.harness.hermes.router import BUILD, CHAT, JEV, SCOUT, SMITH
from kgirl.harness.llm import BackendError, Completion, Message
from kgirl.harness.mind import Mind
from kgirl.harness.mind import builder as builder_module
from kgirl.harness.mind import mind as mind_module
from kgirl.harness.mind.reflect import observed_in
from kgirl.harness.mind.mind import NO_MORE_TOOLS, OUT_OF_STEPS, PERSONA, _clip_result, _int, latest_profile
from kgirl.harness.mind.tools import CALL_FORMAT, STOP
from kgirl.harness.mind.workspace import Workspace

CLIPPED = "more characters not shown; call the tool again with a narrower range]"


class StreamingBackend:
    """A scripted chat backend. Each reply is a string or a list of pieces; `stream` yields the pieces."""

    name = "scripted"

    def __init__(self, replies):
        self.model = "scripted"
        self.replies = [list(r) if isinstance(r, (list, tuple)) else [r] for r in replies]
        self.calls: list[tuple[str, list[Message]]] = []
        self.stops: list = []

    def available(self) -> bool:
        return True

    def _next(self, system, messages) -> list[str]:
        self.calls.append((system, list(messages)))
        if not self.replies:
            raise BackendError("scripted backend exhausted")
        return self.replies.pop(0)

    def complete(self, system, messages, max_tokens=1024, temperature=0.2, stop=None) -> Completion:
        return Completion("".join(self._next(system, messages)), self.model, self.name)

    def stream(self, system, messages, max_tokens=1024, temperature=0.2, stop=None):
        self.stops.append(stop)
        yield from self._next(system, messages)


class MindCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ, {"KGIRL_VERIFY_ALLOWLIST": ""})
        env.start()
        self.addCleanup(env.stop)
        for key in ("KGIRL_CHUNKS_DB", "KGIRL_QINCRS_GATE"):       # optional subsystems stay off
            os.environ.pop(key, None)
        self.base = Path(self.tmp.name).resolve()

    def mind(self, *replies, **kw) -> Mind:
        self.chat = StreamingBackend(replies)
        router = Router({CHAT: [self.chat], BUILD: [], SCOUT: [], SMITH: [], JEV: []})
        self.a = Assistant(home=self.base / "home", router=router, trace=False, life=False)
        self.addCleanup(self.a.close)
        kw.setdefault("background_reflection", False)
        kw.setdefault("reflect_every", 99)
        mind = Mind(self.a, **kw)
        self.addCleanup(mind.store.close)
        return mind

    def workspace(self, **kw) -> Workspace:
        root = self.base / "proj"
        root.mkdir()
        return Workspace(root, **kw)

    def turns(self, mind: Mind, conv: int):
        return mind.store.turns(conv)


class PlainAnswerTests(MindCase):
    def test_an_answer_is_shown_stored_and_kept_with_its_context(self):
        mind = self.mind("The router lives in hermes/router.py.")
        conv = mind.start()
        shown: list[str] = []
        reply = mind.chat(conv, "  where is the router?  ", on_text=shown.append)
        self.assertEqual(reply.text, "The router lives in hermes/router.py.")
        self.assertEqual((reply.conv_id, reply.recalled, reply.tools, reply.stored, reply.verdict),
                         (conv, [], [], [], "none"))
        self.assertEqual("".join(shown), "The router lives in hermes/router.py.\n")

        user, answer = self.turns(mind, conv)
        self.assertEqual((user.role, user.content), ("user", "where is the router?"))
        self.assertEqual((answer.role, answer.content), ("assistant", reply.text))
        self.assertTrue(answer.is_answer)
        self.assertEqual(answer.meta["tools"], [])
        self.assertEqual(set(answer.meta), {"tools", "budget", "related"})      # keys other surfaces read
        self.assertEqual(answer.meta["related"], [])
        self.assertEqual(answer.meta["budget"]["dropped"], {})                  # a small prompt: nothing cut

        system, messages = self.chat.calls[0]
        self.assertEqual(messages, [Message("user", "where is the router?")])
        self.assertTrue(system.startswith(PERSONA + "\n\nTOOLS. To call one"), system[:200])
        self.assertIn(CALL_FORMAT, system)
        self.assertEqual(mind.store.context(answer.id), system)     # the prompt the answer was generated from
        self.assertEqual(self.chat.stops, [STOP])

    def test_a_model_that_returns_nothing(self):
        mind = self.mind("")
        conv = mind.start()
        shown: list[str] = []
        reply = mind.chat(conv, "hello", on_text=shown.append)
        self.assertEqual(reply.text, "(the model returned nothing; try rephrasing)")
        self.assertEqual(shown, [reply.text + "\n"])
        self.assertEqual(self.turns(mind, conv)[-1].content, reply.text)

    def test_reasoning_is_neither_shown_nor_stored(self):
        mind = self.mind(["<think>the user wants", " a greeting</think>", "Hello!"])
        conv = mind.start()
        shown: list[str] = []
        reply = mind.chat(conv, "hi", on_text=shown.append)
        self.assertEqual(reply.text, "Hello!")
        self.assertEqual(shown, ["Hello!", "\n"])
        self.assertEqual(self.turns(mind, conv)[-1].content, "Hello!")


class ToolLoopTests(MindCase):
    def test_text_streamed_before_a_tool_call_is_shown_and_stored(self):
        mind = self.mind(["Let me ", "check.\n<to", "ol>repos {}", "</tool>", " TOOL RESULT invented by the model"],
                         "Nothing is indexed yet.")
        conv = mind.start()
        shown: list[str] = []
        events: list[str] = []
        reply = mind.chat(conv, "which repos are indexed?", on_text=shown.append, on_event=events.append)

        self.assertEqual("".join(shown), "Let me check.\n\nNothing is indexed yet.\n")   # the tag is never shown
        self.assertEqual(events, ["tool repos {}"])
        self.assertEqual(reply.text, "Nothing is indexed yet.")
        self.assertEqual(reply.tools, [("repos", {})])

        user, step, result, answer = self.turns(mind, conv)
        self.assertEqual((step.role, step.content), ("assistant", "Let me check.\n<tool>repos {}</tool>"))
        self.assertTrue(step.partial)
        self.assertEqual(step.meta, {"partial": True, "tool": "repos"})
        self.assertEqual((result.role, result.content), ("tool", "[repos] no repos indexed"))
        self.assertEqual(result.meta, {"tool": "repos", "args": {}})
        self.assertEqual(answer.meta["tools"], ["repos"])
        self.assertEqual(self.chat.calls[1][1], [Message("user", "which repos are indexed?"),
                                                 Message("assistant", "Let me check.\n<tool>repos {}</tool>"),
                                                 Message("user", "TOOL RESULT [repos] no repos indexed")])

    def test_a_result_the_model_invented_after_its_call_is_dropped(self):
        mind = self.mind("<tool>repos {}</tool>\nTOOL RESULT [repos] kgirl (9 files)\nSo there is one repo.",
                         "Nothing is indexed yet.")
        conv = mind.start()
        shown: list[str] = []
        mind.chat(conv, "which repos are indexed?", on_text=shown.append)
        _, step, result, _ = self.turns(mind, conv)
        self.assertEqual(step.content, "<tool>repos {}</tool>")              # the canonical call, nothing after it
        self.assertEqual(result.content, "[repos] no repos indexed")
        self.assertEqual("".join(shown), "Nothing is indexed yet.\n")

    def test_an_unreadable_call_is_explained_to_the_model(self):
        mind = self.mind("<tool></tool>", "Here is the answer without tools.")
        conv = mind.start()
        events: list[str] = []
        reply = mind.chat(conv, "which repos are indexed?", on_event=events.append)
        explained = f"[error] that tool call could not be read: the tool call is empty. Write exactly: {CALL_FORMAT}"
        _, step, result, _ = self.turns(mind, conv)
        self.assertEqual((step.content, step.meta), ("<tool></tool>", {"partial": True, "tool": ""}))
        self.assertEqual(result.content, explained)
        self.assertEqual(self.chat.calls[1][1][-1], Message("user", "TOOL RESULT " + explained))
        self.assertEqual((reply.text, reply.tools, events), ("Here is the answer without tools.", [], []))

    def test_an_unknown_tool_is_a_result_not_a_crash(self):
        mind = self.mind('<tool>teleport {"to": "prod"}</tool>', "I cannot do that.")
        conv = mind.start()
        reply = mind.chat(conv, "teleport me")
        self.assertEqual(self.turns(mind, conv)[2].content,
                         "[teleport] unknown tool 'teleport'; the tools are: search_code, read_file, task, recall, "
                         "remember, build, write, repos")
        self.assertEqual(reply.tools, [("teleport", {"to": "prod"})])

    def test_a_failing_tool_is_a_result_not_a_crash(self):
        mind = self.mind('<tool>read_file {"path": "a.py"}</tool>', "I need a repo name.")
        conv = mind.start()
        mind.chat(conv, "show me a.py")
        self.assertEqual(self.turns(mind, conv)[2].content,
                         '[read_file] read_file failed: ValueError: no workspace is attached, so read_file needs '
                         '"repo": an indexed repository')
        self.assertEqual(mind.call_tool("read_file", {}),
                         "read_file failed: missing argument 'path'; use "
                         '{"repo": "...", "path": "...", "start": 1, "end": 120}')

    def test_when_the_tool_budget_is_spent_the_model_must_answer(self):
        mind = self.mind("<tool>repos {}</tool>", "<tool>repos {}</tool>", "I looked twice; nothing is indexed.",
                         max_steps=2)
        conv = mind.start()
        reply = mind.chat(conv, "which repos are indexed?")
        self.assertEqual(reply.text, "I looked twice; nothing is indexed.")
        self.assertEqual(reply.tools, [("repos", {}), ("repos", {})])
        systems = [system for system, _ in self.chat.calls]
        self.assertFalse(systems[1].endswith(NO_MORE_TOOLS))
        self.assertEqual(systems[2], systems[0] + NO_MORE_TOOLS)
        self.assertEqual([t.role for t in self.turns(mind, conv)],
                         ["user", "assistant", "tool", "assistant", "tool", "assistant"])

    def test_a_model_that_still_calls_a_tool_gets_no_more_steps(self):
        mind = self.mind("<tool>repos {}</tool>", "Still looking.\n<tool>repos {}</tool>", max_steps=1)
        conv = mind.start()
        shown: list[str] = []
        reply = mind.chat(conv, "which repos are indexed?", on_text=shown.append)
        self.assertEqual(reply.text, "Still looking.")                        # what it wrote before the call
        self.assertEqual(len(reply.tools), 1)

        mind = self.mind("<tool>repos {}</tool>", "<tool>repos {}</tool>", max_steps=1)
        conv = mind.start()
        shown = []
        reply = mind.chat(conv, "which repos are indexed?", on_text=shown.append)
        self.assertEqual(reply.text, OUT_OF_STEPS)
        self.assertEqual(shown, [OUT_OF_STEPS + "\n"])
        self.assertEqual(self.turns(mind, conv)[-1].content, OUT_OF_STEPS)


class WorkspaceToolTests(MindCase):
    def test_tools_offered_with_and_without_a_workspace(self):
        self.assertEqual(list(self.mind().tools()),
                         ["search_code", "read_file", "task", "recall", "remember", "build", "write", "repos"])
        with_ws = Mind(self.a, store=None, workspace=self.workspace(), background_reflection=False)
        self.addCleanup(with_ws.store.close)
        self.assertEqual(list(with_ws.tools()),
                         ["search_code", "read_file", "ls", "grep", "edit", "write_file", "run", "task", "recall",
                          "remember", "build", "write", "repos"])

    def test_the_model_can_write_a_file_only_with_consent(self):
        call = '<tool>write_file {"path": "notes.txt", "content": "hello\\n"}</tool>'
        ws = self.workspace(auto_edit=True)
        mind = self.mind(call, "Created notes.txt.", call, "I need your approval for that.", workspace=ws)
        conv = mind.start()
        mind.chat(conv, "write hello into notes.txt")
        self.assertEqual((ws.root / "notes.txt").read_text(encoding="utf-8"), "hello\n")
        self.assertTrue(self.turns(mind, conv)[2].content.startswith("[write_file] created notes.txt\n"))
        self.assertIn(f"WORKSPACE: {ws.root}", self.chat.calls[0][0])

        (ws.root / "notes.txt").unlink()
        ws.auto_edit = False                                                  # nobody to ask, nothing pre-approved
        mind.chat(conv, "write it again")
        refused = self.turns(mind, conv)[-2].content
        self.assertTrue(refused.startswith("[write_file] write_file refused: changing notes.txt needs the user's "
                                           "approval"), refused)
        self.assertFalse((ws.root / "notes.txt").exists())

    def test_the_model_cannot_read_outside_the_workspace(self):
        (self.base / "secret.txt").write_text("not for the model\n", encoding="utf-8")
        ws = self.workspace()
        mind = self.mind('<tool>read_file {"path": "../secret.txt"}</tool>', "I cannot read that.", workspace=ws)
        conv = mind.start()
        mind.chat(conv, "read ../secret.txt")
        result = self.turns(mind, conv)[2].content
        self.assertEqual(result, f"[read_file] read_file refused: ../secret.txt is outside the workspace {ws.root}")
        self.assertNotIn("not for the model", "".join(m.content for m in self.chat.calls[1][1]))

    def test_scope_start_and_resume(self):
        mind = self.mind()
        self.assertEqual(mind.scope(), "*")
        conv = mind.resume()                                                  # nothing yet: a new conversation
        self.assertEqual(mind.resume(), conv)
        self.assertEqual(mind.store.get(conv).scope, "*")

        scoped = Mind(self.a, store=mind.store, workspace=self.workspace(), background_reflection=False)
        self.assertEqual(scoped.scope(), "proj")
        proj = scoped.resume()                                                # the "*" conversation is not this project's
        self.assertNotEqual(proj, conv)
        self.assertEqual(mind.store.get(proj).scope, "proj")
        mind.store.append(conv, "user", "a later message elsewhere")
        self.assertEqual(scoped.resume(), proj)
        self.assertEqual(mind.resume(), conv)                                 # without a workspace: the latest of any


class TaskToolTests(MindCase):
    """The model may name a verifier only from what the user allowed; the swarm never starts otherwise."""

    def mind_with_swarm(self, **kw) -> Mind:
        mind = self.mind(**kw)
        self.a.task = mock.Mock(return_value=mock.Mock(winner=None, render=lambda: "swarm report"))
        return mind

    def test_no_workspace_and_no_repo(self):
        mind = self.mind_with_swarm()
        self.assertEqual(mind.call_tool("task", {"goal": "fix the bug"}),
                         'task failed: ValueError: no workspace is attached, so task needs "repo": an indexed repository')
        self.assertEqual(mind.call_tool("task", {"goal": "fix the bug", "repo": "nowhere"}),
                         "task failed: ValueError: 'nowhere' is not an indexed repo (see the repos tool)")
        self.assertEqual(mind.call_tool("task", {"goal": "  "}), "task failed: ValueError: task needs a goal")
        self.a.task.assert_not_called()

    def test_a_verifier_the_user_never_allowed_is_refused(self):
        mind = self.mind_with_swarm(workspace=self.workspace())
        self.assertEqual(mind.call_tool("task", {"goal": "fix the bug", "verify": "rm -rf /"}),
                         "task failed: ValueError: no verifier is allowed yet; ask the user to set one with "
                         "/verify <command>")
        self.assertEqual(mind.call_tool("task", {"goal": "fix the bug"}),
                         "task failed: ValueError: task needs a verifier, and none is allowed yet; ask the user to "
                         "set one with /verify <command>")
        mind.verify_cmd = ["python3", "-m", "unittest"]
        self.assertEqual(mind.call_tool("task", {"goal": "fix the bug", "verify": "make deploy"}),
                         "task failed: ValueError: verify must be exactly one of: `python3 -m unittest`")
        self.a.task.assert_not_called()

    def test_the_users_verifier_is_used_and_an_unverified_swarm_leaves_nothing_to_apply(self):
        ws = self.workspace()
        mind = self.mind_with_swarm(workspace=ws, verify_cmd=["python3", "-m", "unittest"])
        events: list[str] = []
        out = mind.call_tool("task", {"goal": " fix the bug "}, on_event=events.append)
        self.assertEqual(out, "swarm report\nNo agent produced a change that passes the verifier.")
        self.a.task.assert_called_once_with("fix the bug", str(ws.root), ["python3", "-m", "unittest"],
                                            repo_name="proj", origin="chat")
        self.assertEqual(events, ["swarm: sandboxed agents work on proj, verified by `python3 -m unittest` "
                                  "(takes minutes)"])
        self.assertIsNone(mind.last_task)
        self.assertEqual(mind.apply_task(), "there is no verified swarm diff waiting")

    def test_a_command_allowed_in_the_session_can_be_named(self):
        ws = self.workspace()
        ws.allow("make test")
        mind = self.mind_with_swarm(workspace=ws)
        mind.call_tool("task", {"goal": "fix the bug", "verify": "make test"})
        self.assertEqual(self.a.task.call_args.args[2], ["make", "test"])
        self.assertEqual(mind.call_tool("task", {"goal": "fix the bug"}),
                         'task failed: ValueError: task needs "verify": one of `make test`')


class MemoryTests(MindCase):
    def test_a_reaction_credits_the_memory_the_answer_was_built_from(self):
        mind = self.mind("Use sorted(xs).", "Glad it helped.")
        fid = self.a.soup.add("The user wants short answers with one code example.", kind="preference")
        conv = mind.start()
        self.assertEqual(mind.feedback(conv, "positive"), "there is no answer to judge yet")
        first = mind.chat(conv, "how do I sort a list?")
        self.assertEqual(first.recalled, [fid])
        self.assertIn("WHAT THE USER WANTS (their preferences):\n- The user wants short answers with one code example.",
                      self.chat.calls[0][0])

        events: list[str] = []
        second = mind.chat(conv, "thanks, that works", on_event=events.append)
        self.assertEqual(second.verdict, "positive")
        self.assertEqual(events, [f"+ read as feedback on the last answer; credited memories [{fid}]"])
        answer = self.turns(mind, conv)[1]
        self.assertEqual(answer.meta["feedback"], {"verdict": "positive", "source": "reaction", "weight": 1.0})
        self.assertEqual(self.a.soup.get(fid).wins, 1.0)

        self.assertEqual(mind.feedback(conv, "negative"), f"noted: negative; credited memories [{fid}]")
        self.assertEqual(self.a.soup.get(fid).losses, 1.0)
        self.assertEqual(self.turns(mind, conv)[-1].meta["feedback"]["source"], "command")

    def test_a_message_that_only_mentions_failure_is_not_a_verdict(self):
        mind = self.mind("Use sorted(xs).", "Guard the empty case first.")
        fid = self.a.soup.add("The user wants short answers with one code example.", kind="preference")
        conv = mind.start()
        mind.chat(conv, "how do I sort a list?")
        reply = mind.chat(conv, "the test fails when the input is empty")
        self.assertEqual(reply.verdict, "none")
        self.assertEqual(self.turns(mind, conv)[1].verdict, "")
        self.assertEqual((self.a.soup.get(fid).wins, self.a.soup.get(fid).losses), (0.0, 0.0))

    def test_an_explicit_verdict_on_an_answer_that_used_no_memory(self):
        mind = self.mind("Use sorted(xs).")
        conv = mind.start()
        mind.chat(conv, "how do I sort a list?")
        self.assertEqual(mind.feedback(conv, "positive"), "noted: positive; that answer used no stored memory")

    def test_remember_that_is_stored_at_once_as_the_users_word(self):
        mind = self.mind("Noted.")
        conv = mind.start()
        events: list[str] = []
        reply = mind.chat(conv, "remember that the staging box is 10.0.0.7", on_event=events.append)
        (fact,) = self.a.soup.all(kind="fact")
        self.assertEqual((fact.text, fact.status, fact.source, fact.tags),
                         ("the staging box is 10.0.0.7", "active", "user", "conversation"))
        self.assertEqual(reply.stored, [f"fact#{fact.id}"])
        self.assertEqual(events, [f"remembered fact #{fact.id}"])

    def test_the_remember_tool_only_proposes(self):
        mind = self.mind()
        out = mind.call_tool("remember", {"text": "the deploy host is box7", "kind": "not-a-kind"})
        (fact,) = self.a.soup.all(kind="fact")
        self.assertEqual(out, f"staged fact #{fact.id} (becomes active once it is observed again or the user "
                              "promotes it)")
        self.assertEqual((fact.status, fact.source), ("staged", "reflection:chat"))
        self.assertEqual(mind.call_tool("remember", {"text": "   "}), "remember failed: ValueError: remember needs text")

    def test_the_recall_tool_searches_memory_and_earlier_turns(self):
        mind = self.mind()
        self.assertEqual(mind.call_tool("recall", {"query": "zebra"}),
                         "(no stored memory matches)\n\nconversation turns:\n(none)")
        fid = self.a.soup.add("the zebra service runs on port 9000", kind="fact")
        conv = mind.start()
        mind.store.append(conv, "user", "the zebra migration is done")
        self.assertEqual(mind.call_tool("recall", {"query": "zebra"}),
                         f"- (fact#{fid} u=0.50) the zebra service runs on port 9000\n\nconversation turns:\n"
                         f"- conv #{conv} user: the zebra migration is done")


    def test_the_recall_tool_also_lists_earlier_conversations_on_the_same_topic(self):
        mind = self.mind()
        earlier = mind.start("Zebra migration")
        mind.store.update(earlier, summary="Moved the zebra service to the new host.", tags=["zebra", "migration"])
        self.assertEqual(mind.call_tool("recall", {"query": "the zebra move"}),
                         "(no stored memory matches)\n\nconversation turns:\n(none)\n\n"
                         "earlier conversations on this topic:\n"
                         f"- conv #{earlier} Zebra migration (tags: zebra): Moved the zebra service to the new host.")

    def test_the_profile_and_earlier_conversations_reach_the_prompt(self):
        mind = self.mind("It finished last week.")
        self.assertIsNone(latest_profile(self.a.soup))
        profile = self.a.soup.add("- Builds kgirl. - Wants no filler.", kind="note", tags="profile", source="evolve")
        self.assertEqual(latest_profile(self.a.soup).id, profile)
        earlier = mind.start()
        mind.store.append(earlier, "user", "the zebra migration is done")
        conv = mind.start()
        reply = mind.chat(conv, "what happened with the zebra migration?")
        system = self.chat.calls[0][0]
        self.assertIn("WHO THE USER IS (kgirl's consolidated profile):\n- Builds kgirl. - Wants no filler.", system)
        self.assertIn(f"EARLIER CONVERSATIONS (same topic, or matching turns):\n- conv #{earlier} user: the zebra migration is done",
                      system)
        self.assertEqual(reply.recalled[0], profile)


class PromptBudgetTests(MindCase):
    def roomy(self, *replies) -> Mind:
        return self.mind(*replies, context_tokens=24576, memory_budget=2000)

    def tighten(self, mind: Mind, limit: int) -> None:
        """Shrink the context so the prompt limit is about `limit` tokens."""
        mind.answer_tokens = 400
        mind.context_tokens = int(limit / 0.92) + 401

    def seed(self, mind: Mind) -> int:
        soup = self.a.soup
        soup.add("Never run git push.", kind="preference")
        soup.add("The user likes concise answers about zebra services.", kind="preference")
        for i, word in enumerate(["amber", "basalt", "cobalt", "dune", "ember", "flint", "granite", "harbor",
                                  "indigo", "jasper"]):           # distinct wording, so recall keeps all of them
            soup.add(f"zebra service {word} listens on port {9000 + i} and stores its {word} data in /srv/{word} "
                     f"with nightly {word} backups kept for {i + 3} days", kind="fact")
        earlier = mind.start("Zebra migration", scope="*")
        mind.store.update(earlier, summary="Moved the zebra service to the new host " + "and checked it " * 10,
                          tags=["zebra", "migration"])
        conv = mind.start()
        mind.store.update(conv, summary="Working on the zebra service.", threads=["pick a port"])
        return conv

    def test_the_budget_cuts_low_priority_blocks_first_and_never_the_rules(self):
        mind = self.roomy()
        conv = self.seed(mind)
        mind.store.append(conv, "user", "tell me about the zebra service")
        whole = mind._prompt(conv, "tell me about the zebra service")
        before = whole.meta["budget"]
        self.assertEqual(before["dropped"], {})
        self.assertGreater(before["blocks"]["memory"], 300)
        self.assertGreater(before["blocks"]["earlier"], 0)
        self.assertIn("RULES THE USER SET", whole.system)

        self.tighten(mind, before["used"] - 400)
        cut = mind._prompt(conv, "tell me about the zebra service")
        after = cut.meta["budget"]
        self.assertLessEqual(after["used"], after["limit"])
        self.assertEqual(after["overflow"], 0)
        self.assertIn("earlier", after["dropped"])                      # the lowest priority goes first
        self.assertEqual(after["blocks"]["earlier"], 0)
        self.assertIn("memory", after["dropped"])                       # then what the first cut did not cover
        self.assertGreater(after["dropped"]["memory"]["items"], 0)
        self.assertNotIn("wants", after["dropped"])
        self.assertNotIn("so_far", after["dropped"])
        self.assertNotIn("rules", after["dropped"])
        self.assertEqual(after["blocks"]["rules"], before["blocks"]["rules"])
        self.assertIn("Never run git push.", cut.system)
        self.assertNotIn("EARLIER CONVERSATIONS (same topic", cut.system)      # the block heading (the persona names the block)
        self.assertEqual(cut.meta["related"], [])                       # a cut one is not reported as shown
        gone = set(whole.recalled) - set(cut.recalled)                  # memory that was cut was not "recalled"
        self.assertTrue(gone)

    def test_context_shows_block_sizes_and_what_was_dropped(self):
        mind = self.roomy("Zebra services listen on 9000.")
        conv = self.seed(mind)
        self.assertEqual(mind.context_report(conv), "no prompt has been built in this conversation yet")
        mind.chat(conv, "tell me about the zebra service")
        text = mind.context_report(conv)
        self.assertIn("prompt budget", text)
        self.assertRegex(text, r"RULES\s+\d+\s+never cut")
        self.assertIn("nothing was cut", text)
        mind._budgets.clear()                                           # a later process: the answer's stored meta
        self.assertEqual(mind.context_report(conv), text)

        self.tighten(mind, mind.store.last_answer(conv).meta["budget"]["used"] - 300)
        mind._prompt(conv, "tell me about the zebra service")
        self.assertRegex(mind.context_report(conv), r"EARLIER\s+0\s+cut \d+ tokens")

    def test_older_turns_get_only_the_room_the_budget_leaves(self):
        mind = self.mind()
        conv = mind.start()
        for i in range(20):
            mind.store.append(conv, "user", f"question {i} " + "q" * 380)
            mind.store.append(conv, "assistant", f"answer {i} " + "a" * 380)
        mind.store.append(conv, "user", "and the last one?")
        whole = mind._prompt(conv, "and the last one?")
        roomy = mind._messages(conv)
        self.assertEqual(whole.meta["budget"]["dropped"], {})
        self.assertGreater(len(roomy), 6)
        self.assertLess(len(roomy), 41)                                 # the history budget still caps it (40 older turns)

        self.tighten(mind, whole.meta["budget"]["used"] - 500)
        tight = mind._prompt(conv, "and the last one?")
        fewer = mind._messages(conv)
        self.assertIn("history", tight.meta["budget"]["dropped"])
        self.assertLess(len(fewer), len(roomy))
        self.assertEqual(fewer[-1], Message("user", "and the last one?"))   # this exchange is never cut
        self.assertEqual(fewer[0].role, "user")

    def test_the_rules_check_gets_the_active_rules_when_it_takes_them(self):
        mind = self.mind()
        mind._rules = ["rule-object"]
        calls = []

        def v2(answer, text, rules=None):
            calls.append((answer, text, rules))
            return []

        def v1(answer, text):
            calls.append((answer, text))
            return []

        with mock.patch.object(mind_module, "rules_check", v2), mock.patch.object(mind_module, "_CHECK_TAKES_RULES", True):
            mind._rules_check("a", "b")
        with mock.patch.object(mind_module, "rules_check", v1), mock.patch.object(mind_module, "_CHECK_TAKES_RULES", False):
            mind._rules_check("c", "d")
        self.assertEqual(calls, [("a", "b", ["rule-object"]), ("c", "d")])


class UnderstandingTests(MindCase):
    def test_open_commitments_and_threads_are_in_the_prompt_and_closed_ones_are_not(self):
        mind = self.mind("Sure.")
        conv = mind.start()
        mind.store.update(conv, summary="We are porting the parser.", threads=["pick an error type"])
        mind.store.add_commitments(conv, [("I'll benchmark the tokenizer", "kgirl"), ("read the changelog first", "user"),
                                          ("tidy the imports", "kgirl")])
        self.assertEqual(mind.close_commitment(conv, "C2"), "closed: C2 read the changelog first")
        mind.chat(conv, "where were we?")
        system = self.chat.calls[0][0]
        self.assertIn("THIS CONVERSATION SO FAR:\nWe are porting the parser.\n"
                      "OPEN COMMITMENTS (not done yet; keep them in mind):\n- C1 (kgirl: I'll benchmark the tokenizer)\n"
                      "- C3 (kgirl: tidy the imports)\nOPEN THREADS:\n- pick an error type", system)
        self.assertNotIn("read the changelog first", system)
        self.assertEqual(mind.recap(conv).splitlines()[2:6],
                         ["open: pick an error type", "C1 [open] kgirl: I'll benchmark the tokenizer",
                          "C2 [done] user: read the changelog first", "C3 [open] kgirl: tidy the imports"])

    def test_close_commitment_says_what_happened(self):
        mind = self.mind()
        conv = mind.start()
        self.assertEqual(mind.close_commitment(conv, "C1"), "no open commitment matches 'C1' (none are open)")
        mind.store.add_commitments(conv, [("write the changelog entry", "kgirl")])
        self.assertEqual(mind.close_commitment(conv, "zzzz"), "no open commitment matches 'zzzz' (open: C1)")
        self.assertEqual(mind.close_commitment(conv, "changelog"), "closed: C1 write the changelog entry")
        self.assertEqual(mind.store.get(conv).commitments[0].closed_by, "user")

    def test_earlier_conversations_are_picked_by_topic_and_the_reason_is_recorded(self):
        mind = self.mind("Use thiserror.")
        earlier = mind.start("Parser port")
        mind.store.update(earlier, summary="Porting the parser to Rust.", tags=["rust", "parser"])
        unrelated = mind.start("Docs")
        mind.store.update(unrelated, summary="Writing the docs site.", tags=["docs"])
        conv = mind.start()
        reply = mind.chat(conv, "how should the Rust parser report errors?")
        system = self.chat.calls[0][0]
        self.assertIn(f'EARLIER CONVERSATIONS (same topic, or matching turns):\n- conv #{earlier} "Parser port" '
                      "(same topic: rust, parser): Porting the parser to Rust.", system)
        self.assertNotIn("Writing the docs site", system)
        meta = mind.store.last_answer(conv).meta
        self.assertEqual(meta["related"], [{"conv": earlier, "via": "tags", "tags": ["rust", "parser"]}])
        self.assertEqual(reply.text, "Use thiserror.")

    def test_the_conversations_own_tags_find_related_ones_and_other_scopes_are_left_out(self):
        mind = self.mind("Ok.")
        same = mind.start("Rust one", scope="kgirl")
        mind.store.update(same, summary="Rust work in kgirl.", tags=["rust"])
        other = mind.start("Rust two", scope="elsewhere")
        mind.store.update(other, summary="Rust work elsewhere.", tags=["rust"])
        conv = mind.start(scope="kgirl")
        mind.store.update(conv, tags=["rust"])
        mind.chat(conv, "what next?")
        system = self.chat.calls[0][0]
        self.assertIn("Rust work in kgirl.", system)
        self.assertNotIn("Rust work elsewhere.", system)
        self.assertEqual([r["conv"] for r in mind.store.last_answer(conv).meta["related"]], [same])

    def test_matching_turns_still_come_after_the_topic_matches(self):
        mind = self.mind("It finished.")
        earlier = mind.start("Zebra")
        mind.store.update(earlier, summary="Zebra migration.", tags=["zebra"])
        mind.store.append(earlier, "user", "the zebra migration is done")
        conv = mind.start()
        mind.chat(conv, "what happened with the zebra migration?")
        related = mind.store.last_answer(conv).meta["related"]
        self.assertEqual([(r["conv"], r["via"]) for r in related], [(earlier, "tags"), (earlier, "words")])
        system = self.chat.calls[0][0]
        self.assertLess(system.index("(same topic: zebra)"), system.index("user: the zebra migration is done"))


class AuditTests(MindCase):
    def test_new_workspace_audit_events_are_copied_once_and_only_when_there_is_an_audit(self):
        import dataclasses

        @dataclasses.dataclass(frozen=True)
        class Event:
            seq: int
            action: str
            target: str
            outcome: str

        class FakeWorkspace:
            def __init__(self):
                self.events = []

            def audit_since(self, seq):
                return [e for e in self.events if e.seq > seq]

        mind = self.mind()
        self.assertEqual(mind._audit_new(), [])                         # no workspace
        ws = FakeWorkspace()
        mind.workspace = ws
        self.assertEqual(mind._audit_new(), [])
        ws.events += [Event(1, "edit", "a.py", "applied"), Event(2, "run", "pytest -q", "exit 0")]
        self.assertEqual(mind._audit_new(), [{"seq": 1, "action": "edit", "target": "a.py", "outcome": "applied"},
                                             {"seq": 2, "action": "run", "target": "pytest -q", "outcome": "exit 0"}])
        self.assertEqual(mind._audit_new(), [])                         # copied once
        ws.events.append(Event(3, "undo", "a.py", "restored"))
        self.assertEqual([e["seq"] for e in mind._audit_new()], [3])
        mind.workspace = object()                                       # an older workspace without the audit API
        self.assertEqual(mind._audit_new(), [])


class EditBuildTests(MindCase):
    OLD = "def add(a, b):\n    return a - b\n"
    NEW = "def add(a, b):\n    return a + b\n"
    DIFF = "--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"

    def setup_edit(self, approve=True):
        ws = self.workspace(confirm=lambda question, detail: approve)
        (ws.root / "calc.py").write_text(self.OLD)
        mind = self.mind(workspace=ws)
        calls = []

        class Result:
            code, diff = self.NEW, self.DIFF

            def summary(self):
                return "edit ready for calc.py: candidate #2, existing 3/3, spec 2/2"

        def fake_edit(spec, target, tests):
            calls.append((spec, target, tests))
            return Result()

        mind._edit_build = fake_edit
        return mind, ws, calls

    def test_build_with_edit_returns_a_diff_and_changes_nothing(self):
        mind, ws, calls = self.setup_edit()
        (ws.root / "test_calc.py").write_text("")
        out = mind.call_tool("build", {"spec": "add must add", "edit": "calc.py", "tests": "test_calc.py"})
        self.assertIn("edit ready for calc.py: candidate #2", out)
        self.assertIn("```diff\n" + self.DIFF.rstrip() + "\n```", out)
        self.assertIn("NOT applied", out)
        self.assertEqual((ws.root / "calc.py").read_text(), self.OLD)
        self.assertEqual(calls, [("add must add", ws.root / "calc.py", [ws.root / "test_calc.py"])])
        self.assertIsNotNone(mind.last_edit)

    def test_apply_writes_the_edit_through_the_workspace_and_undo_takes_it_back(self):
        mind, ws, _ = self.setup_edit()
        mind.call_tool("build", {"spec": "add must add", "edit": "calc.py"})
        self.assertIn("edited calc.py", mind.apply_task())
        self.assertEqual((ws.root / "calc.py").read_text(), self.NEW)
        self.assertIsNone(mind.last_edit)
        ws.undo()
        self.assertEqual((ws.root / "calc.py").read_text(), self.OLD)
        self.assertEqual(mind.apply_task(), "there is no verified swarm diff waiting")

    def test_a_declined_edit_stays_waiting_and_a_changed_file_is_refused(self):
        mind, ws, _ = self.setup_edit(approve=False)
        mind.call_tool("build", {"spec": "s", "edit": "calc.py"})
        self.assertIn("declined", mind.apply_task())
        self.assertEqual((ws.root / "calc.py").read_text(), self.OLD)
        self.assertIsNotNone(mind.last_edit)                              # the user may /apply again
        (ws.root / "calc.py").write_text(self.OLD + "# touched\n")
        self.assertIn("the file changed since the builder read it", mind.apply_task())
        self.assertIsNone(mind.last_edit)
        self.assertEqual((ws.root / "calc.py").read_text(), self.OLD + "# touched\n")

    def test_no_workspace_a_missing_file_and_an_escaping_path_are_refused(self):
        plain = self.mind()
        self.assertIn("edit needs a project directory", plain.call_tool("build", {"spec": "s", "edit": "calc.py"}))
        mind, ws, calls = self.setup_edit()
        self.assertIn("is not a file in the workspace", mind.call_tool("build", {"spec": "s", "edit": "nope.py"}))
        self.assertIn("build refused", mind.call_tool("build", {"spec": "s", "edit": "../../etc/passwd"}))
        self.assertEqual(calls, [])

    def test_a_failed_edit_build_keeps_nothing_waiting(self):
        mind, ws, _ = self.setup_edit()

        class Failed:
            code, diff = "", ""

            def summary(self):
                return "edit failed for calc.py: no candidate passed"

        mind._edit_build = lambda spec, target, tests: Failed()
        out = mind.call_tool("build", {"spec": "s", "edit": "calc.py"})
        self.assertEqual(out, "edit failed for calc.py: no candidate passed")
        self.assertIsNone(mind.last_edit)

    def test_without_the_builders_edit_mode_the_tool_says_so(self):
        mind, _, _ = self.setup_edit()
        del mind._edit_build
        with mock.patch.object(Mind, "builder", return_value=types.SimpleNamespace()), \
                mock.patch.object(builder_module, "edit_build", None, create=True):
            out = mind.call_tool("build", {"spec": "s", "edit": "calc.py"})
        self.assertIn("has no edit mode", out)


class DoctorTests(MindCase):
    def test_a_build_without_doctor_says_so(self):
        mind = self.mind()
        with mock.patch.dict(sys.modules, {"kgirl.harness.doctor": None}):
            self.assertEqual(mind.doctor(), "kgirl doctor is not available in this build")

    def test_checks_are_listed_one_line_each(self):
        Check = types.SimpleNamespace
        seen = []
        fake = types.ModuleType("kgirl.harness.doctor")
        fake.collect = lambda home: seen.append(home) or [Check(name="Ollama", status="OK", detail="up", fix=""),
                                                          Check(name="KGIRL_CODE", status="FAIL", detail="unset",
                                                                fix="export KGIRL_CODE=...")]
        mind = self.mind()
        with mock.patch.dict(sys.modules, {"kgirl.harness.doctor": fake}):
            out = mind.doctor()
        self.assertEqual(out.splitlines(), ["OK   Ollama: up", "FAIL KGIRL_CODE: unset - fix: export KGIRL_CODE=...",
                                            "2 checks, 1 failing"])
        self.assertEqual(seen, [self.a.home])


class WriteTests(MindCase):
    OUTLINE = {"title": "Small tasks", "sections": [{"heading": "One", "points": ["p"]},
                                                    {"heading": "Two", "points": ["q"]}]}
    NO_ISSUES = {"issues": []}

    def test_a_piece_is_written_in_the_users_style_and_kept_as_a_draft(self):
        mind = self.mind(json.dumps(self.OUTLINE), "body one", "body two", json.dumps(self.NO_ISSUES))
        pref = self.a.soup.add("I want short sentences.", kind="preference")
        profile = self.a.soup.add("- Builds kgirl.", kind="note", tags="profile", source="evolve")
        conv = mind.start()
        draft, art = mind.write(conv, "write about small tasks")
        self.assertEqual(draft.markdown(), "# Small tasks\n\n## One\n\nbody one\n\n## Two\n\nbody two\n")
        saved = mind.store.artifact(art)
        self.assertEqual((saved["conv_id"], saved["kind"], saved["title"], saved["data"]["stage"]),
                         (conv, "draft", "Small tasks", "done"))
        self.assertIn("- Builds kgirl.", saved["data"]["style"])
        self.assertIn("- I want short sentences.", saved["data"]["style"])
        self.assertEqual(draft.recalled, [profile, pref])
        plan_prompt = self.chat.calls[0][1][0].content
        self.assertIn("STYLE (the user's; write in it):\nWho the user is and how they like to work:\n- Builds kgirl.",
                      plan_prompt)
        self.assertEqual(mind.drafts(conv), [(art, conv, "Small tasks", "done: 2 sections")])
        self.assertEqual(mind.drafts(conv + 1), [])
        self.assertEqual(mind.store.latest().id, conv)                  # keeping a draft is not conversation activity

    def test_resume_continues_an_interrupted_draft_in_its_own_conversation(self):
        mind = self.mind(json.dumps(self.OUTLINE), "body one")
        conv = mind.start()

        class Stop(Exception):
            pass

        def stop_before_the_second_section(message: str) -> None:
            if message.startswith("drafting 2/2"):
                raise Stop

        with self.assertRaises(Stop):
            mind.write(conv, "write about small tasks", progress=stop_before_the_second_section)
        (art,) = [a["id"] for a in mind.store.artifacts(conv, "draft")]
        self.assertEqual(mind.drafts(conv), [(art, conv, "Small tasks", "drafting: 1/2 sections")])

        self.chat.replies += [["body two"], [json.dumps(self.NO_ISSUES)]]
        other = mind.start()
        draft, again = mind.write(other, resume=art)                    # the conversation argument is the draft's own
        self.assertEqual(again, art)
        self.assertEqual([t for _, t in draft.sections], ["body one", "body two"])
        self.assertEqual(mind.store.artifact(art)["conv_id"], conv)
        self.assertEqual(mind.store.artifact(art)["data"]["stage"], "done")
        self.assertEqual(len(mind.store.artifacts()), 1)                # continued in place, not copied
        self.assertEqual(len(self.chat.calls), 4)                       # outline, one section, one section, critique

        with self.assertRaisesRegex(ValueError, "there is no draft #99"):
            mind.write(conv, resume=99)

    def test_the_write_tool_says_where_the_draft_went(self):
        mind = self.mind(json.dumps(self.OUTLINE), "body one", "body two", json.dumps(self.NO_ISSUES))
        out = mind.call_tool("write", {"brief": "write about small tasks"})
        self.assertTrue(out.startswith("# Small tasks\n\n## One"), out)
        self.assertTrue(out.endswith("(saved as draft #1; `kgirl write --resume 1` continues one)"), out)
        self.assertIn("write failed: ValueError: write needs a brief", mind.call_tool("write", {"brief": " "}))


class ReflectionTests(MindCase):
    REFLECTION = {"summary": "The user said hello.", "title": "Greeting",
                  "facts": ["The user's staging box is called box7."]}

    def test_reflection_runs_when_due_and_its_summary_reaches_the_next_prompt(self):
        mind = self.mind("Hello.", json.dumps(self.REFLECTION), "You said hello.", json.dumps(self.REFLECTION),
                         reflect_every=1)
        conv = mind.start()
        mind.chat(conv, "hello there")
        got = mind.store.get(conv)
        self.assertEqual((got.summary, got.title), ("The user said hello.", "Greeting"))
        self.assertEqual(mind.recap(conv), f"conversation #{conv}: Greeting\nsummary: The user said hello.\nturns: 2")
        mind.chat(conv, "what did I say?")
        self.assertIn("THIS CONVERSATION SO FAR:\nThe user said hello.", self.chat.calls[2][0])

    def test_end_reflects_over_what_is_left(self):
        mind = self.mind("Hello.", json.dumps(self.REFLECTION))
        conv = mind.start()
        mind.chat(conv, "hello there")
        self.assertEqual(mind.store.get(conv).summary, "")                    # not due yet
        self.assertEqual(mind.recap(conv), f"conversation #{conv}: (untitled)\nsummary: (not reflected yet)\nturns: 2")
        mind.end(conv)
        self.assertEqual(mind.store.get(conv).summary, "The user said hello.")

    def test_forget_keeps_a_proposal_that_another_conversation_restated(self):
        mind = self.mind("Hello.", json.dumps(self.REFLECTION), "Hello again.", json.dumps(self.REFLECTION))
        soup = self.a.soup
        first = mind.start()
        mind.chat(first, "hello there")
        mind.reflector.reflect(first, force=True)
        second = mind.start()
        mind.chat(second, "hello again")
        mind.reflector.reflect(second, force=True)   # (no curator step: it would promote it)
        (fact,) = soup.all(status="staged", kind="fact")
        self.assertEqual(fact.support, 2)                                   # two conversations said it
        self.assertEqual(mind.forget(first), f"deleted conversation #{first}: 2 turns, 1 memory fragments retired")
        kept = soup.get(fact.id)                                            # only the episode went; the fact stands
        self.assertEqual((kept.status, kept.source), ("staged", f"reflection:conv{second}"))
        self.assertEqual(observed_in(kept), {second})
        self.assertEqual(mind.forget(second), f"deleted conversation #{second}: 2 turns, 2 memory fragments retired")
        self.assertEqual(soup.get(fact.id).status, "retired")

    def test_forget_removes_the_turns_and_what_only_this_conversation_supported(self):
        mind = self.mind("Hello.", json.dumps(self.REFLECTION))
        conv = mind.start()
        mind.chat(conv, "hello there")
        mind.end(conv)
        soup = self.a.soup
        self.assertEqual(len(soup.all(status="active", kind="episode")), 1)
        self.assertEqual(len(soup.all(status="staged", kind="fact")), 1)
        self.assertEqual(mind.forget(conv), f"deleted conversation #{conv}: 2 turns, 2 memory fragments retired")
        self.assertIsNone(mind.store.get(conv))
        self.assertEqual(soup.all(status="active", kind="episode"), [])
        self.assertEqual(soup.all(status="staged"), [])
        self.assertEqual(mind.forget(conv), f"no conversation #{conv}")
        self.assertEqual(mind.recap(conv), "no such conversation")


class MessageWindowTests(MindCase):
    def test_an_old_tool_result_is_a_stub_once_its_exchange_is_over(self):
        mind = self.mind()
        conv = mind.start()
        listing = "[ls] " + "x" * 2000
        mind.store.append(conv, "user", "list the files")
        mind.store.append(conv, "assistant", "<tool>ls {}</tool>", meta={"partial": True, "tool": "ls"})
        mind.store.append(conv, "tool", listing)
        mind.store.append(conv, "assistant", "there are many files")
        mind.store.append(conv, "user", "and now?")
        messages = mind._messages(conv)
        self.assertEqual([m.role for m in messages], ["user", "assistant", "user", "assistant", "user"])
        self.assertEqual(messages[2].content, f"TOOL RESULT {listing[:500]}\n…[1505 {CLIPPED}")

    def test_within_an_exchange_the_newest_result_is_whole_and_earlier_ones_shrink(self):
        mind = self.mind()
        conv = mind.start()
        listing = "[ls] " + "x" * 2000
        mind.store.append(conv, "user", "list the files")
        mind.store.append(conv, "assistant", "<tool>ls {}</tool>", meta={"partial": True, "tool": "ls"})
        mind.store.append(conv, "tool", listing)
        self.assertEqual(mind._messages(conv)[-1].content, "TOOL RESULT " + listing)
        mind.store.append(conv, "assistant", "<tool>repos {}</tool>", meta={"partial": True, "tool": "repos"})
        mind.store.append(conv, "tool", "[repos] no repos indexed")
        messages = mind._messages(conv)
        self.assertEqual(mind.tool_chars // 3, 819)
        self.assertEqual(messages[2].content, f"TOOL RESULT {listing[:819]}\n…[1186 {CLIPPED}")
        self.assertEqual(messages[-1].content, "TOOL RESULT [repos] no repos indexed")

    def test_neighbours_of_one_role_merge_and_the_window_starts_with_the_user(self):
        mind = self.mind()
        conv = mind.start()
        mind.store.append(conv, "assistant", "output of a slash command")
        mind.store.append(conv, "user", "first")
        mind.store.append(conv, "user", "second")
        self.assertEqual(mind._messages(conv), [Message("user", "first\n\nsecond")])


class HelperTests(unittest.TestCase):
    def test_clip_result(self):
        self.assertEqual(_clip_result("short", 10), "short")
        self.assertEqual(_clip_result("abcdefghij", 4), f"abcd\n…[6 {CLIPPED}")

    def test_int(self):
        self.assertEqual((_int("7"), _int(3.9), _int(None), _int("seven"), _int([1])), (7, 3, None, None, None))


if __name__ == "__main__":
    unittest.main()
