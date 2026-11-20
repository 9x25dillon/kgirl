"""HarnessBackend over a real Assistant and Console; only the model is scripted. No textual needed.

Pins what the TUI relies on: a chat job's result carries the rules-check fields (empty when nothing was
caught), history drops partial/tool turns and keeps each answer's chips, the workspace verbs work with
and without a project directory, jobs started here are console jobs, and close() detaches from the Hub.
Written, not run.
"""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import tests.harness.ui  # noqa: F401  (sys.path)
from kgirl.harness.assistant import Assistant
from kgirl.harness.console.core import rules_meta
from kgirl.harness.hermes.router import BUILD, CHAT, JEV, SCOUT, SMITH, Router
from kgirl.harness.llm.scripted import ScriptedBackend
from kgirl.harness.mind.reflect import REFLECT_SYSTEM
from kgirl.harness.ui.backend import NO_WORKSPACE, HarnessBackend

ANSWER = "add(a, b) returns the sum."


def fake_llm(system: str, messages) -> str:
    if system.startswith(REFLECT_SYSTEM):
        return ('{"summary": "A test.", "title": "t", "threads": [], "facts": [], "preferences": [],'
                ' "lessons": [], "mistakes": []}')
    return ANSWER


class Base(unittest.TestCase):
    workspace = True

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.repo = root / "repo"
        self.repo.mkdir()
        (self.repo / "x.py").write_text("x = 1\n")
        env = mock.patch.dict(os.environ, {"KGIRL_CHUNKS_DB": "", "KGIRL_MATRIX": ""})
        env.start()
        self.addCleanup(env.stop)
        router = Router({role: [ScriptedBackend(fake_llm)] for role in (SCOUT, SMITH, JEV, CHAT, BUILD)})
        self.a = Assistant(home=root / "home", router=router, life=False)
        self.addCleanup(self.a.close)
        self.b = HarnessBackend(self.a, workspace_root=self.repo if self.workspace else None,
                                code_root=root / "not-a-repo")
        self.addCleanup(self.b.close)

    def wait(self, job_id, timeout=60.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.b.job(job_id)
            if job["status"] != "running":
                return job
            time.sleep(0.05)
        self.fail(f"job {job_id} still running")


class ChatAndHistory(Base):
    def test_chat_job_result_carries_empty_rules_fields(self):
        conv = self.b.conversation(new=True, title="t")
        job = self.wait(self.b.chat(conv, "what does add do?"))
        self.assertEqual(job["status"], "done", job.get("error"))
        res = job["result"]
        self.assertEqual(res["text"], ANSWER)
        self.assertEqual((res["rules_hits"], res["rules_left"], res["rules_revised"]), ([], [], False))
        self.assertIn(job["id"], [j["id"] for j in self.b.jobs()])

    def test_events_queue_sees_the_chat(self):
        conv = self.b.conversation(new=True)
        jid = self.b.chat(conv, "hello")
        self.wait(jid)
        seen = []
        q = self.b.events()
        while not q.empty():
            seen.append(q.get_nowait())
        self.assertTrue(any(e["type"] == "job_start" and e["job"] == jid for e in seen))
        self.assertTrue(any(e["type"] == "job_end" and e["job"] == jid for e in seen))

    def test_history_skips_partial_and_tool_turns_and_keeps_chips(self):
        store = self.b.console.store
        conv = store.new("h")
        store.append(conv, "user", "push it")
        store.append(conv, "assistant", "CALL shell", meta={"partial": True, "tool": "shell"})
        store.append(conv, "tool", "[shell] ok", meta={"tool": "shell"})
        store.append(conv, "assistant", "done", meta={"rules_hits": ["push"], "rules_left": [], "rules_revised": True})
        hist = self.b.history(conv)
        self.assertEqual([h["role"] for h in hist], ["user", "assistant"])
        self.assertEqual(hist[1]["rules_hits"], ["push"])
        self.assertNotIn("rules_hits", hist[0])

    def test_history_carries_budget_related_and_the_audit_of_its_tool_turns(self):
        store = self.b.console.store
        conv = store.new("m")
        store.append(conv, "user", "edit x")
        store.append(conv, "tool", "[edit] ok", meta={"tool": "edit", "audit": [
            {"seq": 1, "action": "edit", "target": "x.py", "outcome": "applied"}, "junk"]})
        store.append(conv, "assistant", "edited", meta={"budget": {"limit": 9, "used": 5},
                                                        "related": [{"conv": 2, "via": "words"}]})
        store.append(conv, "user", "thanks")
        store.append(conv, "assistant", "welcome")
        hist = self.b.history(conv)
        first, last = hist[1], hist[3]
        self.assertEqual(first["audit"], [{"seq": 1, "action": "edit", "target": "x.py", "outcome": "applied"}])
        self.assertEqual(first["budget"], {"limit": 9, "used": 5})
        self.assertEqual(first["related"], [{"conv": 2, "via": "words"}])
        self.assertEqual((last["audit"], last["budget"], last["related"]), ([], None, []))

    def test_write_result_splits_the_draft_footer(self):
        conv = self.b.conversation(new=True)
        footer = "# T\n\nbody\n(saved as draft #5; `kgirl write --resume 5` continues one)"
        with mock.patch.object(self.b.console.mind(conv), "call_tool", return_value=footer):
            job = self.wait(self.b.write(conv, "notes"))
        self.assertEqual(job["result"], {"markdown": "# T\n\nbody\n", "draft": 5})

    def test_ask_is_a_one_shot_job(self):
        from kgirl.harness.assistant import Answer
        before = len(self.b.conversations())
        with mock.patch.object(self.b.console.a, "ask",
                               return_value=Answer("answer", "explain", ["r:a.py:1"], "scripted:m", 12)) as ask:
            job = self.wait(self.b.ask("  what is x?  "))
        ask.assert_called_once_with("what is x?", None)
        self.assertEqual(job["kind"], "ask")
        self.assertEqual(job["result"], {"text": "answer", "intent": "explain", "backend": "scripted:m",
                                         "citations": ["r:a.py:1"], "context_tokens": 12})
        self.assertEqual(len(self.b.conversations()), before)
        with self.assertRaises(ValueError):
            self.b.ask("   ")

    def test_conversation_resume_returns_the_latest(self):
        first = self.b.conversation(new=True)
        self.assertEqual(self.b.conversation(), first)
        self.assertTrue(self.b.exists(first))
        self.assertFalse(self.b.exists(first + 1000))

    def test_close_unsubscribes(self):
        hub = self.b.console.hub
        q = self.b.events()
        self.assertIn(q, hub._clients)
        self.b.close()
        self.assertNotIn(q, hub._clients)

    def test_code_watch_without_git_is_unavailable(self):
        st = self.b.code()
        self.assertFalse(st["available"])
        self.assertFalse(st["changed"])


class Workspace(Base):
    def test_diff_and_undo_with_nothing_changed(self):
        self.assertEqual(self.b.diff(), "kgirl has changed nothing in this session")
        self.assertEqual(self.b.undo(), "nothing to undo")

    def test_recorded_change_shows_in_diff_and_undoes(self):
        ws = self.b.console.workspace
        (self.repo / "x.py").write_text("x = 2\n")
        ws.record("x.py", "x = 1\n", "x = 2\n")
        self.assertIn("+x = 2", self.b.diff())
        self.assertIn("undid", self.b.undo())
        self.assertEqual((self.repo / "x.py").read_text(), "x = 1\n")

    def test_auto_toggle(self):
        self.assertFalse(self.b.workspace()["auto_edit"])
        self.b.set_auto(True)
        self.assertTrue(self.b.console.workspace.auto_edit)
        self.assertEqual(self.b.workspace()["root"], str(self.repo.resolve()))

    def test_apply_with_nothing_waiting(self):
        conv = self.b.conversation(new=True)
        self.assertIn("no verified swarm diff", self.b.apply(conv))


class NoWorkspace(Base):
    workspace = False

    def test_workspace_verbs_explain_themselves(self):
        self.assertIsNone(self.b.workspace())
        self.assertEqual(self.b.diff(), NO_WORKSPACE)
        self.assertEqual(self.b.undo(), NO_WORKSPACE)
        self.assertEqual(self.b.set_auto(True), NO_WORKSPACE)


class RulesMeta(unittest.TestCase):
    def test_plain_turn(self):
        self.assertEqual(rules_meta(None), {"rules_hits": [], "rules_left": [], "rules_revised": False})

    def test_malformed_meta_is_ignored(self):
        turn = mock.Mock(meta={"rules_hits": "push", "rules_left": None})
        self.assertEqual(rules_meta(turn)["rules_hits"], [])

    def test_fields_pass_through(self):
        turn = mock.Mock(meta={"rules_hits": ["a"], "rules_left": ["a"], "rules_revised": True})
        self.assertEqual(rules_meta(turn), {"rules_hits": ["a"], "rules_left": ["a"], "rules_revised": True})


if __name__ == "__main__":
    unittest.main()
