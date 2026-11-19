"""`kgirl ui` driven by textual's pilot against a fake backend: nothing is generated, no harness is built.

Pins the console-parity essentials: the resumed history and its rules chips, a streamed answer and its
chips, approvals answered from a modal (and closed when they time out elsewhere), /diff /undo /apply
/auto reaching the workspace, /task /build /helix starting jobs that show in the tray, the CodeWatch
restart hint, the help overlay and the stacked layout on a narrow terminal. Skipped without textual.
Written, not run.
"""

import importlib.util
import tempfile
import time
import unittest
from pathlib import Path

import tests.harness.ui  # noqa: F401  (sys.path)
from tests.harness.ui.fakes import FakeBackend

HAVE_TEXTUAL = importlib.util.find_spec("textual") is not None
if HAVE_TEXTUAL:
    from kgirl.harness.ui.app import ApprovalScreen, ComposeScreen, HelpScreen, KgirlUI

WIDE, NARROW = (150, 45), (90, 40)
REVISED = {"rules_hits": ["force-push"], "rules_left": [], "rules_revised": True}
STILL = {"rules_hits": ["force-push"], "rules_left": ["force-push"], "rules_revised": True}


def text_of(log) -> str:
    """The log's text with whitespace collapsed: RichLog wraps to the pane width, so a phrase may span lines."""
    return " ".join(" ".join(line.text for line in log.lines).split())


@unittest.skipUnless(HAVE_TEXTUAL, "textual is not installed")
class KgirlUIPilot(unittest.IsolatedAsyncioTestCase):
    def make(self, **kw):
        self.backend = FakeBackend(**kw)
        app = KgirlUI(self.backend, pump_interval=0.01)
        app.STATUS_EVERY = app.CODE_EVERY = 3600     # one probe at mount, none during the test
        return app

    async def settle(self, pilot, n=3):
        for _ in range(n):
            await pilot.pause(0.03)

    def chat(self, app) -> str:
        return text_of(app.query_one("#chatlog"))

    async def submit(self, pilot, app, text):
        prompt = app.query_one("#prompt")
        prompt.focus()
        prompt.value = text
        await pilot.press("enter")
        await self.settle(pilot)

    # ---------------------------------------------------------------- resume + chips
    async def test_resumes_latest_with_history_and_rules_chips(self):
        app = self.make(history=[{"role": "user", "text": "push it"},
                                 {"role": "assistant", "text": "pushed with a plain push", **REVISED}])
        async with app.run_test(size=WIDE) as pilot:
            await self.settle(pilot)
            out = self.chat(app)
            self.assertIn("conversation #7", out)
            self.assertIn("push it", out)
            self.assertIn("rules check: revised (force-push)", out)
            self.assertEqual(self.backend.calls[0], ("conversation", False, ""))
            self.assertIn("#7", str(app.query_one("#left").border_title))

    # ---------------------------------------------------------------- streaming chat
    async def test_streamed_answer_ends_with_chips_and_timing(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            await self.submit(pilot, app, "how do I deploy?")
            self.assertEqual(self.backend.calls[-1], ("chat", 7, "how do I deploy?"))
            jid = app.chat_job
            self.assertIsNotNone(jid)
            self.backend.push("job", job=jid, event={"type": "delta", "text": "first line\nsecond "})
            self.backend.push("job", job=jid, event={"type": "delta", "text": "half"})
            self.backend.finish(jid, result={"text": "first line\nsecond half", "recalled": [4], "stored": [], **STILL})
            await self.settle(pilot)
            out = self.chat(app)
            self.assertIn("first line", out)
            self.assertIn("second half", out)
            self.assertIn("still breaks force-push", out)
            self.assertIn("recalled 1", out)
            self.assertIsNone(app.chat_job)

    async def test_unstreamed_answer_is_shown_from_the_result(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            await self.submit(pilot, app, "hi")
            self.backend.finish(app.chat_job, result={"text": "hello from the result"})
            await self.settle(pilot)
            self.assertIn("hello from the result", self.chat(app))

    async def test_second_message_waits_for_the_first(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            await self.submit(pilot, app, "one")
            await self.submit(pilot, app, "two")
            self.assertEqual([c for c in self.backend.calls if c[0] == "chat"], [("chat", 7, "one")])
            self.assertIn("still answering", self.chat(app))

    async def test_failed_chat_is_reported(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            await self.submit(pilot, app, "hi")
            self.backend.finish(app.chat_job, status="error", error="ConnectionError: no model")
            await self.settle(pilot)
            self.assertIn("chat failed: ConnectionError: no model", self.chat(app))
            self.assertIsNone(app.chat_job)

    # ---------------------------------------------------------------- approvals
    async def test_approval_modal_yes_is_sent(self):
        app = self.make(workspace={"root": "/tmp/repo", "auto_edit": False, "allowed": []})
        async with app.run_test(size=WIDE) as pilot:
            self.backend.push("approval", id="ap1", question="edit x.py?", detail="-old\n+new")
            await self.settle(pilot)
            self.assertIsInstance(app.screen, ApprovalScreen)
            await pilot.press("y")
            await self.settle(pilot)
            self.assertIn(("approve", "ap1", True), self.backend.calls)
            self.assertNotIsInstance(app.screen, ApprovalScreen)
            self.assertIn("approved: edit x.py?", self.chat(app))

    async def test_approvals_queue_and_escape_declines(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            self.backend.push("approval", id="a", question="edit a?", detail="")
            self.backend.push("approval", id="b", question="run b?", detail="")
            await self.settle(pilot)
            await pilot.press("escape")
            await self.settle(pilot)
            self.assertIn(("approve", "a", False), self.backend.calls)
            self.assertIsInstance(app.screen, ApprovalScreen)
            self.assertEqual(app.screen.approval_id, "b")
            await pilot.press("n")
            await self.settle(pilot)
            self.assertIn(("approve", "b", False), self.backend.calls)

    async def test_timed_out_approval_closes_without_an_answer(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            self.backend.push("approval", id="ap2", question="edit y.py?", detail="")
            await self.settle(pilot)
            self.backend.push("approval_done", id="ap2", answer=False)
            await self.settle(pilot)
            self.assertNotIsInstance(app.screen, ApprovalScreen)
            self.assertFalse([c for c in self.backend.calls if c[0] == "approve"])

    async def test_workspace_notice_is_shown(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            self.backend.push("workspace", text="edited x.py (+1 -1)")
            await self.settle(pilot)
            self.assertIn("edited x.py (+1 -1)", self.chat(app))

    # ---------------------------------------------------------------- workspace commands
    async def test_diff_undo_apply_auto(self):
        ws = {"root": "/tmp/repo", "auto_edit": False, "allowed": []}
        app = self.make(workspace=ws)
        async with app.run_test(size=WIDE) as pilot:
            for cmd in ("/diff", "/undo", "/apply", "/auto on"):
                await self.submit(pilot, app, cmd)
            names = [c[0] for c in self.backend.calls]
            for name in ("diff", "undo", "apply", "auto"):
                self.assertIn(name, names)
            self.assertIn(("apply", 7), self.backend.calls)
            self.assertIn(("auto", True), self.backend.calls)
            out = self.chat(app)
            self.assertIn("+new", out)
            self.assertIn("undid the change to x.py", out)
            self.assertIn("edits auto", str(app.query_one("#mode").content))

    async def test_auto_needs_on_or_off(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            await self.submit(pilot, app, "/auto maybe")
            self.assertNotIn("auto", [c[0] for c in self.backend.calls])
            self.assertIn("/auto on|off", self.chat(app))

    # ---------------------------------------------------------------- jobs
    async def test_task_build_helix_start_jobs_in_the_tray(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            await self.submit(pilot, app, "/task fix add in calc.py")
            await self.submit(pilot, app, "/build add two ints")
            await self.submit(pilot, app, "/helix plan the week")
            tray = str(app.query_one("#jobs").content)
            for kind in ("task", "build", "helix"):
                self.assertIn(kind, tray)
            self.assertIn(("task", 7, "fix add in calc.py"), self.backend.calls)
            task_id = next(j["id"] for j in self.backend.jobs() if j["kind"] == "task")
            self.backend.finish(task_id, result={"files": [{"path": "calc.py", "diff": "-a - b\n+a + b"}],
                                                 "text": "verified"})
            await self.settle(pilot)
            self.assertIn("/apply writes it", self.chat(app))
            await self.submit(pilot, app, f"/job {task_id[:6]}")
            self.assertIn("+a + b", self.chat(app))

    async def test_build_reads_a_spec_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = Path(tmp) / "spec.md"
            spec.write_text("Implement add(a, b).\n")
            app = self.make()
            async with app.run_test(size=WIDE) as pilot:
                await self.submit(pilot, app, f"/build @{spec}")
                self.assertIn(("build", "Implement add(a, b).\n"), self.backend.calls)
                await self.submit(pilot, app, f"/build @{spec}.missing")
                self.assertIn("cannot read", self.chat(app))

    async def test_explain_shows_the_latest_build_votes(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            await self.submit(pilot, app, "/explain")
            self.assertIn("no finished build yet", self.chat(app))
            await self.submit(pilot, app, "/build add two ints")
            bid = next(j["id"] for j in self.backend.jobs() if j["kind"] == "build")
            self.backend.finish(bid, result={"confidence": 0.9, "explain": "candidate 2 won 3/3 suites"})
            await self.settle(pilot)
            self.assertIn("confidence 0.90", self.chat(app))
            await self.submit(pilot, app, "/explain")
            self.assertIn("candidate 2 won 3/3 suites", self.chat(app))

    async def test_failed_job_and_stuck_job(self):
        long_ago = time.time() - 600
        stuck = {"id": "ta0000000099", "kind": "task", "label": "slow goal", "status": "running",
                 "created": long_ago, "ended": None, "error": "", "last_event": long_ago}
        app = self.make(jobs=[stuck])
        async with app.run_test(size=WIDE) as pilot:
            await self.settle(pilot)
            self.assertIn("stuck?", str(app.query_one("#jobs").content))
            self.backend.finish("ta0000000099", status="error", error="TimeoutError: swarm")
            await self.settle(pilot)
            self.assertIn("failed: TimeoutError: swarm", self.chat(app))

    # ---------------------------------------------------------------- restart hint
    async def test_code_change_shows_the_restart_hint_once(self):
        app = self.make(code={"available": True, "root": "/tmp/code", "startup": "aaa", "current": "bbb",
                              "changed": True})
        async with app.run_test(size=WIDE) as pilot:
            await self.settle(pilot, 6)
            self.assertIn("restart kgirl ui", str(app.query_one("#mode").content))
            app.check_code()
            await self.settle(pilot, 6)
            self.assertEqual(self.chat(app).count("changed since kgirl ui started"), 1)

    async def test_no_hint_while_code_matches(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            await self.settle(pilot, 6)
            self.assertNotIn("restart", str(app.query_one("#mode").content))

    # ---------------------------------------------------------------- the rest
    async def test_new_conversation_and_feedback_and_rules(self):
        app = self.make(history=[{"role": "user", "text": "old message"}])
        async with app.run_test(size=WIDE) as pilot:
            await self.submit(pilot, app, "/good")
            self.assertIn(("feedback", 7, "positive"), self.backend.calls)
            await self.submit(pilot, app, "/rules")
            self.assertIn("never force-push", self.chat(app))
            await self.submit(pilot, app, "/new release notes")
            self.assertIn(("conversation", True, "release notes"), self.backend.calls)
            self.assertNotIn("old message", self.chat(app))
            self.assertEqual(app.conv, 7)

    async def test_search_pane(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            await self.submit(pilot, app, "/s do thing")
            await self.settle(pilot, 6)
            self.assertIn(("search", "do thing"), self.backend.calls)
            self.assertIn("do_thing", text_of(app.query_one("#hits")))

    async def test_help_overlay_and_unknown_command(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.press("f1")
            await self.settle(pilot)
            self.assertIsInstance(app.screen, HelpScreen)
            await pilot.press("escape")
            await self.settle(pilot)
            self.assertNotIsInstance(app.screen, HelpScreen)
            await self.submit(pilot, app, "/nope")
            self.assertIn("unknown command /nope", self.chat(app))

    async def test_composer_sends_a_multiline_message(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            await pilot.press("ctrl+o")
            await self.settle(pilot)
            self.assertIsInstance(app.screen, ComposeScreen)
            app.screen.query_one("#compose-text").text = "line one\nline two"
            await pilot.press("ctrl+s")
            await self.settle(pilot)
            self.assertNotIsInstance(app.screen, ComposeScreen)
            self.assertIn(("chat", 7, "line one\nline two"), self.backend.calls)

    async def test_composer_builds_a_spec_and_cancels_cleanly(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            await self.submit(pilot, app, "/paste Implement add(a, b).")
            self.assertIsInstance(app.screen, ComposeScreen)
            await pilot.press("ctrl+b")
            await self.settle(pilot)
            self.assertIn(("build", "Implement add(a, b)."), self.backend.calls)
            await pilot.press("ctrl+o")
            await self.settle(pilot)
            await pilot.press("escape")
            await self.settle(pilot)
            self.assertNotIsInstance(app.screen, ComposeScreen)
            self.assertEqual(len([c for c in self.backend.calls if c[0] in ("chat", "build")]), 1)

    # ---------------------------------------------------------------- round 2: meta, drafts, doctor, md
    async def test_answer_shows_budget_related_and_audit(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            await self.submit(pilot, app, "fix it")
            self.backend.finish(app.chat_job, result={
                "text": "Fixed.", "budget": {"limit": 4000, "used": 3100, "overflow": 0,
                                             "blocks": {"memory": 300}, "dropped": {"earlier": {"items": 2}}},
                "related": [{"conv": 3, "via": "tags", "tags": ["deploy"]}, {"conv": 5, "via": "words", "turn": 9}],
                "audit": [{"seq": 1, "action": "edit", "target": "x.py", "outcome": "applied"},
                          {"seq": 2, "action": "run", "target": "rm -rf /", "outcome": "refused"}]})
            await self.settle(pilot)
            out = self.chat(app)
            self.assertIn("context 3100/4000 · cut earlier", out)
            self.assertIn("conv #3 (same topic: deploy)", out)
            self.assertIn("conv #5 (matching words)", out)
            self.assertIn("edit x.py → applied", out)
            self.assertIn("run rm -rf / → refused", out)

    async def test_history_without_new_meta_shows_no_meta_lines(self):
        app = self.make(history=[{"role": "assistant", "text": "old answer"}])
        async with app.run_test(size=WIDE) as pilot:
            await self.settle(pilot)
            out = self.chat(app)
            self.assertIn("old answer", out)
            self.assertNotIn("context ", out)
            self.assertNotIn("related:", out)

    async def test_streaming_goes_to_the_live_pane_then_the_log(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            await self.submit(pilot, app, "hi")
            self.backend.push("job", job=app.chat_job, event={"type": "delta", "text": "partial **bold"})
            await self.settle(pilot)
            live = app.query_one("#live")
            self.assertTrue(live.display)
            self.assertIn("partial **bold", str(live.content))
            self.assertNotIn("partial", self.chat(app))
            self.backend.finish(app.chat_job, result={"text": "final **bold** answer"})
            await self.settle(pilot)
            self.assertFalse(live.display)
            out = self.chat(app)
            self.assertIn("final bold answer", out)        # rendered as Markdown: no asterisks
            self.assertNotIn("partial", out)

    async def test_md_off_keeps_the_raw_text(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            await self.submit(pilot, app, "/md off")
            await self.submit(pilot, app, "hi")
            self.backend.finish(app.chat_job, result={"text": "a **raw** answer"})
            await self.settle(pilot)
            self.assertIn("a **raw** answer", self.chat(app))

    async def test_write_drafts_resume(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            await self.submit(pilot, app, "/write release notes for 1.2")
            self.assertIn(("write", 7, "release notes for 1.2"), self.backend.calls)
            wid = next(j["id"] for j in self.backend.jobs() if j["kind"] == "write")
            self.backend.finish(wid, result={"markdown": "# Notes\n", "draft": 4})
            await self.settle(pilot)
            self.assertIn("saved as draft #4", self.chat(app))
            await self.submit(pilot, app, "/drafts")
            self.assertIn("release notes", self.chat(app))
            await self.submit(pilot, app, "/resume 4")
            self.assertIn(("resume", 4), self.backend.calls)
            await self.submit(pilot, app, "/resume x")
            self.assertIn("/resume DRAFT_ID", self.chat(app))

    async def test_doctor_context_verify_allow(self):
        app = self.make(workspace={"root": "/tmp/repo", "auto_edit": False, "allowed": []})
        async with app.run_test(size=WIDE) as pilot:
            await self.submit(pilot, app, "/doctor")
            await self.settle(pilot, 6)
            out = self.chat(app)
            self.assertIn("chunk index: missing", out)
            self.assertIn("fix: run kgirl chunks", out)
            self.assertIn("2 checks · 1 failing", out)
            await self.submit(pilot, app, "/context")
            self.assertIn("nothing was cut", self.chat(app))
            await self.submit(pilot, app, "/verify python3 -m pytest -q")
            self.assertIn(("verify", "python3 -m pytest -q"), self.backend.calls)
            await self.submit(pilot, app, "/allow make test")
            self.assertIn(("allow", "make test"), self.backend.calls)
            self.assertIn("kgirl may now run without asking: make test", self.chat(app))

    async def test_narrow_terminal_stacks_the_panes(self):
        app = self.make()
        async with app.run_test(size=NARROW) as pilot:
            await self.settle(pilot)
            self.assertTrue(app.has_class("-narrow"))
            left, right = app.query_one("#left"), app.query_one("#right")
            self.assertLess(left.region.y, right.region.y)

    async def test_wide_terminal_is_side_by_side(self):
        app = self.make()
        async with app.run_test(size=WIDE) as pilot:
            await self.settle(pilot)
            self.assertFalse(app.has_class("-narrow"))
            self.assertLess(app.query_one("#left").region.x, app.query_one("#right").region.x)


if __name__ == "__main__":
    unittest.main()
