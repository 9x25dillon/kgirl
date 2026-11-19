"""Offline tests for the conversation-understanding verbs of cli_mind: /context, /done, write --resume, evolve --dry-run.

The Mind is the one the mind tests use: a real assistant on a temporary home and a chat backend that replays
fixed replies. Output is captured; nothing touches a terminal, the network or ~/.kgirl.
"""

import argparse
import contextlib
import io
import json
import unittest

import tests.harness._fixtures  # noqa: F401  (sys.path)
from kgirl.harness import cli_mind
from tests.harness.mind import test_mind as base

OUTLINE = {"title": "Small tasks", "sections": [{"heading": "One", "points": ["p"]},
                                                {"heading": "Two", "points": ["q"]}]}
WRITE_REPLIES = [json.dumps(OUTLINE), "body one", "body two", json.dumps({"issues": []})]


def captured(fn, *args, **kw):
    """(return value, stdout, stderr) of a call."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        value = fn(*args, **kw)
    return value, out.getvalue(), err.getvalue()


def write_args(**kw) -> argparse.Namespace:
    values = dict(brief=None, words=None, out=None, no_revise=False, resume=None, list=False, conv=None)
    values.update(kw)
    return argparse.Namespace(**values)


class ParserTests(unittest.TestCase):
    def parse(self, *argv):
        parser = argparse.ArgumentParser()
        cli_mind.add_parsers(parser.add_subparsers(dest="cmd", required=True))
        return parser.parse_args(argv)

    def test_the_new_flags_and_verbs(self):
        a = self.parse("write", "--resume", "3")
        self.assertEqual((a.brief, a.resume, a.list, a.conv), (None, 3, False, None))
        a = self.parse("write", "a brief", "-w", "600", "--conv", "4")
        self.assertEqual((a.brief, a.words, a.conv, a.resume), ("a brief", 600, 4, None))
        self.assertTrue(self.parse("write", "--list").list)
        self.assertTrue(self.parse("evolve", "--dry-run").dry_run)
        self.assertFalse(self.parse("evolve").dry_run)
        a = self.parse("convo", "done", "7", "C2")
        self.assertEqual((a.convo_cmd, a.id, a.which), ("done", 7, "C2"))

    def test_the_help_lists_the_new_commands(self):
        for word in ("/context", "/done", "/drafts", "/write --resume", "/doctor"):
            self.assertIn(word, cli_mind.CHAT_HELP)


class SlashTests(base.MindCase):
    def slash(self, mind, conv, cmd, rest=""):
        return captured(cli_mind._slash, mind, conv, cmd, rest, lambda m: None)

    def test_context_before_and_after_a_turn(self):
        mind = self.mind("Hello.")
        conv = mind.start()
        self.assertIn("no prompt has been built", self.slash(mind, conv, "context")[1])
        mind.chat(conv, "hi")
        text = self.slash(mind, conv, "context")[1]
        self.assertIn("prompt budget", text)
        self.assertRegex(text, r"PERSONA\s+\d+\s+never cut")

    def test_doctor_prints_the_minds_health_report(self):
        mind = self.mind()
        mind.doctor = lambda: "OK   Ollama: up\n1 checks, none failing"
        self.assertEqual(self.slash(mind, mind.start(), "doctor")[1], "OK   Ollama: up\n1 checks, none failing\n")

    def test_done_lists_closes_and_explains(self):
        mind = self.mind()
        conv = mind.start()
        self.assertIn("no open commitments in this conversation", self.slash(mind, conv, "done")[1])
        mind.store.add_commitments(conv, [("write the changelog entry", "kgirl"), ("read the docs", "user")])
        self.assertEqual(self.slash(mind, conv, "done")[1],
                         "C1 kgirl: write the changelog entry\nC2 user: read the docs\n")
        self.assertEqual(self.slash(mind, conv, "done", "C1")[1], "closed: C1 write the changelog entry\n")
        self.assertEqual(self.slash(mind, conv, "done")[1], "C2 user: read the docs\n")
        self.assertIn("no open commitment matches 'C9' (open: C2)", self.slash(mind, conv, "done", "C9")[1])

    def test_write_drafts_and_resume_in_chat(self):
        mind = self.mind(*WRITE_REPLIES)
        conv = mind.start()
        _, out, _ = self.slash(mind, conv, "write", "about small tasks")
        self.assertIn("# Small tasks\n\n## One\n\nbody one", out)
        self.assertIn("(draft #1 saved; /write --resume 1 or `kgirl write --resume 1` continues it)", out)
        self.assertEqual(mind.store.turns(conv)[-1].role, "tool")                 # the piece is in the conversation
        self.assertRegex(self.slash(mind, conv, "drafts")[1], r"#1\s+Small tasks\s+done: 2 sections")
        calls = len(self.chat.calls)
        _, again, _ = self.slash(mind, conv, "write", "--resume 1")               # a finished draft: no model call
        self.assertIn("body two", again)
        self.assertEqual(len(self.chat.calls), calls)
        self.assertIn("there is no draft #99", self.slash(mind, conv, "write", "--resume 99")[1])

    def test_evolve_dry_run_calls_no_model_and_says_so(self):
        mind = self.mind()
        conv = mind.start()
        mind.store.append(conv, "user", "hello")
        _, out, _ = self.slash(mind, conv, "evolve", "--dry-run")
        self.assertTrue(out.startswith("dry run: nothing was changed, no model was called"), out)
        self.assertIn(f"would reflect on conversations: [{conv}]", out)
        self.assertEqual(self.chat.calls, [])
        self.assertEqual(mind.store.get(conv).reflected_turn, 0)


class WriteVerbTests(base.MindCase):
    def test_list_with_nothing_saved_and_the_argument_errors(self):
        mind = self.mind()
        self.assertEqual(captured(cli_mind._write, mind, write_args(list=True))[:2], (0, "no drafts yet\n"))
        code, _, err = captured(cli_mind._write, mind, write_args())
        self.assertEqual(code, 2)
        self.assertIn("write needs a brief, or --resume ID", err)
        code, _, err = captured(cli_mind._write, mind, write_args(brief="x", conv=42))
        self.assertEqual(code, 1)
        self.assertIn("no conversation #42", err)
        code, _, err = captured(cli_mind._write, mind, write_args(resume=9))
        self.assertEqual(code, 1)
        self.assertIn("there is no draft #9", err)

    def test_a_piece_is_written_kept_listed_and_exported(self):
        mind = self.mind(*WRITE_REPLIES)
        conv = mind.start()
        out_file = self.base / "piece.md"
        code, out, err = captured(cli_mind._write, mind, write_args(brief="about small tasks", conv=conv,
                                                                    out=str(out_file)))
        self.assertEqual(code, 0)
        self.assertEqual(out, "")
        self.assertTrue(out_file.read_text().startswith("# Small tasks"))
        self.assertIn("draft #1 saved (done: 2 sections); `kgirl write --resume 1` continues it", err)
        _, listing, _ = captured(cli_mind._write, mind, write_args(list=True, conv=conv))
        self.assertRegex(listing, rf"#1\s+conv #{conv}\s+Small tasks\s+done: 2 sections")
        _, shown, _ = captured(cli_mind._convo, mind, argparse.Namespace(convo_cmd="show", id=conv))
        self.assertIn("draft #1: Small tasks (done: 2 sections)", shown)
        _, exported, _ = captured(cli_mind._convo, mind, argparse.Namespace(convo_cmd="export", id=conv, out=None))
        self.assertIn("**Drafts:**\n- #1 Small tasks (done)", exported)

    def test_resume_continues_an_interrupted_draft_and_ctrl_c_leaves_it_saved(self):
        mind = self.mind(json.dumps(OUTLINE), "body one")
        conv = mind.start()
        real = mind._llm_text
        state = {"n": 0}

        def interrupt_on_the_second_draft(*a, **kw):
            state["n"] += 1
            if state["n"] == 3:                                                   # outline, section 1, then Ctrl-C
                raise KeyboardInterrupt
            return real(*a, **kw)

        mind._llm_text = interrupt_on_the_second_draft
        code, _, err = captured(cli_mind._write, mind, write_args(brief="about small tasks", conv=conv))
        self.assertEqual(code, 130)
        self.assertIn("draft #1 is saved: kgirl write --resume 1", err)
        mind._llm_text = real
        self.chat.replies += [["body two"], [json.dumps({"issues": []})]]
        code, out, err = captured(cli_mind._write, mind, write_args(resume=1))
        self.assertEqual(code, 0)
        self.assertIn("## Two\n\nbody two", out)
        self.assertIn("draft #1 saved (done: 2 sections)", err)

    def test_convo_done_closes_a_commitment_by_label(self):
        mind = self.mind()
        conv = mind.start()
        mind.store.add_commitments(conv, [("write the changelog entry", "kgirl")])
        _, out, _ = captured(cli_mind._convo, mind, argparse.Namespace(convo_cmd="done", id=conv, which="C1"))
        self.assertEqual(out, "closed: C1 write the changelog entry\n")
        code, _, err = captured(cli_mind._convo, mind, argparse.Namespace(convo_cmd="done", id=99, which="C1"))
        self.assertEqual(code, 1)
        self.assertIn("no conversation #99", err)


if __name__ == "__main__":
    unittest.main()
