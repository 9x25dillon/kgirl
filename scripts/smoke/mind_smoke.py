#!/usr/bin/env python3
"""Exercise the real mind pipeline with fixed replies and disposable state; no model server."""

from __future__ import annotations

import json
import os
import socket
import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from _support import Stages, isolated_home, require

SPEC = "Implement add(a: int, b: int) -> int, returning the sum of two integers."
CODE = "def add(a: int, b: int) -> int:\n    return a + b\n"
SUITE = '''import unittest
from solution import add

class Arithmetic(unittest.TestCase):
    def test_positive(self):
        self.assertEqual(add(2, 3), 5)
    def test_zero(self):
        self.assertEqual(add(0, 0), 0)
    def test_negative(self):
        self.assertEqual(add(-2, 1), -1)
'''
PROBE = SUITE + '''
class Outcomes(unittest.TestCase):
    def test_failure(self):
        self.assertEqual(add(2, 3), 99)
    def test_error(self):
        raise ValueError("intentional smoke error")
    @unittest.skip("intentional smoke skip")
    def test_skip(self):
        self.fail("a skipped case must not run")
'''
ANSWER = "Addition returns the sum of two integers."
FACT = "The smoke project implements integer addition."
REFLECTION = {
    "summary": "Discussed integer addition for the smoke project.",
    "title": "Smoke project addition",
    "threads": ["Build the addition module."],
    "facts": [FACT], "preferences": [], "lessons": [], "mistakes": [], "feedback": {},
}


def fenced(text):
    return f"```python\n{text}\n```"


def main():
    print(f"Mind smoke on Python {sys.version.split()[0]} (fixed fixtures; no quality measurement)")
    stages = Stages()
    state = SimpleNamespace()
    attempts = []

    def deny_network(*args, **kwargs):
        attempts.append("network call")
        raise RuntimeError("network access is disabled in the mind smoke script")

    with isolated_home() as root, ExitStack() as cleanup:
        # Only the source literals above execute in the child. Avoid host-specific isolation probes.
        # Never replace this fake backend with a real model while sandbox isolation is disabled.
        os.environ["KGIRL_SANDBOX"] = "off"
        for target in ((socket.socket, "connect"), (socket.socket, "connect_ex"),
                       (socket, "create_connection"), (socket, "getaddrinfo")):
            cleanup.enter_context(patch.object(*target, side_effect=deny_network))

        def imports():
            from kgirl.harness.assistant import Assistant
            from kgirl.harness.hermes.router import BUILD, CHAT, JEV, SCOUT, SMITH, Router
            from kgirl.harness.llm.scripted import ScriptedBackend
            from kgirl.harness.mind import Builder, Mind, Writer
            from kgirl.harness.mind import builder, reflect, writer
            from kgirl.harness.mind.mind import PERSONA

            class FakeBackend(ScriptedBackend):
                reflection = "```json\n" + json.dumps(REFLECTION) + "\n```"
                streamed = 0

                def stream(self, system, messages, **kwargs):
                    text = self.complete(system, messages, **kwargs).text
                    for offset in range(0, len(text), 5):
                        self.streamed += 1
                        yield text[offset:offset + 5]

            def respond(system, messages):
                if system.startswith(PERSONA):
                    return ANSWER
                if system == reflect.REFLECT_SYSTEM:
                    return state.backend.reflection
                if system == builder.TEST_SYSTEM:
                    return fenced(SUITE)
                if system == builder.IMPL_SYSTEM:
                    return fenced(CODE)
                if system == writer.PLAN_SYSTEM:
                    require(CODE.strip() in messages[-1].content, "writer did not receive the built module")
                    return json.dumps({"title": "Addition guide", "audience": "beginners", "thesis": ANSWER,
                                       "sections": [{"heading": heading, "points": [ANSWER], "words": 30}
                                                    for heading in ("Purpose", "Example", "Limits")]})
                if system == writer.DRAFT_SYSTEM:
                    return "The add function returns the sum of its two integer arguments."
                if system == writer.CRITIC_SYSTEM:
                    return json.dumps({"issues": [{"section": "Example", "problem": "Missing example",
                                                   "fix": "Show add(2, 3) returning 5."}]})
                if system == writer.REVISE_SYSTEM:
                    return "For example, add(2, 3) returns 5."
                raise RuntimeError("unexpected model request in smoke fixture")

            state.backend = FakeBackend(respond)
            router = Router({CHAT: [state.backend], BUILD: [state.backend], SCOUT: [], SMITH: [], JEV: []})
            state.assistant = Assistant(home=root / "home", router=router, trace=False, life=False)
            cleanup.callback(state.assistant.close)
            state.mind = Mind(state.assistant, background_reflection=False, reflect_every=99)
            cleanup.callback(state.mind.store.close)
            state.Builder, state.Writer, state.builder = Builder, Writer, builder
            require(state.assistant.home == root / "home", "assistant escaped scratch home")
            return "mind, router, stores, builder and writer imported"

        def chat():
            state.conv = state.mind.start("Smoke addition")
            chunks = []
            reply = state.mind.chat(state.conv, "Explain addition of integers.", on_text=chunks.append)
            require(reply.text == ANSWER, "unexpected chat reply")
            require("".join(chunks).strip() == ANSWER and len(chunks) > 1, "streamed text was lost")
            require(state.backend.streamed > 1, "fake backend did not stream")
            answer = state.mind.store.last_answer(state.conv)
            require(answer is not None and answer.content == ANSWER, "answer was not stored")
            require(bool(state.mind.store.context(answer.id)), "answer context was not stored")
            return "streamed pieces match the persisted answer"

        def reflect():
            ref = state.mind.reflector.reflect(state.conv, force=True)
            require(ref is not None and ref.summary == REFLECTION["summary"], "fenced JSON reflection failed")
            require(ref.facts == [FACT] and len(ref.proposed) == 1, "reflection proposal missing")
            state.fact_id = ref.proposed[0][0]
            require(state.assistant.soup.get(state.fact_id).status == "staged", "single observation was promoted")
            bad = state.mind.start("Malformed reflection")
            state.mind.store.append(bad, "user", "Explain addition.")
            state.mind.store.append(bad, "assistant", ANSWER)
            valid = state.backend.reflection
            try:
                state.backend.reflection = "This is not JSON."
                require(state.mind.reflector.reflect(bad, force=True) is None, "malformed JSON was accepted")
                require(not state.mind.store.get(bad).summary, "malformed reflection changed the summary")
            finally:
                state.backend.reflection = valid
            return "valid JSON staged a fact; malformed JSON left the conversation unchanged"

        def curate():
            assistant = state.assistant
            first = assistant.curator.step()
            require(state.fact_id not in first.promoted, "one observation was enough for promotion")
            another = state.mind.start("Independent smoke conversation")
            state.mind.chat(another, "Explain addition of integers.")
            ref = state.mind.reflector.reflect(another, force=True)
            require(ref is not None and (state.fact_id, "reinforced") in ref.proposed,
                    "second conversation did not reinforce the fact")
            second = assistant.curator.step()
            require(state.fact_id in second.promoted, "reinforced fact did not pass the default promotion policy")
            recalled = assistant.soup.recall("smoke project integer addition", kinds=("fact",), fallback=False)
            require(state.fact_id in recalled.ids, "promoted fact was not recalled")
            return "independent reinforcement promoted a recallable fact"

        def run_suite():
            passed, failed = state.builder.run_suite(CODE, PROBE, "test_smoke", timeout=5)
            expected = {f"test_smoke:Arithmetic.test_{name}" for name in ("positive", "zero", "negative")}
            require(passed == expected, f"unexpected passed ids: {sorted(passed)}")
            require(set(failed) == {"test_smoke:Outcomes.test_failure", "test_smoke:Outcomes.test_error"},
                    f"unexpected failures: {failed}")
            require("AssertionError" in failed["test_smoke:Outcomes.test_failure"], "failure detail was lost")
            require("ValueError" in failed["test_smoke:Outcomes.test_error"], "error detail was lost")
            return "real child results: 3 passes, 2 intentional failures, skip excluded"

        def build():
            builder = state.Builder(state.mind._code_llm, soup=state.assistant.soup,
                                    n_impl=2, n_suites=1, repair_rounds=0, workers=1, timeout=5)
            state.built = builder.build(SPEC)
            require(state.built.code.strip() == CODE.strip(), "builder did not deliver the fixture module")
            require(state.built.passed == 3 and state.built.credible == 3, "builder lost suite results")
            require(state.built.group_size == 2 and bool(state.built.agreed), "agreement or pruned suite missing")
            fid = builder.learn(state.built, accepted=False, origin="mcp")
            require(fid is not None and state.assistant.soup.get(fid).status == "staged",
                    "unreviewed exemplar was not staged")
            state.assistant.curator.step()
            require(state.assistant.soup.get(fid).status == "staged", "MCP exemplar promoted without review")
            return "two fixed implementations agreed; unreviewed exemplar remains staged"

        def write():
            draft = state.Writer(state.mind._llm_text).write(
                "Explain the addition module to a beginner.", context=state.built.code, words=120)
            require(draft.title == "Addition guide" and len(draft.sections) == 3, "outline/draft missing")
            require(len(draft.issues) == 1, "critique missing")
            require(dict(draft.sections)["Example"] == "For example, add(2, 3) returns 5.", "revision missing")
            require(draft.markdown().startswith("# Addition guide\n"), "Markdown rendering failed")
            return "built code reached plan, draft, critique and revision"

        stages.check("imports", imports)
        stages.check("chat", chat, needs=("imports",))
        stages.check("reflect", reflect, needs=("chat",))
        stages.check("curate", curate, needs=("reflect",))
        stages.check("run_suite", run_suite, needs=("imports",))
        stages.check("builder", build, needs=("imports", "run_suite"))
        stages.check("writer", write, needs=("builder",))
        stages.check("offline", lambda: require(not attempts, "a network operation was attempted"))
        return stages.status()


if __name__ == "__main__":
    raise SystemExit(main())
