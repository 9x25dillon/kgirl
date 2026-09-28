import unittest

import tests.harness._fixtures  # noqa: F401
from kgirl.harness.hermes import Bus, HermesError, Intent, NoBackend, Router, classify
from kgirl.harness.llm import BackendError, Message, ScriptedBackend


class BusTests(unittest.TestCase):
    def test_glob_order_cascade_and_causality(self):
        bus, seen = Bus(), []
        bus.subscribe("jev.*", lambda e: seen.append(("a", e.topic)))
        bus.subscribe("jev.step", lambda e: [("soup.note", {"from": e.id})])
        bus.subscribe("soup.*", lambda e: seen.append(("b", e.topic)))
        root = bus.publish("jev.step", {"n": 1}, "test")
        self.assertEqual(seen, [("a", "jev.step"), ("b", "soup.note")])
        chain = bus.trace(correlation_id=root.id)
        self.assertEqual([e.topic for e in chain], ["jev.step", "soup.note"])
        self.assertEqual(chain[1].causation_id, root.id)

    def test_loops_are_bounded_and_handler_errors_isolated(self):
        bus = Bus(max_hops=5)
        bus.subscribe("ping", lambda e: [("ping", None)])
        bus.subscribe("ping", lambda e: 1 / 0)
        bus.publish("ping")
        topics = [e.topic for e in bus.trace()]
        self.assertIn("hermes.dropped", topics)
        self.assertIn("hermes.handler_error", topics)
        self.assertLessEqual(topics.count("ping"), 7)

    def test_request_reply(self):
        bus = Bus()
        bus.serve("math.double", lambda p: p * 2)
        self.assertEqual(bus.request("math.double", 21), 42)
        self.assertEqual([e.topic for e in bus.trace()], ["math.double", "math.double.reply"])
        with self.assertRaises(HermesError):
            bus.request("nope")
        with self.assertRaises(HermesError):
            bus.serve("math.double", lambda p: p)
        self.assertEqual(bus.stats["math.double"].count, 1)


class RouterTests(unittest.TestCase):
    def test_fallback_chain_and_usage(self):
        def boom(system, messages):
            raise BackendError("rate limited")
        r = Router({"scout": [ScriptedBackend([], ok=False), ScriptedBackend(boom), ScriptedBackend(["ok"])]})
        out = r.complete("scout", "sys", [Message("user", "hi")])
        self.assertEqual(out.text, "ok")
        self.assertEqual(r.usage["scout"].calls, 1)
        self.assertEqual(r.usage["scout"].failures, 1)
        with self.assertRaises(NoBackend) as cm:
            Router({"smith": [ScriptedBackend([], ok=False)]}).complete("smith", "", [])
        self.assertIn("unavailable", str(cm.exception))

    def test_classify(self):
        self.assertIs(classify("fix the tokenizer padding bug"), Intent.TASK)
        self.assertIs(classify("what is the blast radius of llm_adapters.py"), Intent.BLAST)
        self.assertIs(classify("give me an architecture overview of numbskull"), Intent.ARCH)
        self.assertIs(classify("how does the chaos router pick a backend?"), Intent.ASK)


if __name__ == "__main__":
    unittest.main()
