"""ScriptedBackend replays its script in order: every scripted test in the harness relies on this."""
import unittest

from kgirl.harness.llm import BackendError, Message
from kgirl.harness.llm.scripted import ScriptedBackend

ASK = [Message("user", "hi")]


class ScriptedBackendTests(unittest.TestCase):
    def test_script_replays_first_in_first_out_then_raises(self):
        b = ScriptedBackend(["first", "second", "third"])
        self.assertEqual([b.complete("", ASK).text for _ in range(3)], ["first", "second", "third"])
        with self.assertRaises(BackendError):
            b.complete("", ASK)

    def test_empty_script_raises_rather_than_replying_blank(self):
        with self.assertRaises(BackendError):
            ScriptedBackend([]).complete("", ASK)

    def test_responder_and_call_log(self):
        b = ScriptedBackend(lambda system, messages: f"{system}:{messages[0].content}")
        self.assertEqual(b.complete("sys", ASK).text, "sys:hi")
        self.assertEqual(b.calls, [("sys", ASK)])

    def test_disabled_backend_is_unavailable_and_raises(self):
        self.assertTrue(ScriptedBackend(["x"]).available())
        b = ScriptedBackend(["x"], ok=False)
        self.assertFalse(b.available())
        with self.assertRaises(BackendError):
            b.complete("", ASK)
        self.assertEqual(b.calls, [])


if __name__ == "__main__":
    unittest.main()
