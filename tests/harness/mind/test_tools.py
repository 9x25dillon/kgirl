"""Offline tests for mind.tools: the tool-call protocol between the chat model and kgirl.

Pure parsing only: no model, no network. The cases follow what small local models actually write (see the
module docstring of tools.py): the canonical form, the Qwen `<tool_call>` form, a missing closing tag, raw
newlines inside a JSON string, and calls that are recognisably calls but cannot be read.
"""

import unittest

import tests.harness._fixtures  # noqa: F401  (sys.path)
from kgirl.harness.mind.tools import (CALL_FORMAT, STOP, Tool, ToolCall, call_complete, parse_tool_call,
                                      strip_calls, tools_doc)


class ParseToolCallTests(unittest.TestCase):
    def test_canonical_call(self):
        call = parse_tool_call('<tool>search_code {"query": "router"}</tool>')
        self.assertEqual(call, ToolCall("search_code", {"query": "router"}))
        self.assertEqual(call.prefix, "")
        self.assertEqual(call.error, "")

    def test_an_answer_is_not_a_call(self):
        self.assertIsNone(parse_tool_call("The router lives in hermes/router.py."))
        self.assertIsNone(parse_tool_call(""))
        self.assertIsNone(parse_tool_call(None))

    def test_prefix_before_the_call_is_captured(self):
        raw = 'Let me look at the layout first.\n<tool>ls {"path": "."}</tool>'
        call = parse_tool_call(raw)
        self.assertEqual(call.name, "ls")
        self.assertEqual(call.args, {"path": "."})
        self.assertEqual(call.prefix, "Let me look at the layout first.")
        self.assertEqual(call.prefix, strip_calls(raw))      # the stored prefix is exactly what was shown

    def test_missing_arguments_mean_no_arguments(self):
        self.assertEqual(parse_tool_call("<tool>repos</tool>"), ToolCall("repos", {}))
        self.assertEqual(parse_tool_call("<tool>repos {}</tool>"), ToolCall("repos", {}))

    def test_name_is_lowercased(self):
        self.assertEqual(parse_tool_call('<tool>Read_File {"path": "a.py"}</tool>').name, "read_file")

    def test_only_the_first_call_is_honored(self):
        call = parse_tool_call('<tool>ls {"path": "a"}</tool>\n<tool>ls {"path": "b"}</tool>')
        self.assertEqual(call.args, {"path": "a"})

    def test_missing_closing_tag(self):
        call = parse_tool_call('<tool>read_file {"path": "a.py", "start": 1}')
        self.assertEqual(call, ToolCall("read_file", {"path": "a.py", "start": 1}))

    def test_raw_newline_inside_a_json_string(self):
        call = parse_tool_call('<tool>build {"spec": "line one\nline two"}')
        self.assertEqual(call.error, "")
        self.assertEqual(call.args, {"spec": "line one\nline two"})

    def test_tags_are_case_and_space_tolerant(self):
        self.assertEqual(parse_tool_call('< TOOL >ls {"path": "src"}</ TOOL >'), ToolCall("ls", {"path": "src"}))

    def test_text_after_the_arguments_is_ignored(self):
        call = parse_tool_call('<tool>ls {"path": "src"} and then I will read it</tool>')
        self.assertEqual(call, ToolCall("ls", {"path": "src"}))


class QwenFormatTests(unittest.TestCase):
    def test_name_and_arguments_object(self):
        call = parse_tool_call('<tool_call>{"name": "Recall", "arguments": {"query": "staging"}}</tool_call>')
        self.assertEqual(call, ToolCall("recall", {"query": "staging"}))

    def test_tool_and_args_keys(self):
        self.assertEqual(parse_tool_call('<tool_call>{"tool": "ls", "args": {"path": "."}}</tool_call>'),
                         ToolCall("ls", {"path": "."}))

    def test_parameters_key(self):
        self.assertEqual(parse_tool_call('<tool_call>{"name": "ls", "parameters": {"depth": 2}}</tool_call>'),
                         ToolCall("ls", {"depth": 2}))

    def test_arguments_written_as_a_json_string(self):
        call = parse_tool_call('<tool_call>{"name": "ls", "arguments": "{\\"path\\": \\"src\\"}"}</tool_call>')
        self.assertEqual(call, ToolCall("ls", {"path": "src"}))

    def test_name_without_arguments(self):
        self.assertEqual(parse_tool_call('<tool_call>{"name": "repos"}</tool_call>'), ToolCall("repos", {}))

    def test_fenced_body(self):
        raw = '<tool_call>```json\n{"name": "ls", "arguments": {"path": "."}}\n```</tool_call>'
        self.assertEqual(parse_tool_call(raw), ToolCall("ls", {"path": "."}))


class UnreadableCallTests(unittest.TestCase):
    """A call that is recognisably a call but cannot be read comes back with `error` set, never as None."""

    def test_empty_call(self):
        call = parse_tool_call("thinking...\n<tool></tool>")
        self.assertEqual(call.name, "")
        self.assertEqual(call.error, "the tool call is empty")
        self.assertEqual(call.prefix, "thinking...")

    def test_malformed_json_keeps_the_name(self):
        call = parse_tool_call('<tool>grep {"pattern": </tool>')
        self.assertEqual(call.name, "grep")
        self.assertEqual(call.args, {})
        self.assertTrue(call.error.startswith("the arguments are not valid JSON"), call.error)

    def test_arguments_that_are_not_an_object(self):
        call = parse_tool_call('<tool>grep "pattern"</tool>')
        self.assertEqual(call.name, "grep")
        self.assertIn("must be a JSON object", call.error)

    def test_arguments_list_in_the_qwen_form(self):
        call = parse_tool_call('<tool_call>{"name": "ls", "arguments": [1, 2]}</tool_call>')
        self.assertEqual(call.name, "ls")
        self.assertEqual(call.error, "the arguments must be a JSON object")

    def test_object_that_names_no_tool(self):
        call = parse_tool_call('<tool_call>{"arguments": {"path": "."}}</tool_call>')
        self.assertEqual(call.name, "")
        self.assertEqual(call.error, "the tool call names no tool")

    def test_body_that_does_not_start_with_a_name(self):
        call = parse_tool_call("<tool>123 {}</tool>")
        self.assertEqual(call.name, "")
        self.assertEqual(call.error, "the tool call does not start with a tool name")

    def test_broken_object_in_the_qwen_form(self):
        call = parse_tool_call('<tool_call>{"name": "ls", </tool_call>')
        self.assertEqual(call.name, "")
        self.assertTrue(call.error.startswith("the arguments are not valid JSON"), call.error)


class ToolCallRenderTests(unittest.TestCase):
    def test_render_is_the_canonical_form(self):
        self.assertEqual(ToolCall("ls", {"path": "src"}).render(), '<tool>ls {"path": "src"}</tool>')
        self.assertEqual(ToolCall("repos").render(), "<tool>repos {}</tool>")

    def test_render_keeps_non_ascii(self):
        self.assertEqual(ToolCall("recall", {"query": "café"}).render(), '<tool>recall {"query": "café"}</tool>')

    def test_render_round_trips_through_the_parser(self):
        for call in (ToolCall("ls", {"path": "src", "depth": 2}),
                     ToolCall("write_file", {"path": "a.py", "content": "x = 1\ny = 2\n"}),
                     ToolCall("repos")):
            self.assertEqual(parse_tool_call(call.render()), call)

    def test_a_qwen_call_renders_canonically(self):
        call = parse_tool_call('<tool_call>{"name": "ls", "arguments": {"path": "."}}</tool_call>')
        self.assertEqual(call.render(), '<tool>ls {"path": "."}</tool>')


class CallCompleteTests(unittest.TestCase):
    def test_whole_call(self):
        self.assertTrue(call_complete("<tool>ls {}</tool>"))
        self.assertTrue(call_complete('text first\n<tool_call>{"name": "ls"}</tool_call>'))

    def test_open_call_is_not_complete(self):
        self.assertFalse(call_complete("<tool>ls {}"))
        self.assertFalse(call_complete("<tool"))

    def test_plain_text_and_empty(self):
        self.assertFalse(call_complete("no call here"))
        self.assertFalse(call_complete(""))
        self.assertFalse(call_complete(None))

    def test_a_closing_tag_before_the_opening_one_does_not_count(self):
        self.assertFalse(call_complete("</tool> and then <tool>ls {}"))

    def test_every_stop_sequence_closes_a_call(self):
        self.assertEqual(STOP, ["</tool>", "</tool_call>"])
        for stop in STOP:
            self.assertTrue(call_complete("<tool>ls {}" + stop), stop)


class StripCallsTests(unittest.TestCase):
    def test_text_before_the_call_stays(self):
        self.assertEqual(strip_calls("Here is what I found.\n<tool>ls {}</tool> trailing"), "Here is what I found.")

    def test_answer_without_a_call_is_only_trimmed(self):
        self.assertEqual(strip_calls("  just an answer \n"), "just an answer")

    def test_a_reply_that_is_only_a_call_is_empty(self):
        self.assertEqual(strip_calls("<tool>ls {}</tool>"), "")
        self.assertEqual(strip_calls("<tool_call>"), "")

    def test_none_is_empty(self):
        self.assertEqual(strip_calls(None), "")


class ToolsDocTests(unittest.TestCase):
    def test_lists_the_format_and_each_tool(self):
        tools = [Tool("ls", '{"path": "."}', "list a directory", lambda args: "ok"),
                 Tool("repos", "{}", "list the indexed repositories", lambda args: "ok")]
        doc = tools_doc(tools)
        lines = doc.split("\n")
        self.assertIn(CALL_FORMAT, lines)
        self.assertEqual(lines[-2:], ['- ls {"path": "."}: list a directory',
                                      "- repos {}: list the indexed repositories"])

    def test_the_documented_format_parses(self):
        self.assertEqual(parse_tool_call(CALL_FORMAT), ToolCall("name", {"arg": "value"}))


if __name__ == "__main__":
    unittest.main()
