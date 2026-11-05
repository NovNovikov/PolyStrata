"""#4923: an odd inline-code count must not swallow the tool call that follows.

A reasoning line that closed a code span with prose after its last mark left the old
counter odd, and the parser read every following tool call as quoted reasoning: the turn
ended with no call at all.  A mark that ends a line keeps the next line quoted on purpose,
and a call inside a fence or after an opener on its own line stays quoted: both hold too.
"""
import unittest

from serve import frontend

T = chr(96)   # the inline-code mark, kept out of the literals on purpose
CALL = ("<tool_call><function=write><parameter=text>hi</parameter></function></tool_call>")
TOOLS = [{"name": "write", "parameters": {"properties": {"text": {"type": "string"}}}}]


def calls_of(text):
    parser = frontend.OutputParser(thinking=False, tools=TOOLS)
    events = parser.feed(text) + parser.finish()
    return [(e.call.name, e.call.arguments) for e in events if e.call]


class InlineCodeTests(unittest.TestCase):
    def test_even_marks_with_prose_after_the_last_one_pass_the_call(self):
        text = "counting marks " + T + " like this" + T + " and then prose" + chr(10) + CALL
        self.assertEqual(calls_of(text), [("write", {"text": "hi"})])

    def test_an_odd_mark_closed_by_its_own_text_passes_the_call(self):
        text = "one lonely " + T + " then prose, the mark closed again" + chr(10) + CALL
        self.assertEqual(calls_of(text), [("write", {"text": "hi"})])

    def test_a_mark_at_the_line_end_keeps_the_next_line_quoted(self):
        text = "a code span opens at the end " + T + chr(10) + CALL
        self.assertEqual(calls_of(text), [])

    def test_a_call_inside_a_fence_stays_quoted(self):
        text = T * 3 + chr(10) + CALL + chr(10) + T * 3 + chr(10) + "done"
        self.assertEqual(calls_of(text), [])

    def test_calls_are_parsed_on_any_part_of_the_line(self):
        text = "see " + CALL + chr(10) + "and now a real one:" + chr(10) + CALL
        self.assertEqual(calls_of(text), [("write", {"text": "hi"})] * 2)


if __name__ == "__main__":
    unittest.main()
