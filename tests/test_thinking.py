"""B. split_thinking."""
from skillopt.llm import split_thinking


def test_split_thinking_full_tags():
    assert split_thinking("<think>abc</think>Answer") == ("abc", "Answer")


def test_split_thinking_no_tags():
    assert split_thinking("  Just an answer.  ") == ("", "Just an answer.")


def test_split_thinking_unfinished():
    assert split_thinking("<think>unfinished") == ("unfinished", "")


def test_split_thinking_closing_tag_only():
    assert split_thinking("abc</think> X") == ("abc", "X")


def test_split_thinking_strips_whitespace():
    assert split_thinking("<think>\n  a b c \n</think>\n\nThe answer is 3.\n") == ("a b c", "The answer is 3.")


def test_split_thinking_uses_last_closing_tag():
    reasoning, visible = split_thinking("<think>one</think>two</think>final")
    assert visible == "final"
    assert "one" in reasoning and "two" in reasoning


def test_split_thinking_empty():
    assert split_thinking("") == ("", "")
