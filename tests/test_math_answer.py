"""Math answer extraction / normalization / equivalence (loop 2026-11-16-003)."""
import pytest

from skillopt.benchmarks.math_answer import (
    MATH_ANSWER_FORMAT,
    extract_math_answer,
    last_boxed,
    math_answers_equal,
    normalize_math_answer,
)

EQUAL_PAIRS = [
    ("25", "025"),
    ("\\frac{14}{3}", "\\dfrac{14}{3}"),
    ("\\frac{14}{3}", "14/3"),
    ("0.5", "\\frac{1}{2}"),
    (".5", "0.5"),
    ("\\frac12", "\\frac{1}{2}"),
    ("\\left( 3, \\frac{\\pi}{2} \\right)", "(3,\\frac{\\pi}{2})"),
    ("90^\\circ", "90"),
    ("\\text{Evelyn}", "Evelyn"),
    ("x=5", "5"),
    ("10\\%", "10"),
    ("1,000", "1000"),
    ("11\\sqrt2", "11\\sqrt{2}"),
    ("3, 5, 7", "3,5,7"),
    ("6 - 5i", "6-5i"),
    ("\\$18.90", "18.9"),
    ("-\\frac{1}{2}", "-0.5"),
]

UNEQUAL_PAIRS = [
    ("5", "6"),
    ("\\frac{1}{3}", "0.33"),
    ("x^2", "x^3"),
    (None, "5"),
    ("5", None),
    ("", "0"),
    ("1,2", "1,2,3"),
]


# --- crit 1: answer format ------------------------------------------------------------

def test_math_answer_format_exact():
    assert MATH_ANSWER_FORMAT == "Put your final answer within \\boxed{}."


def test_math_answer_format_has_no_angle_brackets():
    assert "<" not in MATH_ANSWER_FORMAT and ">" not in MATH_ANSWER_FORMAT


# --- crit 2: last_boxed -------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    (r"a \boxed{1} b \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
    (r"\fbox{7}", "7"),
    (r"so \boxed{{a}+{b}} done", "{a}+{b}"),
    (r"x \boxed{\sqrt{\frac{1}{2}}}.", r"\sqrt{\frac{1}{2}}"),
    (r"first \boxed{3} then \boxed{ }", "3"),
    (r"first \boxed{3} then \boxed{}", "3"),
], ids=["last_wins_nested", "fbox", "nested_braces", "deep_nested", "blank_box_skipped", "empty_box_skipped"])
def test_last_boxed(text, expected):
    assert last_boxed(text) == expected


@pytest.mark.parametrize("text", [None, "", "   ", "no box here 42", r"\boxed{ }"],
                         ids=["none", "empty", "blank", "no_box", "only_blank_box"])
def test_last_boxed_none(text):
    assert last_boxed(text) is None


# --- crit 3: extract_math_answer ----------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("... so \\boxed{42}.", "42"),
    ("The final answer is $\\frac{3}{4}$.", "\\frac{3}{4}"),
    ("the ANSWER IS: 17\nmore text", "17"),
    ("The answer is 3. Then the answer is 9.", "9"),
    ("\\boxed{x<3}", "x<3"),
    ("The answer is 5 but actually \\boxed{6}", "6"),
], ids=["boxed", "answer_is_dollar", "case_insensitive_colon_first_line", "last_answer_is",
        "angle_bracket", "boxed_beats_answer_is"])
def test_extract_math_answer(text, expected):
    assert extract_math_answer(text) == expected


@pytest.mark.parametrize("text", [None, "", "   ", "I think 12 and 13"],
                         ids=["none", "empty", "blank", "no_last_number_fallback"])
def test_extract_math_answer_none(text):
    assert extract_math_answer(text) is None


# --- crit 4 / 5: equivalence ------------------------------------------------------

@pytest.mark.parametrize("a,b", EQUAL_PAIRS)
def test_math_answers_equal_true(a, b):
    assert math_answers_equal(a, b) is True
    assert math_answers_equal(b, a) is True


@pytest.mark.parametrize("a,b", UNEQUAL_PAIRS)
def test_math_answers_equal_false(a, b):
    assert math_answers_equal(a, b) is False


# --- crit 6: normalize ------------------------------------------------------------

def test_normalize_none():
    assert normalize_math_answer(None) is None


@pytest.mark.parametrize("s", sorted({x for pair in EQUAL_PAIRS for x in pair}))
def test_normalize_idempotent(s):
    n = normalize_math_answer(s)
    assert normalize_math_answer(n) == n
