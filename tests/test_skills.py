"""F. Skills & baselines."""
import pytest

from skillopt.baselines import BASELINE_NAMES, get_baseline
from skillopt.skills import FewShotExample, Skill, render_prompt


# --- AC23 ------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["a/b", "a b", "", "x\\y", "q?"])
def test_skill_id_illegal_characters(bad):
    with pytest.raises(ValueError):
        Skill(id=bad, instruction="x")


def test_skill_id_legal_characters():
    s = Skill(id="opro-1_2.v3", instruction="x")
    assert s.id == "opro-1_2.v3"


def test_skill_roundtrip():
    s = Skill(
        id="opro-1-0",
        instruction="Solve: {question}",
        few_shot=[FewShotExample(question="1+1?", answer="1+1=2. The answer is 2."),
                  FewShotExample(question="2+2?", answer="The answer is 4.")],
        parent_id="cot",
        metadata={"strategy": "opro", "iteration": 1, "nested": {"k": [1, 2]}},
    )
    d = s.to_dict()
    s2 = Skill.from_dict(d)
    assert s2 == s
    assert s2.to_dict() == d


def test_skill_roundtrip_defaults():
    s = Skill(id="plain", instruction="")
    assert Skill.from_dict(s.to_dict()) == s


# --- AC24 ------------------------------------------------------------------

def test_render_prompt_contains_all_parts():
    s = Skill(
        id="s1",
        instruction="Think carefully INSTR-MARK.",
        few_shot=[FewShotExample(question="FSQ-one", answer="FSA-one"),
                  FewShotExample(question="FSQ-two", answer="FSA-two")],
    )
    p = render_prompt(s, "MAIN-QUESTION-XYZ", "FORMAT-LINE-ABC")
    for part in ("Think carefully INSTR-MARK.", "FSQ-one", "FSA-one", "FSQ-two", "FSA-two",
                 "MAIN-QUESTION-XYZ", "FORMAT-LINE-ABC"):
        assert part in p, part
    assert p.rstrip().endswith("A:")
    assert p.index("FSQ-one") < p.index("MAIN-QUESTION-XYZ")


def test_render_prompt_question_placeholder_used_once():
    s = Skill(id="s2", instruction="Solve: {question}")
    q = "How many zebras QZX-UNIQUE?"
    p = render_prompt(s, q)
    assert p.count(q) == 1
    assert "{question}" not in p
    assert p.rstrip().endswith("A:")


def test_render_prompt_other_braces_do_not_raise():
    s = Skill(id="s3", instruction="use {x} and {y} and {0}")
    p = render_prompt(s, "QQQ-1", "fmt")
    assert "use {x} and {y} and {0}" in p
    assert "QQQ-1" in p


def test_render_prompt_empty_instruction():
    s = Skill(id="s4", instruction="")
    p = render_prompt(s, "QUESTION-EMPTY-INSTR")
    assert "QUESTION-EMPTY-INSTR" in p
    assert p.rstrip().endswith("A:")


# --- AC25 ------------------------------------------------------------------

def test_baseline_names():
    assert BASELINE_NAMES == ("direct", "cot", "cod")


@pytest.mark.parametrize("name", ["direct", "cot", "cod"])
def test_get_baseline(name):
    s = get_baseline(name)
    assert isinstance(s, Skill)
    assert s.id == name
    assert s.few_shot == []


def test_baselines_are_distinct():
    instrs = {get_baseline(n).instruction for n in BASELINE_NAMES}
    assert len(instrs) == 3


def test_get_baseline_unknown():
    with pytest.raises(KeyError):
        get_baseline("nope")
