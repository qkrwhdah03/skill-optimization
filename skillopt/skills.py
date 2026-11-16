"""Skills: a reasoning skill is an instruction template plus optional few-shot demonstrations."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
QUESTION_PLACEHOLDER = "{question}"


@dataclass
class FewShotExample:
    question: str
    answer: str  # full demonstration (rationale + final answer)


@dataclass
class SkillOrigin:
    """How an optimizer derived a skill (optional; shown verbatim in the run report, never interpreted)."""

    parents: list[str]  # ids of the skills this one was derived from, primary first; [] = written from scratch
    operation: str  # free label chosen by the optimizer, e.g. "initial", "rewrite", "mutate", "crossover"
    note: str = ""  # free-text rationale, shown verbatim

    def __post_init__(self) -> None:
        if not isinstance(self.parents, (list, tuple)):
            raise TypeError(f"SkillOrigin.parents must be a list of skill ids, got {type(self.parents).__name__}")
        for pid in self.parents:
            if not isinstance(pid, str):
                raise TypeError(f"SkillOrigin.parents entries must be str, got {type(pid).__name__}")
            if not _ID_RE.match(pid):
                raise ValueError(f"invalid parent id {pid!r}: must match {_ID_RE.pattern}")
        self.parents = list(self.parents)
        if not isinstance(self.operation, str):
            raise TypeError(f"SkillOrigin.operation must be str, got {type(self.operation).__name__}")
        if not self.operation.strip():
            raise ValueError("SkillOrigin.operation must not be empty")
        if not isinstance(self.note, str):
            raise TypeError(f"SkillOrigin.note must be str, got {type(self.note).__name__}")

    def to_dict(self) -> dict:
        return {"parents": list(self.parents), "operation": self.operation, "note": self.note}

    @classmethod
    def from_dict(cls, d: dict) -> "SkillOrigin":
        return cls(parents=d["parents"], operation=d["operation"], note=d.get("note", ""))


@dataclass
class Skill:
    id: str
    instruction: str
    few_shot: list[FewShotExample] = field(default_factory=list)
    parent_id: str | None = None
    metadata: dict = field(default_factory=dict)
    origin: SkillOrigin | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _ID_RE.match(self.id):
            raise ValueError(f"invalid skill id {self.id!r}: must match {_ID_RE.pattern}")
        if self.origin is not None and not isinstance(self.origin, SkillOrigin):
            raise TypeError(f"Skill.origin must be a SkillOrigin or None, got {type(self.origin).__name__}")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "instruction": self.instruction,
            "few_shot": [{"question": fs.question, "answer": fs.answer} for fs in self.few_shot],
            "parent_id": self.parent_id,
            "metadata": dict(self.metadata),
            "origin": None if self.origin is None else self.origin.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        return cls(
            id=d["id"],
            instruction=d.get("instruction", ""),
            few_shot=[
                fs if isinstance(fs, FewShotExample) else FewShotExample(question=fs["question"], answer=fs["answer"])
                for fs in d.get("few_shot", []) or []
            ],
            parent_id=d.get("parent_id"),
            metadata=dict(d.get("metadata", {}) or {}),
            origin=_origin_from(d.get("origin")),
        )


def _origin_from(raw: object) -> SkillOrigin | None:
    if raw is None:
        return None
    if isinstance(raw, SkillOrigin):
        return raw
    return SkillOrigin.from_dict(raw)


def _skill_blocks(skill: Skill, question: str | None = None) -> list[str]:
    """Instruction (if non-empty; ``{question}`` filled in when ``question`` is given) + one block per demo."""
    blocks: list[str] = []
    instruction = (skill.instruction or "").strip()
    if instruction:
        if question is not None:
            instruction = instruction.replace(QUESTION_PLACEHOLDER, question)
        blocks.append(instruction)
    blocks.extend(f"Q: {fs.question}\nA: {fs.answer}" for fs in skill.few_shot)
    return blocks


def skill_text(skill: Skill) -> str:
    """The part of the prompt that is the skill itself: instruction (``{question}`` left as is) + few-shot demos."""
    return "\n\n".join(_skill_blocks(skill))


def render_prompt(skill: Skill, question: str, answer_format: str = "") -> str:
    """Skill blocks, then a final block: ``Q: <question>`` (unless the instruction placed it), format line, ``A:``."""
    placed = QUESTION_PLACEHOLDER in (skill.instruction or "")
    final: list[str] = [] if placed else [f"Q: {question}"]
    if answer_format:
        final.append(answer_format)
    final.append("A:")
    return "\n\n".join(_skill_blocks(skill, question) + ["\n".join(final)])
