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
class Skill:
    id: str
    instruction: str
    few_shot: list[FewShotExample] = field(default_factory=list)
    parent_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _ID_RE.match(self.id):
            raise ValueError(f"invalid skill id {self.id!r}: must match {_ID_RE.pattern}")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "instruction": self.instruction,
            "few_shot": [{"question": fs.question, "answer": fs.answer} for fs in self.few_shot],
            "parent_id": self.parent_id,
            "metadata": dict(self.metadata),
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
        )


def render_prompt(skill: Skill, question: str, answer_format: str = "") -> str:
    blocks: list[str] = []
    instruction = (skill.instruction or "").strip()
    placed = False
    if instruction:
        if QUESTION_PLACEHOLDER in instruction:
            instruction = instruction.replace(QUESTION_PLACEHOLDER, question)
            placed = True
        blocks.append(instruction)
    for fs in skill.few_shot:
        blocks.append(f"Q: {fs.question}\nA: {fs.answer}")
    final: list[str] = []
    if not placed:
        final.append(f"Q: {question}")
    if answer_format:
        final.append(answer_format)
    final.append("A:")
    blocks.append("\n".join(final))
    return "\n\n".join(blocks)
