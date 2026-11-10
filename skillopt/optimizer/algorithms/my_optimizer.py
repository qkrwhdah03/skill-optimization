"""YOUR optimizer: the empty template. Fill in the two methods; the framework does the rest.

- ``initial_prompts()`` -> the skills evaluated in round 0. Ready-made skills (direct / cot / cod):
  ``from skillopt.baselines import get_baseline``.
- ``update(results, history)`` -> the next skills to evaluate; return ``[]`` to stop.
  ``results``: this round, one ``SkillEvaluation`` per skill (``.skill``, ``.results[benchmark] -> EvalResult``
  with ``n``, ``accuracy``, ``avg_output_tokens`` (thinking included), ``avg_reasoning_tokens``,
  ``avg_total_tokens``, ``failures(k)``, ``items``). ``history``: earlier rounds, oldest first.
- Params: every key of ``optimizer:`` in configs/config.yaml except ``name`` -> ``self.params`` / ``__init__`` kwargs.
  ``self.llm`` is the optimizer LLM (``self.llm.generate(prompt)``).
- Worked example: ``opro.py`` in this folder.
"""
from __future__ import annotations

from skillopt.evaluator import SkillEvaluation
from skillopt.optimizer.base import Optimizer, register_optimizer
from skillopt.skills import Skill


@register_optimizer("my_optimizer")
class MyOptimizer(Optimizer):
    def initial_prompts(self) -> list[Skill]:
        raise NotImplementedError(
            "TODO: implement initial_prompts() in skillopt/optimizer/algorithms/my_optimizer.py "
            "(to try the example: --set optimizer.name=opro)"
        )

    def update(self, results: list[SkillEvaluation], history: list[list[SkillEvaluation]]) -> list[Skill]:
        raise NotImplementedError(
            "TODO: implement update() in skillopt/optimizer/algorithms/my_optimizer.py "
            "(to try the example: --set optimizer.name=opro)"
        )
