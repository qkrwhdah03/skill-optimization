"""vLLM in-process backend. ``vllm`` is imported lazily inside the engine factory."""
from __future__ import annotations

from typing import Any

from .base import BaseLLM, RawGeneration
from .registry import register_llm
from .resources import engine_key, get_or_create_engine
from .thinking import split_thinking

_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "max_tokens", "seed")


@register_llm("vllm")
class VLLMBackend(BaseLLM):
    """Runs a model with ``vllm.LLM``. Roles with the same backend/model/engine share one engine."""

    def __init__(self, config: dict, role: str = "default") -> None:
        super().__init__(config, role=role)
        self.engine_config: dict = dict(config.get("engine", {}) or {})

    # ------------------------------------------------------------------ engine
    def _engine(self) -> Any:
        def factory() -> Any:
            import vllm  # lazy heavy import

            kwargs = dict(self.engine_config)
            seed = kwargs.pop("seed", self.config.get("seed", 0))
            return vllm.LLM(model=self.model, seed=seed, **kwargs)

        return get_or_create_engine(engine_key(self.config), factory)

    def _tokenizer(self) -> Any:
        return self._engine().get_tokenizer()

    def _render(self, prompt: str, gen: dict) -> str:
        tok = self._tokenizer()
        messages = [{"role": "user", "content": prompt}]
        enable_thinking = bool(gen.get("enable_thinking", True))
        try:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": enable_thinking},
            )
        except TypeError:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
            )

    def _sampling_params(self, gen: dict) -> Any:
        from vllm import SamplingParams  # lazy heavy import

        return SamplingParams(**{k: gen[k] for k in _SAMPLING_KEYS if gen.get(k) is not None})

    # ------------------------------------------------------------------ generation
    def _generate(self, prompt: str, gen: dict) -> RawGeneration:
        return self._generate_batch([prompt], gen)[0]

    def _generate_batch(self, prompts: list[str], gen: dict) -> list[RawGeneration]:
        engine = self._engine()
        rendered = [self._render(p, gen) for p in prompts]
        outputs = engine.generate(rendered, self._sampling_params(gen), use_tqdm=False)
        results: list[RawGeneration] = []
        for text_in, out in zip(rendered, outputs):
            completion = out.outputs[0]
            reasoning, visible = split_thinking(completion.text)
            results.append(
                RawGeneration(
                    text=visible,
                    reasoning_text=reasoning,
                    prompt_text=text_in,
                    input_tokens=len(out.prompt_token_ids or []),
                    output_tokens=len(completion.token_ids or []),
                    reasoning_tokens=None,
                    raw=completion.text,
                )
            )
        return results

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer().encode(text, add_special_tokens=False))

    @classmethod
    def estimate_vram_gb(cls, config: dict, vram_budget_gb: float) -> float:
        if config.get("est_vram_gb") is not None:
            return float(config["est_vram_gb"])
        util = (config.get("engine", {}) or {}).get("gpu_memory_utilization", 0.9)
        return float(util) * float(vram_budget_gb)
