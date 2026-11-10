"""HuggingFace transformers backend. torch/transformers are imported lazily in the engine factory."""
from __future__ import annotations

from typing import Any

from .base import BaseLLM, RawGeneration
from .registry import register_llm
from .resources import engine_key, get_or_create_engine
from .thinking import split_thinking


@register_llm("hf")
class HFTransformersLLM(BaseLLM):
    """Plain ``model.generate`` backend; useful for 4-bit loading or when vLLM is unavailable."""

    def __init__(self, config: dict, role: str = "default") -> None:
        super().__init__(config, role=role)
        self.engine_config: dict = dict(config.get("engine", {}) or {})

    def _engine(self) -> tuple[Any, Any]:
        def factory() -> tuple[Any, Any]:
            import torch  # noqa: F401  lazy heavy import
            from transformers import AutoModelForCausalLM, AutoTokenizer

            eng = self.engine_config
            kwargs: dict[str, Any] = {"device_map": eng.get("device_map", "cuda")}
            if eng.get("load_in_4bit"):
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=getattr(torch, eng.get("dtype") or "bfloat16")
                )
            else:
                kwargs["torch_dtype"] = eng.get("dtype") or "bfloat16"
            tokenizer = AutoTokenizer.from_pretrained(self.config.get("tokenizer", self.model))
            model = AutoModelForCausalLM.from_pretrained(self.model, **kwargs)
            model.eval()
            return model, tokenizer

        return get_or_create_engine(engine_key(self.config), factory)

    def _generate(self, prompt: str, gen: dict) -> RawGeneration:
        import torch  # lazy heavy import

        model, tok = self._engine()
        enable_thinking = bool(gen.get("enable_thinking", True))
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = tok(rendered, return_tensors="pt", add_special_tokens=False).to(model.device)
        if gen.get("seed") is not None:
            torch.manual_seed(int(gen["seed"]))
        temperature = float(gen.get("temperature", 0.0) or 0.0)
        gen_kwargs: dict[str, Any] = {"max_new_tokens": int(gen.get("max_tokens", 1024))}
        if temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature)
            if gen.get("top_p") is not None:
                gen_kwargs["top_p"] = gen["top_p"]
            if gen.get("top_k") is not None:
                gen_kwargs["top_k"] = gen["top_k"]
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        n_in = inputs["input_ids"].shape[1]
        new_ids = out[0][n_in:]
        text = tok.decode(new_ids, skip_special_tokens=True)
        reasoning, visible = split_thinking(text)
        return RawGeneration(
            text=visible,
            reasoning_text=reasoning,
            prompt_text=rendered,
            input_tokens=int(n_in),
            output_tokens=int(new_ids.shape[0]),
            raw=text,
        )

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        _, tok = self._engine()
        return len(tok.encode(text, add_special_tokens=False))
