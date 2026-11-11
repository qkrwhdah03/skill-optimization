"""E. OpenAI-compatible backend (no network)."""
import json

import pytest

from helpers import CONFIGS
from skillopt.llm import OpenAICompatLLM, build_llm

KEY_ENV = "SKILLOPT_TEST_API_KEY"
SECRET = "sk-secret-VALUE-9f8e7d6c"

FULL_RESPONSE = {
    "choices": [{"message": {"content": "Answer 5", "reasoning_content": "think think"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 30,
              "completion_tokens_details": {"reasoning_tokens": 20}},
}


def _cfg(enable_thinking=False):
    return {
        "backend": "openai",
        "model": "remote-model",
        "engine": {"base_url": "http://127.0.0.1:9/v1", "api_key_env": KEY_ENV},
        "generation": {"max_tokens": 64, "temperature": 0.0, "enable_thinking": enable_thinking},
    }


@pytest.fixture
def llm(monkeypatch):
    monkeypatch.setenv(KEY_ENV, SECRET)
    inst = build_llm(_cfg(), role="target")
    assert isinstance(inst, OpenAICompatLLM)
    payloads = []

    def fake_post(payload):
        payloads.append(payload)
        return inst._next_response

    inst._next_response = FULL_RESPONSE
    monkeypatch.setattr(inst, "_post", fake_post)
    inst._payloads = payloads
    return inst


def test_openai_parses_content_reasoning_and_usage(llm):
    resp = llm.generate("What is 2+3?")
    assert resp.text == "Answer 5"
    assert resp.reasoning_text == "think think"
    assert resp.usage.to_dict() == {"input_tokens": 10, "output_tokens": 30,
                                    "reasoning_tokens": 20, "total_tokens": 40}


def test_openai_payload_contents(llm):
    prompt = "What is 2+3?"
    llm.generate(prompt)
    assert len(llm._payloads) == 1
    p = llm._payloads[0]
    assert p["model"] == "remote-model"
    assert p["messages"][0] == {"role": "user", "content": prompt}
    assert p["max_tokens"] == 64
    assert p["temperature"] == 0.0
    assert p["chat_template_kwargs"] == {"enable_thinking": False}


def test_openai_gen_kwargs_override_in_payload(llm):
    llm.generate("hi", max_tokens=7)
    assert llm._payloads[-1]["max_tokens"] == 7


def test_openai_think_tags_in_content_without_reasoning_content(llm, monkeypatch):
    # usage lacks completion_tokens_details -> reasoning is counted locally; avoid transformers dependency
    monkeypatch.setattr(llm, "count_tokens", lambda text: len(text.split()))
    llm._next_response = {
        "choices": [{"message": {"content": "<think>r</think>v"}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 9},
    }
    resp = llm.generate("q")
    assert resp.reasoning_text == "r"
    assert resp.text == "v"


def test_openai_missing_usage_falls_back_to_count_tokens(llm, monkeypatch):
    monkeypatch.setattr(llm, "count_tokens", lambda text: len(text.split()))
    llm._next_response = {"choices": [{"message": {"content": "<think>r</think>v"}}]}
    resp = llm.generate("hello world now")
    u = resp.usage
    assert u.input_tokens == 3
    assert u.reasoning_tokens == 1
    assert u.output_tokens == 2
    assert u.total_tokens == 5


def test_api_key_not_in_config_or_payload(llm):
    llm.generate("hello")
    assert SECRET not in json.dumps(llm.config, default=str)
    assert SECRET not in repr(llm.config)
    assert SECRET not in json.dumps(llm._payloads[0], default=str)


def test_api_key_not_in_run_dir_config(monkeypatch, tmp_path):
    from skillopt.cli import main

    monkeypatch.setenv(KEY_ENV, SECRET)

    def fake_post(self, payload):
        return FULL_RESPONSE

    monkeypatch.setattr(OpenAICompatLLM, "_post", fake_post)
    run = main([
        "--config", str(CONFIGS / "config.yaml"),
        "--mock", "--eval-only",
        "--set", f"run.root={tmp_path.as_posix()}",
        "--set", "llm.target.backend=openai",
        "--set", "llm.target.model=remote-model",
        "--set", "llm.target.engine.base_url=http://127.0.0.1:9/v1",
        "--set", f"llm.target.engine.api_key_env={KEY_ENV}",
        "--limit", "1",
    ])
    text = (run / "config.yaml").read_text(encoding="utf-8")
    assert SECRET not in text
    for f in run.rglob("*"):
        if f.is_file():
            assert SECRET not in f.read_text(encoding="utf-8", errors="ignore"), f


# --- loop 2026-11-11-003: truncation from finish_reason ------------------------

def _resp_with_finish(reason):
    choice = {"message": {"content": "Answer 5", "reasoning_content": "think think"}}
    if reason is not None:
        choice["finish_reason"] = reason
    return {"choices": [choice], "usage": FULL_RESPONSE["usage"]}


@pytest.mark.parametrize("reason,expected", [("length", True), ("stop", False), (None, False)])
def test_openai_truncated_from_finish_reason(llm, reason, expected):
    llm._next_response = _resp_with_finish(reason)
    assert llm.generate("q").truncated is expected


def test_openai_payload_still_uses_chat_template_kwargs(llm):
    llm.generate("q")
    assert llm._payloads[-1]["chat_template_kwargs"] == {"enable_thinking": False}
