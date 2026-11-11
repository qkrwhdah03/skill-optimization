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


# --- loop 2026-11-11-003: render_chat_prompt / thinking toggle ---------------

import logging
import uuid
from types import SimpleNamespace

import pytest

from skillopt.llm import build_llm

_THINKING_LOGGER = "skillopt.llm.thinking"


class _RecordingTok:
    """HF-like tokenizer: accepts **kwargs; renders an empty think block only when enable_thinking is False."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        self.calls.append({"messages": messages, "tokenize": tokenize,
                           "add_generation_prompt": add_generation_prompt, "kwargs": dict(kwargs)})
        body = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        if add_generation_prompt:
            body += "<|assistant|>"
            if kwargs.get("enable_thinking") is False:
                body += "<think>\n\n</think>\n\n"
        return body

    def encode(self, text, add_special_tokens=False):
        return text.split()


def _no_kwargs_tok_class():
    """Fresh, uniquely named tokenizer class whose template has no **kwargs (rejects enable_thinking)."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.calls += 1
        return "PLAIN:" + messages[0]["content"]

    def __init__(self):
        self.calls = 0

    name = f"NoKwargsTok_{uuid.uuid4().hex[:8]}"
    return type(name, (), {"__init__": __init__, "apply_chat_template": apply_chat_template})


@pytest.mark.parametrize("flag", [False, True])
def test_render_chat_prompt_passes_enable_thinking_directly(flag):
    from skillopt.llm.thinking import render_chat_prompt

    tok = _RecordingTok()
    render_chat_prompt(tok, "hi", flag)
    assert tok.calls, "apply_chat_template not called"
    for c in tok.calls:
        assert "chat_template_kwargs" not in c["kwargs"]
    last = tok.calls[-1]
    assert last["kwargs"].get("enable_thinking") is flag
    assert last["tokenize"] is False
    assert last["add_generation_prompt"] is True
    assert last["messages"] == [{"role": "user", "content": "hi"}]


def test_render_chat_prompt_flag_reaches_template():
    from skillopt.llm.thinking import render_chat_prompt

    off = render_chat_prompt(_RecordingTok(), "hi", False)
    on = render_chat_prompt(_RecordingTok(), "hi", True)
    assert "</think>" in off
    assert "</think>" not in on


def test_render_chat_prompt_typeerror_fallback_warns_once(monkeypatch, caplog):
    from skillopt.llm import thinking

    monkeypatch.setattr(thinking, "_WARNED_TOKENIZERS", set())
    caplog.set_level(logging.WARNING, logger="skillopt")
    cls = _no_kwargs_tok_class()
    tok = cls()
    out = thinking.render_chat_prompt(tok, "hi", False)
    assert out == "PLAIN:hi"
    warns = [r for r in caplog.records if r.name == _THINKING_LOGGER and r.levelno == logging.WARNING]
    assert len(warns) == 1
    out2 = thinking.render_chat_prompt(cls(), "again", True)
    assert out2 == "PLAIN:again"
    warns = [r for r in caplog.records if r.name == _THINKING_LOGGER and r.levelno == logging.WARNING]
    assert len(warns) == 1


def test_render_chat_prompt_other_exceptions_propagate():
    from skillopt.llm.thinking import render_chat_prompt

    class _BoomTok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
            raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        render_chat_prompt(_BoomTok(), "hi", False)


def _vllm(monkeypatch, tok, engine=None):
    llm = build_llm({"backend": "vllm", "model": "x"}, role="target")
    monkeypatch.setattr(llm, "_tokenizer", lambda: tok)
    if engine is not None:
        monkeypatch.setattr(llm, "_engine", lambda: engine)
        monkeypatch.setattr(llm, "_sampling_params", lambda gen: None)
    return llm


@pytest.mark.parametrize("gen,expected", [({"enable_thinking": False}, False),
                                          ({"enable_thinking": True}, True),
                                          ({}, True)])
def test_vllm_render_passes_enable_thinking_directly(monkeypatch, gen, expected):
    tok = _RecordingTok()
    llm = _vllm(monkeypatch, tok)
    out = llm._render("hi", gen)
    assert tok.calls
    for c in tok.calls:
        assert "chat_template_kwargs" not in c["kwargs"]
    assert tok.calls[-1]["kwargs"].get("enable_thinking") is expected
    assert ("</think>" in out) is (not expected)


class _FakeEngine:
    def __init__(self, completions):
        self.completions = completions
        self.calls = []

    def generate(self, prompts, params, use_tqdm=False):
        self.calls.append(list(prompts))
        return [SimpleNamespace(prompt_token_ids=[1, 2, 3], outputs=[c]) for c in self.completions[: len(prompts)]]


def _completion(text, finish_reason="__missing__"):
    c = SimpleNamespace(text=text, token_ids=[7] * max(1, len(text.split())))
    if finish_reason != "__missing__":
        c.finish_reason = finish_reason
    return c


@pytest.mark.parametrize("reason,expected", [("length", True), ("stop", False), ("__missing__", False)])
def test_vllm_generate_batch_truncated_from_finish_reason(monkeypatch, reason, expected):
    engine = _FakeEngine([_completion("The answer is 3.", reason)])
    llm = _vllm(monkeypatch, _RecordingTok(), engine)
    raws = llm._generate_batch(["q"], {"enable_thinking": False})
    assert len(raws) == 1
    assert raws[0].truncated is expected


def test_vllm_generate_batch_per_prompt_flags(monkeypatch):
    engine = _FakeEngine([_completion("a", "length"), _completion("b", "stop")])
    llm = _vllm(monkeypatch, _RecordingTok(), engine)
    raws = llm._generate_batch(["q1", "q2"], {"enable_thinking": False})
    assert [r.truncated for r in raws] == [True, False]
    resps = llm.generate_batch(["q1", "q2"], enable_thinking=False)
    assert [r.truncated for r in resps] == [True, False]
