"""A. Token accounting & usage tracking (skillopt.llm)."""
import pytest

from helpers import make_mock, mock_config, uniq
from skillopt.llm import (
    BaseLLM,
    LLMResponse,
    RawGeneration,
    TokenUsage,
    UsageSnapshot,
    build_llm,
    build_role_llms,
    register_llm,
)


class _RecordingLLM(BaseLLM):
    """Test backend: whitespace token counting, records gen dicts, returns `self.next_raw` or echoes."""

    uses_gpu = False

    def __init__(self, config, role="default"):
        super().__init__(config, role=role)
        self.gens = []
        self.next_raw = None
        self.count_calls = []

    def _generate(self, prompt, gen):
        self.gens.append(dict(gen))
        if self.next_raw is not None:
            return self.next_raw
        return RawGeneration(text=f"echo {prompt}")

    def count_tokens(self, text):
        self.count_calls.append(text)
        return len(text.split())


_REC_NAME = uniq("rec")
register_llm(_REC_NAME)(_RecordingLLM)


def _rec(**cfg):
    config = {"backend": _REC_NAME, "model": "rec-model"}
    config.update(cfg)
    return build_llm(config, role="target")


# --- AC1 -------------------------------------------------------------------

def test_token_usage_total_defaults_to_input_plus_output():
    u = TokenUsage(input_tokens=3, output_tokens=5, reasoning_tokens=2)
    assert u.total_tokens == 8


def test_token_usage_add_sums_every_field():
    a = TokenUsage(input_tokens=3, output_tokens=5, reasoning_tokens=2)
    b = TokenUsage(input_tokens=10, output_tokens=20, reasoning_tokens=7)
    c = a + b
    assert (c.input_tokens, c.output_tokens, c.reasoning_tokens, c.total_tokens) == (13, 25, 9, 38)


def test_token_usage_to_dict_keys():
    d = TokenUsage(input_tokens=3, output_tokens=5, reasoning_tokens=2).to_dict()
    assert set(d) == {"input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"}
    assert d == {"input_tokens": 3, "output_tokens": 5, "reasoning_tokens": 2, "total_tokens": 8}


# --- AC2 -------------------------------------------------------------------

def test_counts_with_count_tokens_when_backend_reports_none():
    llm = _rec()
    llm.next_raw = RawGeneration(text="x y", reasoning_text="r1 r2 r3 r4")
    resp = llm.generate("a b c")
    assert isinstance(resp, LLMResponse)
    u = resp.usage
    assert u.input_tokens == 3
    assert u.reasoning_tokens == 4
    assert u.output_tokens == 2 + 4
    assert u.total_tokens == 3 + 6


def test_prompt_text_used_for_input_count_when_given():
    llm = _rec()
    llm.next_raw = RawGeneration(text="x", prompt_text="p q r s t")
    resp = llm.generate("a b")
    assert resp.usage.input_tokens == 5


def test_empty_reasoning_counts_zero():
    llm = _rec()
    llm.next_raw = RawGeneration(text="one two three")
    resp = llm.generate("a")
    assert resp.usage.reasoning_tokens == 0
    assert resp.usage.output_tokens == 3
    assert resp.usage.total_tokens == 4


# --- AC3 -------------------------------------------------------------------

def test_reported_counts_used_verbatim():
    llm = _rec()
    llm.next_raw = RawGeneration(text="x", reasoning_text="r", input_tokens=100,
                                 output_tokens=50, reasoning_tokens=20)
    resp = llm.generate("a b c")
    u = resp.usage
    assert (u.input_tokens, u.output_tokens, u.reasoning_tokens, u.total_tokens) == (100, 50, 20, 150)


def test_reported_reasoning_clamped_to_reported_output():
    llm = _rec()
    llm.next_raw = RawGeneration(text="x", input_tokens=1, output_tokens=5, reasoning_tokens=20)
    u = llm.generate("a").usage
    assert u.output_tokens == 5
    assert u.reasoning_tokens == 5
    assert u.total_tokens == 6


def test_counted_reasoning_clamped_to_reported_output():
    llm = _rec()
    llm.next_raw = RawGeneration(text="x", reasoning_text="a b c d", output_tokens=2)
    u = llm.generate("a").usage
    assert u.output_tokens == 2
    assert u.reasoning_tokens == 2


# --- AC4 -------------------------------------------------------------------

def test_mock_output_tokens_include_thinking():
    mock = make_mock(["<think>a b c</think>The answer is 3."])
    resp = mock.generate("What is 1+2?")
    assert resp.reasoning_text == "a b c"
    assert resp.text == "The answer is 3."
    assert resp.usage.reasoning_tokens == 3
    assert resp.usage.output_tokens == 3 + mock.count_tokens("The answer is 3.")
    assert resp.usage.input_tokens == mock.count_tokens("What is 1+2?")


def test_mock_count_tokens_rule():
    mock = make_mock(["x"])
    assert mock.count_tokens("The answer is 3.") == 5
    assert mock.count_tokens("") == 0


def test_mock_records_calls_and_cycles_responses():
    mock = make_mock(["A1", "A2"])
    texts = [mock.generate(p).text for p in ("p1", "p2", "p3")]
    assert texts == ["A1", "A2", "A1"]
    assert mock.calls == ["p1", "p2", "p3"]


def test_mock_default_is_deterministic_and_respects_enable_thinking():
    m1 = make_mock(seed=0)
    m2 = make_mock(seed=0)
    r1 = m1.generate("Question one?")
    r2 = m2.generate("Question one?")
    assert r1.text == r2.text
    assert r1.reasoning_text == r2.reasoning_text
    assert r1.text.startswith("The answer is")
    assert r1.usage.reasoning_tokens > 0

    m3 = make_mock(seed=0, generation={"enable_thinking": False})
    r3 = m3.generate("Question one?")
    assert r3.reasoning_text == ""
    assert r3.usage.reasoning_tokens == 0
    assert "<think>" not in r3.text


# --- AC5 -------------------------------------------------------------------

def test_usage_accumulates_across_calls():
    llm = _rec()
    llm.next_raw = RawGeneration(text="x", input_tokens=2, output_tokens=7, reasoning_tokens=3)
    llm.generate("a")
    llm.generate("b")
    s = llm.usage.snapshot()
    assert isinstance(s, UsageSnapshot)
    assert s.calls == 2
    assert s.input_tokens == 4
    assert s.output_tokens == 14
    assert s.reasoning_tokens == 6
    assert s.total_tokens == 18
    assert s.latency >= 0.0


def test_snapshot_is_independent_copy():
    llm = _rec()
    llm.generate("a")
    snap = llm.usage.snapshot()
    before = snap.to_dict()
    llm.generate("b c d")
    assert snap.to_dict() == before
    assert snap.calls == 1
    assert llm.usage.snapshot().calls == 2


def test_usage_reset_zeroes_everything():
    llm = _rec()
    llm.generate("a b")
    llm.usage.reset()
    s = llm.usage.snapshot()
    assert s.to_dict() == {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                           "reasoning_tokens": 0, "total_tokens": 0, "latency": 0.0}


def test_usage_snapshot_to_dict_keys():
    assert set(UsageSnapshot().to_dict()) == {"calls", "input_tokens", "output_tokens",
                                              "reasoning_tokens", "total_tokens", "latency"}


# --- AC6 -------------------------------------------------------------------

def test_track_counts_only_calls_inside_block():
    llm = _rec()
    llm.generate("before")
    with llm.track() as t:
        llm.generate("inside one")
        llm.generate("inside two")
    llm.generate("after")
    assert t.snapshot().calls == 2
    assert llm.usage.snapshot().calls == 4


def test_nested_track_both_receive_inner_calls():
    llm = _rec()
    with llm.track() as outer:
        llm.generate("o1")
        with llm.track() as inner:
            llm.generate("i1")
            llm.generate("i2")
        llm.generate("o2")
    assert inner.snapshot().calls == 2
    assert outer.snapshot().calls == 4
    assert llm.usage.snapshot().calls == 4


def test_track_detaches_when_block_raises():
    llm = _rec()
    with pytest.raises(RuntimeError):
        with llm.track() as t:
            llm.generate("x")
            raise RuntimeError("boom")
    llm.generate("after")
    assert t.snapshot().calls == 1
    assert llm.usage.snapshot().calls == 2


def test_track_values_persist_after_exit():
    llm = _rec()
    llm.next_raw = RawGeneration(text="x", input_tokens=4, output_tokens=6, reasoning_tokens=1)
    with llm.track() as t:
        llm.generate("x")
    s = t.snapshot()
    assert (s.calls, s.input_tokens, s.output_tokens, s.reasoning_tokens, s.total_tokens) == (1, 4, 6, 1, 10)


# --- AC7 -------------------------------------------------------------------

def test_generate_batch_returns_one_response_per_prompt_in_order():
    llm = _rec()
    prompts = ["p0", "p1", "p2", "p3"]
    out = llm.generate_batch(prompts)
    assert [r.text for r in out] == [f"echo {p}" for p in prompts]
    assert all(isinstance(r, LLMResponse) for r in out)
    assert all(r.latency >= 0 for r in out)
    assert llm.usage.snapshot().calls == 4


def test_generate_batch_empty():
    llm = _rec()
    assert llm.generate_batch([]) == []
    assert llm.usage.snapshot().calls == 0


def test_mock_generate_batch_records_one_call_per_prompt():
    mock = make_mock(["<think>a</think>The answer is 1."])
    out = mock.generate_batch(["q1", "q2", "q3"])
    assert len(out) == 3
    assert mock.usage.snapshot().calls == 3
    assert all(r.latency >= 0 for r in out)


# --- AC8 -------------------------------------------------------------------

def test_gen_kwargs_override_config_defaults():
    llm = _rec(generation={"max_tokens": 10, "temperature": 0.5})
    llm.generate("p", max_tokens=3)
    assert llm.gens[-1]["max_tokens"] == 3
    assert llm.gens[-1]["temperature"] == 0.5
    llm.generate("p")
    assert llm.gens[-1]["max_tokens"] == 10
    llm.generate_batch(["a", "b"], temperature=0.0)
    assert llm.gens[-1]["temperature"] == 0.0
    assert llm.gens[-1]["max_tokens"] == 10
    assert llm.gen_defaults == {"max_tokens": 10, "temperature": 0.5}


# --- AC9 -------------------------------------------------------------------

def test_build_role_llms_distinct_objects_independent_trackers():
    cfg = mock_config(["<think>a</think>The answer is 1."])
    llms = build_role_llms({"target": dict(cfg), "optimizer": dict(cfg)})
    target, optimizer = llms["target"], llms["optimizer"]
    assert target is not optimizer
    assert target.usage is not optimizer.usage
    optimizer.generate("hello")
    optimizer.generate("again")
    assert target.usage.snapshot().calls == 0
    assert optimizer.usage.snapshot().calls == 2
    assert target.role == "target"
    assert optimizer.role == "optimizer"
