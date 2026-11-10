# skillopt — reasoning-skill optimization

`skillopt` searches for **reasoning skills** (instruction prompts, optionally with few-shot demos) that make a
**fixed** target LLM (default: Qwen3-8B) stay accurate while spending fewer tokens — thinking tokens included —
across a list of benchmarks. The research algorithm is ONE class you write (an `Optimizer`); the framework only
evaluates its prompts, hands back the results and saves everything. It never scores, ranks or selects.

```mermaid
flowchart TD
    I["optimizer.initial_prompts()"] -->|prompts| E["Evaluator<br/>each prompt × each benchmark"]
    E -->|results| U["optimizer.update(results, history)"]
    U -->|new prompts| E
    U -->|"[] or max_iterations"| S([stop])
```

## The Optimizer interface

```python
from skillopt.optimizer import Optimizer, register_optimizer

@register_optimizer("mine")
class Mine(Optimizer):                       # self.llm = optimizer LLM, self.params = config params
    def initial_prompts(self) -> list[Skill]: ...                # round 0
    def update(self, results, history) -> list[Skill]: ...       # [] = stop
```

- `results`: this round's `SkillEvaluation`s (same order as the prompts); `history`: earlier rounds, oldest first.
- Add one = one file in `skillopt/optimizer/algorithms/` (auto-discovered) + `optimizer: {name: mine, ...params}`.
- `my_optimizer.py` is the empty template (fill it in); `opro.py` is a worked example (OPRO-style).
- `skillopt/baselines.py` = ready-made skills `direct` / `cot` / `cod` an algorithm may use (`get_baseline`).

## Evaluator result format

```
SkillEvaluation            .skill (Skill), .skill_id, .split, .results
  results[benchmark]  ->   EvalResult
    n, accuracy, avg_input_tokens, avg_output_tokens (thinking included), avg_reasoning_tokens,
    avg_total_tokens, avg_latency, failures(k),
    items: list[ItemResult]  (question, prediction, gold, correct, usage, latency, response_text, reasoning_text)
```

`Evaluator.evaluate(skills, split)` returns one `SkillEvaluation` per skill; it only measures.

## Config (`configs/config.yaml`)

```yaml
benchmarks:                       # keyed by name, each measured separately; the loop evaluates split `train`
  gsm8k: {source: hf, train: 200, val: 200}
  bbh: {source: hf, tasks: all, heldout: [web_of_lies], train: 10, val: 10}   # train/val per task; test = rest
optimizer: {name: my_optimizer}   # every key except `name` goes to the class constructor
max_iterations: 5                 # max update() calls
eval: {skills: [direct, cot, cod], split: test, limit: null}   # used only with --eval-only
```

Plus `llm` (roles `target` / `optimizer` -> model presets), `hardware.vram_gb`, `evaluator.batch_size`, `run`, `seed`.
An optional `name` in an entry picks the benchmark class (default: the key), e.g. `bbh_nlu: {name: bbh, tasks: [nlu]}`.

## Benchmarks

| | GSM8K | BBH (BIG-Bench Hard) |
|---|---|---|
| Content | grade-school math word problems | 27 hard reasoning tasks (logic, symbolic, language) |
| Answer | a number | option letter, True/False, Yes/No, number, ... |
| Original data | train 7,473 · test 1,319 | test only, ~250 per task (6,511 total) |
| Links | [paper](https://arxiv.org/abs/2110.14168) · [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | [paper](https://aclanthology.org/2023.findings-acl.824/) · [lukaemon/bbh](https://huggingface.co/datasets/lukaemon/bbh) |

Splits with the default `config.yaml`:

| Split | Purpose | GSM8K | BBH |
|---|---|---|---|
| `train` | evaluated every loop round | 200 | 10 per task (260) |
| `val` | choosing the final skill | 200 | 10 per task (260) |
| `test` | reporting the final score | 1,319 (all) | rest of each task (5,741) |
| `heldout` | transfer to unseen tasks | — | web_of_lies (250) |

- GSM8K `train` / `val` are disjoint samples of the official train set. BBH has no train set, so each task is split.
- Only `train` runs in the loop; the others run with `--eval-only --split <split>`.
- BBH `tasks:` / `heldout:` take task names, `algorithmic` (15 tasks), `nlu` (12), or `all`
  (list: `BBH_TASK_GROUPS` in `skillopt/benchmarks/bbh.py`).

## Run directory

```
runs/<timestamp>_<name>/
  config.yaml, meta.json
  rounds/000/                     round 0 = initial prompts; 001, 002, ... = update() outputs
    skills/<skill_id>.json
    results/<benchmark>/<skill_id>.jsonl          per-item results
    results/<benchmark>/<skill_id>.summary.json
    summary.json                                  [SkillEvaluation.summary(), ...]
    usage.json                                    cumulative target / optimizer usage after this round
  usage/target.json, usage/optimizer.json
  run.json                                        {rounds, stop_reason, max_iterations}
```

`--eval-only` writes `runs/<timestamp>_<name>_eval/` with `skills/`, `results/<split>/<benchmark>/`, `summary.json`,
`usage/target.json`.

## Running

One config (`configs/config.yaml`), one entry point (`run.py`).

```bash
pip install -e .[dev]                       # enough for --mock
python run.py --mock --set optimizer.name=opro      # no GPU: mock presets + bundled sample data
python run.py --mock                        # fails with NotImplementedError until my_optimizer.py is filled in
pip install -e .[vllm,data]                 # real runs
python run.py                               # the loop on split train
python run.py --set max_iterations=10 --set benchmarks.gsm8k.train=100
python run.py --eval-only --skills cot runs/<run>/rounds/001/skills/opro-i1-c0.json --split test
```

`--set a.b=value` overrides any key (value parsed as YAML; list items by index, e.g. `a.0`;
benchmarks are addressed by key, e.g. `benchmarks.gsm8k.train`).

## Extending

- **Optimizer** — see above (`register_optimizer`).
- **LLM backend** — subclass `BaseLLM`, implement `_generate(prompt, gen) -> RawGeneration`, decorate with
  `@register_llm("name")`, import it in `skillopt/llm/__init__.py`, add a preset in `configs/models/`.
- **Benchmark** — subclass `Benchmark` (`load`, `extract_answer`, `score`, `answer_format`, `splits` if not
  train/val/test), decorate with `@register_benchmark("name")`, import it in `skillopt/benchmarks/__init__.py`, add an entry to `benchmarks:`.

## Models and the 24 GB VRAM budget

Presets in `configs/models/`: `qwen3-8b-vllm` (default, bf16, 0.85 x 24 = 20.4 GB), `qwen3-8b-fp8-vllm`,
`qwen3-8b-awq-vllm`, `qwen3-4b-vllm`, `qwen3-8b-hf` / `qwen3-8b-hf-4bit`, `qwen3-8b-server`, `openai-api`, `mock`.
Roles with the same model share one engine; `check_vram_budget` raises `VRAMBudgetError` before loading if the
total exceeds `hardware.vram_gb`.
