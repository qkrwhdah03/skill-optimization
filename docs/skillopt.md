# SkillOpt: Constrained Reasoning-Length Adaptation

Implementation: [`skillopt/optimizer/algorithms/skill_opt.py`](../skillopt/optimizer/algorithms/skill_opt.py) (`@register_optimizer("skillopt")`, class `SkillOptOptimizer`)  
Tests: [`tests/test_skill_opt.py`](../tests/test_skill_opt.py)

---

## 1. Objective Function Reformulation

The original *SkillOpt: Executive Strategy for Self-Evolving Agent Skills* (Yang et al., arXiv:2605.23904v2) optimizes an agent skill $s$ to **maximize task accuracy** $\max_s A(M, s)$ for a fixed model $M$. In this project, we keep $M$ fixed and optimize reasoning skills to **minimize generated token cost without degrading accuracy** beyond a tolerance $\varepsilon$ relative to an anchor baseline (default `cot`):

$$\min_s \; \mathbb{E}\big[T(M, x, s)\big] \qquad \text{s.t.} \qquad A(M, s) \ge A_{\text{anchor}} - \varepsilon$$

To operationalize this constrained objective across heterogeneous benchmarks (grade-school math, formal logic, long-context puzzles, competition math, and cross-domain MCQ) inside `SkillOptOptimizer`:

1. **Macro-normalized token cost (`norm_cost`):**
   Each benchmark $b \in B$ has very different baseline response lengths (e.g., ~587 tokens on StrategyQA vs. ~5,353 tokens on LSAT-AR). Rather than letting long-context benchmarks dominate raw token sums, `_score_eval` normalizes each benchmark's cost against the anchor baseline and averages equally across active benchmarks:

   $$\text{norm\_cost}(s) = \frac{1}{|B|} \sum_{b \in B} \frac{C_b(s)}{\max\big(C_b(\text{anchor}),\, 1.0\big)}$$

   where $C_b(\cdot)$ is selected by `cost_metric`:
   - `"output"` *(default)*: `usage.output_tokens` (includes both `<think>` reasoning tokens and visible final answer tokens),
   - `"reasoning"`: `usage.reasoning_tokens` only,
   - `"total"`: `usage.total_tokens` (prompt + completion).

2. **Feasibility constraint (`feasible`):**
   Evaluated strictly on the held-out internal selection slice $D_{\text{sel}}$:

   $$\Delta A(s) = \bar{A}(s) - \bar{A}(\text{anchor}) \ge -\varepsilon \qquad (\text{default } \varepsilon = 0.02)$$

   with an optional per-benchmark guardrail (`per_benchmark_epsilon`) requiring $\min_{b \in B} \big(A_b(s) - A_b(\text{anchor})\big) \ge -\varepsilon_{\text{bench}}$.

3. **Scalar utility and ranking (`feasible_utility`):**

   $$\text{feasible\_utility}(s) = \big(1 - \text{norm\_cost}(s)\big) + \text{acc\_bonus} \cdot \Delta A(s)$$

   - When both candidate and incumbent are feasible, `_check_gate_improvement` maximizes `feasible_utility` (with `acc_bonus=0.0` by default, this strictly minimizes normalized token cost; setting `acc_bonus > 0` trades extra accuracy headroom against token savings).
   - When the incumbent is infeasible, the gate prioritizes feasibility and higher $\bar{A}(s)$ first, using cost savings only as a tie-breaker (`abs(acc_delta) <= 1e-9`).

4. **Anti-gaming guard for thinking models (`require_reasoning_reduction=True`):**
   On reasoning models like `Qwen3` (`enable_thinking=true`), over 90% of output tokens are inside `<think>...</think>`. If `cost_metric == "output"` and a candidate cuts `norm_cost` solely by trimming the short visible answer while `norm_reasoning` does not decrease (and accuracy did not improve), `_check_gate_improvement` rejects it as `"visible-only shortening (reasoning tokens did not decrease)"`.

5. **Strict improvement gating:**
   Equal-score ties are rejected (`min_improvement = 1e-6`), and an incumbent is only replaced when a candidate strictly improves the constrained objective on $D_{\text{sel}}$.

---

## 2. How SkillOpt Components Were Adapted

Because `SkillOptOptimizer` plugs into `skillopt.optimizer.Optimizer` (`initial_prompts()` and stateless `update(results, history)`), Algorithm 1 and Appendix A of SkillOpt are adapted as follows:

| SkillOpt Concept | Original Purpose (Accuracy) | Adaptation in `skill_opt.py` (Accuracy-Constrained Compression) |
|---|---|---|
| **Validation split** | Separate feedback split from gating | `_partition_example_ids` deterministically splits the framework's `train` items per benchmark (seeded by `split_seed=42`) into disjoint **`D_tr`** (`75%`, for reflection/analysts) and **`D_sel`** (`25%`, `selection_ratio=0.25`, exclusively for validation gating and ranking). |
| **Dual-stream reflection** | Diagnose task errors | Stratified rollout sampling (`rollout_batch_size=40`, `reflection_batch_size=8`) splits items into two parallel analyst streams: **`analyst_failure`** (fixes wrong answers, logical slips, formatting bugs, `max_tokens` truncations) and **`analyst_efficiency`** (compares bloated `<think>` traces vs. compact successes to eliminate redundant reasoning steps). |
| **Hierarchical patch merging & ranking** | Merge per-batch edits cleanly | Two-stage merge: **`merge_failure`** + **`merge_efficiency`** -> **`merge_final`** (prioritizing failure fixes when near the accuracy floor, efficiency edits when accuracy headroom exists), followed by `_rank_edits` and multi-subset candidate generation (`num_candidates=2`). |
| **Atomic edits + budget schedule** | Prevent unconstrained prompt drift | Supports **`append`**, **`insert_after`**, **`replace`**, and **`delete`** (`patch` mode, with bounded `rewrite` fallback up to `max_skill_chars=2400`). Budget $L_t$ anneals from `max_edits=4` down to `min_edits=2` over `edit_schedule_rounds=4` (`cosine`, `linear`, or `constant`). |
| **Validation gate & rejected buffer $B$** | Monotonic progress without regressions | `_replay_gate` deterministically replays `history` on $D_{\text{sel}}$ every round to reconstruct incumbent `v`, anchor `a`, instruction `hash_cache` ($C$), and epoch-local rejected edits ($B$, `max_rejected_buffer=8`) with compact rejection reasons shown in negative memory. |
| **Epoch slow update vs. meta-skill** | Consolidate durable strategy periodically | At `epoch_rounds=2` boundaries, longitudinal deltas across $D_{\text{tr}}$ trigger two separated mechanisms: (1) an optimizer-only **`self._meta_skill`** (`<meta>`) guiding future reflection/merges (never shown to the target LLM), and (2) a candidate **`slow` update** inside `<!-- SLOW_UPDATE_START --> ... <!-- SLOW_UPDATE_END -->` (`skillopt_slow`) that competes through the $D_{\text{sel}}$ gate like any candidate. |

---

## 3. Running SkillOpt

Example command used for the `Qwen/Qwen3-4B` run across all 9 trainable benchmarks in [`configs/config.yaml`](../configs/config.yaml):

```bash
python run.py \
  --set optimizer.name=skillopt \
  --set run.name=skillopt_4b_all9 \
  --set llm.target.preset=qwen3-4b-fp16-vllm \
  --set llm.optimizer.preset=qwen3-4b-fp16-vllm \
  --set llm.target.generation.max_tokens=8192
```

> **24 GB GPU note:** Setting identical `backend`, `model`, and `engine` fields on both `llm.target` and `llm.optimizer` ensures both roles share a single vLLM engine within the 24 GB VRAM budget (`check_vram_budget`). If overriding `llm.target.engine.*`, apply the same engine override to `llm.optimizer.engine.*`.

---

## 4. Saved Results (`Qwen/Qwen3-4B`, FP16, `max_tokens=8192`)

### 4.1 Optimization on `train` (9 Benchmarks, `n = 738` items/skill, 6 Rounds)

Saved summaries & top evolved skills: [`results/skillopt_4b_all9/`](../results/skillopt_4b_all9/)  
Run configuration: `Qwen/Qwen3-4B` (`float16`), `max_tokens=8192` (`enable_thinking=true` on target, `false` on optimizer), `max_iterations=5`, `seed=0`.  
Total usage: `9,594` target calls (`11,787,738` output tokens, `10,907,861` reasoning tokens, `67,653.1s`) and `41` optimizer calls (`72,123` total tokens, `146.4s`).

| Round | Skill | Macro Acc | Macro Out Tok | Micro Acc | Micro Out Tok | Micro Think Tok | Truncated | vs. `cot` Micro Tok |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 000 | `cot` *(anchor)* | 66.85% | 2,635.6 | 79.13% | 1,860.0 | 1,691.5 | 5.0% (37) | baseline |
| 000 | `direct` | **70.94%** | 1,852.4 | 79.95% | 1,249.1 | 1,190.4 | 2.6% (19) | -32.8% |
| 000 | `cod` | 66.63% | 1,575.9 | 76.15% | 1,019.1 | 953.7 | 2.2% (16) | -45.2% |
| 001 | `skillopt-i1-c0` | 65.77% | 1,573.1 | 77.51% | 1,085.3 | 992.6 | **1.8% (13)** | -41.7% |
| 001 | `skillopt-i1-c1` | 70.50% | 2,129.0 | 81.03% | 1,482.2 | 1,365.8 | 2.8% (21) | -20.3% |
| 002 | **`skillopt-i2-c0`** | 69.81% | 2,281.9 | **81.57%** | 1,560.0 | 1,457.9 | 3.9% (29) | -16.1% |
| 002 | `skillopt-i2-c1` | 67.50% | 1,477.4 | 77.51% | **962.9** | 904.9 | 2.3% (17) | -48.2% |
| 003 | `skillopt-i3-c0` | 70.37% | 1,891.0 | 80.08% | 1,278.4 | 1,179.2 | 3.1% (23) | -31.3% |
| 003 | `skillopt-i3-c1` | 69.32% | 1,987.5 | 79.81% | 1,369.7 | 1,270.4 | 2.8% (21) | -26.4% |
| 004 | **`skillopt-i4-c0`** | 69.37% | 1,602.9 | 80.22% | 1,055.4 | 941.4 | 2.2% (16) | -43.3% |
| 004 | `skillopt-i4-c1` | 66.42% | 1,551.9 | 78.18% | 1,035.6 | 946.2 | 2.0% (15) | -44.3% |
| 005 | `skillopt-i5-c0` | 67.33% | 1,601.4 | 78.59% | 1,048.9 | 985.6 | 3.0% (22) | -43.6% |
| 005 | **`skillopt-i5-c1`** | 70.52% | **1,389.1** | 78.73% | 966.0 | **900.8** | 2.0% (15) | **-48.1%** |

**Evolved skills selected for unseen `test` evaluation:**
- [`skillopt-i5-c1.json`](../results/skillopt_4b_all9/skills/skillopt-i5-c1.json): `Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most.\nEnsure critical context is retained in concise steps.`
- [`skillopt-i4-c0.json`](../results/skillopt_4b_all9/skills/skillopt-i4-c0.json): `Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most.\nLabel steps with [STEP] for clarity`
- [`skillopt-i2-c0.json`](../results/skillopt_4b_all9/skills/skillopt-i2-c0.json): `Think directly, simplify expressions, use key values.\nFocus on core logic, avoid extra details`

### 4.2 Unseen `test`-Split Evaluation (All 12 Benchmarks, `--limit 50`)

Saved results: [`results/skillopt_4b_test_all12/`](../results/skillopt_4b_test_all12/) *(pending completion of live run `skillopt_4b_test_all12`)*.
