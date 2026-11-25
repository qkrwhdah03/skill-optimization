# Discrete Prompt Optimization Methods for Reasoning Compression

All implemented optimizers optimize discrete `Skill` objects (`instruction` + optional `few_shot` demonstrations) for a frozen reasoning LLM under the constrained length-reduction objective:

$$\min_s \; \mathbb{E}\big[T(M, x, s)\big] \qquad \text{s.t.} \qquad A(M, s) \ge A_{\text{anchor}} - \varepsilon$$

---

## 1. Collapse Analysis (CS599 Deck 14 $\rightarrow$ Single-LLM, Single-Skill Setting)

In CS599 Lecture Deck 14 (*Optimizing Agentic AI via Textual Optimization*), many agentic textual optimization algorithms assume multi-agent pipelines, gradient access to soft embeddings, or an external tool/harness skill repository. Under our single-LLM (`BaseLLM` text-only), single-prompt setting, the viable discrete methods collapse into **5 orthogonal algorithmic families**:

| Key (`optimizer.name`) | Implementation | Unifies / Adapts | What is distinct |
|---|---|---|---|
| `opro` | [`skillopt/optimizer/algorithms/opro.py`](../skillopt/optimizer/algorithms/opro.py) | **OPRO** (*Yang et al., ICLR 2024*) | Score-ordered history meta-prompting (`acc desc`, `output_tokens asc`) with top-`k` skills and failure examples. |
| `skillopt` | [`skillopt/optimizer/algorithms/skill_opt.py`](../skillopt/optimizer/algorithms/skill_opt.py) | **SkillOpt** (*Yang et al., arXiv:2605.23904v2*) | Internal disjoint `D_tr` / `D_sel` split (`75%/25%`), parallel failure + efficiency analysts, hierarchical JSON patch merging (`append`/`insert_after`/`replace`/`delete`) under cosine edit budget $L_t$, strict `D_sel` validation gate with anti-gaming guard, epoch slow update, meta-skill memory, and rejected-edit buffer. |
| `gepa` | [`skillopt/optimizer/algorithms/gepa.py`](../skillopt/optimizer/algorithms/gepa.py) | **GEPA** (*Agrawal et al., ICLR 2026 Oral*) + **EvoPrompt** (*Guo et al., ICLR 2024*) | Maintains the union Pareto frontier across global `(acc, -norm_cost)` and every per-benchmark `(acc_b, -cost_b)`, samples parents by dominance frequency, and alternates reflective single-parent mutation (`gepa_mutate`) with two-parent specialist crossover (`gepa_merge`). |
| `textgrad` | [`skillopt/optimizer/algorithms/textgrad.py`](../skillopt/optimizer/algorithms/textgrad.py) | **TextGrad** (*Nature 2025*) + **ProTeGi** (*EMNLP 2024*) + **AgentGrad Step 2** (*Chu et al.*) | Extracts per-example natural-language critiques ("textual gradients") for errors and `<think>` bloat, clusters similar critiques via token Jaccard similarity into abstracted rules (preventing per-sample overfitting), and applies beam-search `TGD.step()` updates (`textgrad_step`). |
| `mipro` | [`skillopt/optimizer/algorithms/mipro.py`](../skillopt/optimizer/algorithms/mipro.py) | **MIPROv2** (*Opsahl-Ong et al., 2024*) | Only method jointly optimizing **both `Skill.instruction` AND `Skill.few_shot`**: bootstraps shortest correct traces as compact demos (`<think>` stripped), generates grounded instructions, and selects unobserved `(instruction_idx, demo_set_idx)` pairs via empirical-Bayes UCB. |

### Why Other Course Techniques Collapse or Do Not Apply Directly

- **ProTeGi + AgentGrad Step 2 + TextGrad $\rightarrow$ `textgrad`:** In a single-node prompt pipeline, multi-node graph backpropagation reduces to direct critique generation on the single skill, while AgentGrad Step 1 (counterfactual multi-agent attribution) requires multiple agents; only gradient extraction + semantic clustering/abstraction + beam step remain distinct.
- **EvoPrompt + GEPA $\rightarrow$ `gepa`:** Multi-module System-Aware Merge reduces to two-parent clause crossover between an accuracy specialist and a token-efficiency specialist selected from the per-benchmark Pareto pool.
- **PRESTO / InstructZero / INSTINCT:** Optimize continuous embedding vectors $v$ decoded through `inputs_embeds` of a white-box model. Our `BaseLLM` / `VLLMBackend` exposes a text-in/text-out interface only.
- **RASO:** Requires retrieval over an external procedural skill corpus (`BM25` + cross-harness translation); without an external skill library, rollouts collapse to standard textual-gradient reflection.
- **Deck 7 & Deck 11 methods (GRPO, GSPO, OPSD, LoRA, CoOp/VPT/ProMetaR):** Modify model weights or continuous visual/soft tokens rather than discrete text skills for a frozen model.
- **Deck 13 inference loops (CoT-SC, Multi-Agent Debate, ReAct):** Multi-sample or multi-turn test-time loops multiply generated tokens instead of compressing single-pass reasoning.

---

## 2. Running Any Optimizer

```bash
# Quick no-GPU smoke test
python run.py --mock --set optimizer.name=gepa
python run.py --mock --set optimizer.name=textgrad
python run.py --mock --set optimizer.name=mipro

# Real Qwen3-4B run on a 24 GB GPU
python run.py \
  --set optimizer.name=gepa \
  --set llm.target.preset=qwen3-4b-fp16-vllm \
  --set llm.optimizer.preset=qwen3-4b-fp16-vllm \
  --set llm.target.generation.max_tokens=8192
```

Shared scoring, `Skill.metadata`/`SkillOrigin` tracking, and validation helpers live in [`_common.py`](../skillopt/optimizer/algorithms/_common.py) (skipped cleanly by `discover_optimizers()`).

See [`docs/skillopt.md`](skillopt.md) for the mathematical objective formulation and saved `Qwen/Qwen3-4B` benchmark results.
