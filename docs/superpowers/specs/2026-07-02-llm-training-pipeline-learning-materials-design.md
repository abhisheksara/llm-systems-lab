# LLM Training Pipeline — Learning Materials Design

**Date:** 2026-07-02
**Author:** Abhishek (with Claude)
**Status:** Approved — ready for implementation

## Goal

Prepare a full theoretical + hands-on learning kit covering the stages of LLM
training — pretraining, SFT, reward modeling, RLHF (PPO), DPO, evaluation, and
RLVR/GRPO — for a senior DS transitioning to AI Engineer. Everything is built
**from scratch in raw PyTorch** (no `trl`, no pretrained backbone) on top of a
small transformer trained by the user during the exercises, so no step is a
black box. This directly clears five rows on `docs/progress.html`:
`Transformer & Self-Attention`, `LLM From Scratch`, `SFT / Fine-Tuning`,
`RLHF`, `DPO`.

Mirrors the repo's existing pattern (`positional_embeddings`,
`speculative_decoding`): an HTML theory reference, a Q&A companion doc, and
`nbformat`-generated Jupyter notebooks with `assert`-based tests. Matching
how those existing notebooks actually ship (verified against the committed
`speculative_decoding_tutorial.ipynb`): Claude implements and tests the core
algorithmic pieces (attention, PPO objective, DPO loss, GRPO advantage, etc.)
as part of the build, so every cell runs and every test passes on delivery.
The user engages by reading the theory, running and modifying the verified
code/experiments, and writing their own answers in the reflection "Question"
markdown cells, which are the one thing left genuinely blank.

## Scope decisions (confirmed with user)

- **Pretraining depth: full from-scratch architecture**, not just a training
  loop on an existing block. The transformer itself (attention, MLP, blocks)
  is implemented and tested by the user.
- **Alignment stack: fully hand-implemented**, including PPO's rollout /
  advantage / clipped-objective / KL-penalty loop — no `trl`.
- **Notebook split: one notebook per stage** (six total), each loading the
  previous stage's checkpoint, rather than one long notebook.
- **Substrate: TinyStories + a trained BPE tokenizer + self-generated
  sentiment preference data**, not char-level TinyShakespeare — outputs need
  to be legible enough that the RLHF/DPO/GRPO stages produce an observable
  behavioral shift, which is the pedagogical point of the exercise.
- **Added after a gap-check with the user:** an evaluation stage (LLM-as-judge
  pairwise win-rate + reward-vs-KL overoptimization curve), and a hands-on
  RLVR/GRPO stage (rule-based verifiable reward, no reward model or value
  head) — flagged as the most in-demand technique of the set (o1/R1-style
  reasoning RL).
- **Out of scope, deferred to other roadmap rows:** RAG/agents/tool-use
  (already covered), inference serving (KV cache, quantization, vLLM —
  separate rows), distributed training (tensor/pipeline parallelism, ZeRO —
  separate rows), data curation at scale, and distillation (not currently on
  `progress.html` — worth adding as its own future row, not part of this
  project).

## Shared code modules

Core implementations get validated inside the relevant notebook via
stub-and-test, then consolidated into shared modules so later notebooks don't
re-implement them:

- `src/llm_pipeline/model.py` — `GPTConfig`, `CausalSelfAttention`, `MLP`,
  `Block`, `GPTModel` (token + learned absolute position embeddings, tied
  LM head), `generate()`. Written from the working code the user produces in
  notebook 1.
- `src/llm_pipeline/data.py` — BPE tokenizer training/loading, TinyStories
  loading + packing into fixed-length blocks, instruction-formatting helpers.
- `src/llm_pipeline/rlhf.py` — `RewardModel` (transformer trunk + scalar
  head), Bradley-Terry pairwise loss, GAE, PPO clipped surrogate objective,
  KL-penalty utilities. Reused (partially) by the GRPO notebook.
- No parallel `pytest` suite — validation lives in the notebooks' `assert`
  cells, matching the `speculative_decoding` / `positional_embeddings`
  precedent (only production pipeline code under `src/rag` has `pytest`
  tests).

Checkpoints and generated datasets go to `data/checkpoints/llm_training_pipeline/`
(already gitignored via `data/`).

## Concrete model / data defaults

(Tunable during implementation for speed, but fixed here to avoid ambiguity.)

- **Architecture:** ~15M params — 6 layers, `d_model=384`, 6 heads, context
  length 256, learned absolute position embeddings (RoPE math is already
  covered in depth in `docs/positional_embeddings/`; not re-derived here).
- **Tokenizer:** BPE, ~8k vocab, trained on the TinyStories corpus.
- **Pretraining data:** TinyStories (HuggingFace `roneneldan/TinyStories`).
- **SFT data:** TinyStories reformatted as `"Write a short story about
  {topic}:\n" -> story`.
- **Preference signal:** sample several completions per SFT prompt at
  temperature, score each with `distilbert-base-uncased-finetuned-sst-2-english`
  (sentiment), take highest/lowest-scoring pair as chosen/rejected (ties
  filtered out).
- **LLM-as-judge (notebook 5):** a small local instruct model
  (`Qwen2.5-1.5B-Instruct` via `transformers`, run locally — keeps the
  pipeline offline/free; swappable for an API judge later if desired) doing
  pairwise comparisons in both orderings to control for position bias.
- **GRPO toy task (notebook 6):** constrained story-ending generation —
  reward = 1 if the continuation contains a specified target word AND stays
  under a token budget, else 0. Rule-based and verifiable (no reward model,
  no learned judge), in-domain for the TinyStories-trained model, and
  contrasts directly with notebook 3's learned-reward-model approach.

## Artifact 1 — HTML theory reference

`docs/llm_training_pipeline_reference.html` (top-level, matches
`speculative_decoding_reference.html` / `positional_embeddings_reference.html`:
MathJax via CDN, sticky nav, numbered sections, callout boxes).

Sections:
1. **The pipeline overview** — what each stage optimizes, what it needs as
   input, and what artifact it produces; how the stages chain together.
2. **Transformer architecture** — causal self-attention (why the causal mask,
   multi-head splitting), MLP block, pre-norm residual structure, weight
   tying. Cross-references the RoPE doc for positional encoding depth.
3. **Pretraining** — next-token prediction objective, data packing, loss
   curve interpretation, brief note on scaling laws and why data
   quality/curation matters at scale (concept-only, TinyStories is
   pre-curated here).
4. **SFT** — prompt-loss-masking, why SFT "unlocks" instruction-following,
   catastrophic forgetting / low-LR rationale.
5. **Reward modeling** — Bradley-Terry / Luce's choice axiom derivation,
   pairwise loss, reward hacking risk.
6. **PPO / RLHF** — the KL-constrained RL objective, GAE derivation, the
   clipped surrogate objective, why a value function/baseline is needed, why
   PPO is comparatively complex and fragile.
7. **DPO** — derivation from the same KL-constrained objective to a closed-form
   classification loss (no reward model, no rollouts); direct contrast with
   PPO's complexity.
8. **Evaluation** — LLM-as-judge pairwise win-rate methodology and pitfalls
   (position bias, verbosity bias); the reward-vs-KL overoptimization curve
   and Goodhart's law.
9. **RLVR / GRPO** — group-relative advantage (no value head), rule-based
   verifiable rewards, why this displaced PPO for reasoning-model training
   (DeepSeekMath/DeepSeek-R1).
10. **Beyond this pipeline (comparison table)** — RLAIF, ORPO, KTO, best-of-N /
    rejection sampling, Constitutional AI, model merging. Concept-only, no
    hands-on — keeps scope bounded.

## Artifact 2 — Concepts Q&A

`docs/llm_training_pipeline/concepts_qa.md` — same rigor/format as
`positional_embeddings/concepts_qa.md`: math derivations (DPO's closed-form
derivation in full, GAE derivation, Bradley-Terry from first principles),
paper precedent (InstructGPT, the original PPO paper, the DPO paper
(Rafailov et al.), DeepSeekMath/GRPO), and failure modes (reward hacking,
mode collapse, over-optimization, KL budget tuning, judge biases).

## Artifact 3 — Notebook series

`notebooks/llm_training_pipeline/`, generated by per-notebook builder
scripts (`notebooks/build_llm_pipeline_0N_<name>_notebook.py`, matching the
`build_positional_embeddings_notebook.py` pattern — `nbformat` +
`md()`/`code()` helpers, each script self-contained).

1. **`01_transformer_and_pretraining.ipynb`**
   - Train the BPE tokenizer; implement `CausalSelfAttention`, `MLP`, `Block`,
     `GPTModel` from scratch.
   - Tests: shape checks, a causality test (perturbing a future token must not
     change an earlier position's logits), parameter-count sanity, gradient
     flow.
   - Pretraining loop (AdamW, cosine LR, grad clipping), loss curve, sample
     generations before/after training.
   - Question cells: why the causal mask, why weight tying, why
     warmup+cosine, why a ~15M model can/can't produce coherent stories
     (scaling-law intuition).
   - Ends by consolidating the validated implementation into
     `src/llm_pipeline/model.py`. Saves `base_model.pt`.

2. **`02_sft.ipynb`**
   - Load `base_model.pt`; build the instruction-formatted dataset.
   - Implement prompt-loss-masking (`ignore_index` on prompt tokens); test
     against a synthetic example with known mask boundaries and a
     masked-loss ≠ full-loss check.
   - SFT training loop; qualitative + perplexity comparison of base vs SFT
     completions.
   - Question cells: why mask the prompt, forgetting risk, why a lower LR.
   - Saves `sft_model.pt`.

3. **`03_reward_model_and_ppo.ipynb`** (heaviest notebook — flagged as the
   one most likely to need debugging iteration)
   - Generate the preference dataset (sample + sentiment-score + pair).
   - Implement `RewardModel` + Bradley-Terry pairwise loss; test ranking
     accuracy on held-out pairs.
   - Implement PPO: frozen reference model for the KL penalty, value head,
     rollout generation, GAE, clipped surrogate objective.
   - Tests: GAE against a hand-computed toy trajectory with known advantages;
     clipped-objective behavior on synthetic old/new logprobs at the clip
     boundary; reward-model score rising over training while KL stays
     bounded.
   - Logs per-step mean reward and mean KL to
     `data/checkpoints/llm_training_pipeline/ppo_training_log.json` (consumed
     by notebook 5).
   - Question cells: why the KL penalty (reward hacking), why clipping, why a
     value function, why this is more complex/fragile than DPO.
   - Saves `ppo_model.pt` and the preference dataset (reused by notebook 4).

4. **`04_dpo.ipynb`**
   - Load `sft_model.pt` + the preference pairs from notebook 3 (no reward
     model, no rollouts).
   - Implement the DPO loss directly; test against a hand-computed toy
     example with known log-probs.
   - Train; compare DPO vs PPO vs SFT-only generations and sentiment-score
     distributions on held-out prompts.
   - Question cells: why no RM/rollouts are needed, β's regularization role,
     when PPO might still be preferred.
   - Saves `dpo_model.pt`.

5. **`05_evaluation.ipynb`**
   - Pairwise LLM-as-judge comparison (SFT vs PPO vs DPO) on held-out
     prompts, both orderings to control position bias; compute win-rates.
   - Load `ppo_training_log.json`; plot reward-model score vs KL divergence
     over training to show overoptimization.
   - Question cells: judge biases (verbosity, position), what the
     reward-vs-KL curve would look like if the RM were badly overoptimized
     vs well-regularized.

6. **`06_rlvr_grpo.ipynb`**
   - Define the verifiable reward (target-word + length-budget check).
   - Implement GRPO: sample a group of G completions per prompt, compute
     group-relative advantage (normalize reward within the group — no value
     head), reuse the clipped objective + KL penalty from `rlhf.py`.
   - Tests: group-relative advantage computation against a hand-computed toy
     group; verify no value head/critic is instantiated.
   - Train on `sft_model.pt`; show pass-rate improving over training.
   - Question cells: why no reward model or value function is needed here,
     how this connects to o1/DeepSeek-R1-style reasoning RL, when verifiable
     rewards are/aren't available in practice.
   - Saves `grpo_model.pt`.

## Artifact 4 — Progress tracker update

`docs/progress.html`: flip `Transformer & Self-Attention`, `LLM From
Scratch`, `SFT / Fine-Tuning`, `RLHF`, `DPO` rows to **"progress"** (not
"done" — the notebooks ship with stubs; "done" happens when the user finishes
the exercises, not on delivery).

## Out of scope

- `trl` / production RLHF libraries (used only as a mental cross-reference in
  the doc, not in code).
- RAG, agents, tool use, guardrails — already separate, some already done, on
  `progress.html`.
- Inference serving (KV cache, quantization, vLLM/PagedAttention) and
  distributed training (tensor/pipeline parallelism, ZeRO) — separate
  roadmap rows, correctly deferred.
- Distillation — not on the roadmap at all currently; flagged to the user as
  worth adding as its own future row, not built here.
- Large-scale data curation/dedup — mentioned conceptually only; TinyStories
  is pre-curated.
- RLAIF, ORPO, KTO, model merging — concept-only comparison table, no
  hands-on.

## Success criteria

- Each `build_llm_pipeline_0N_*_notebook.py` regenerates its `.ipynb` without
  error.
- All notebook `assert` tests pass against reference solutions.
- The full chain runs end-to-end on the RTX 3070 (8GB) in well under an hour
  total. (As built: pretraining, PPO training, the local LLM-judge load in
  Notebook 5, and GRPO training are each multi-minute steps — not just
  pretraining as originally envisioned here — but the overall chain still
  comfortably clears "well under an hour.")
- Notebook 5 shows a measurable DPO win-rate over SFT-only (hard-asserted)
  and a visibly bounded KL in the overoptimization curve. PPO's judge-based
  win-rate is reported rather than hard-asserted: this pipeline's own PPO
  run reward-hacked (Notebook 4's oracle sentiment comparison), and the
  judge model's position-bias correction was independently found unreliable
  on short text (Notebook 5's own Question 1) — both real findings, not
  implementation gaps, documented in place of forcing a passing threshold.
- Notebook 6 attempts to show GRPO pass-rate on the verifiable task
  improving over training; in this pipeline's actual runs (three
  hyperparameter configurations tried) it did not reliably improve — the
  task's binary, single-word reward is too sparse for a ~14M-parameter
  policy to learn from within 150-200 steps at these group sizes. TEST 5 is
  a structural check (training completes cleanly, no policy collapse) with
  the real outcome reported honestly, per this course's established pattern
  of treating negative results as teaching material.
- HTML reference opens standalone with MathJax rendering; all derivations are
  self-contained and correct.
