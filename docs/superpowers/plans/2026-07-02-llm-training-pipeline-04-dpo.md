# LLM Training Pipeline — Part 4: DPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the DPO theory section, Q&A addendum, and a verified, from-scratch Direct Preference Optimization notebook that turns `sft_model.pt` into `dpo_model.pt` using the same `preference_pairs.json` Part 3 built — no reward model, no rollouts, no value function. Contrasts directly with Part 3's PPO in both notebook content and the theory reference.

**Architecture:** No new model classes. DPO trains a `copy.deepcopy(sft_model)` directly against a frozen `copy.deepcopy(sft_model)` reference, using a closed-form loss computed from sequence-level log-probabilities under both models — the same `GPTModel.forward` used everywhere else in the pipeline, with prompt-loss-masking reused from Part 2's convention (generalized to arbitrary prompt/response strings, not just SFT's topic template).

**Tech Stack:** Python 3.12, PyTorch (CUDA), `nbformat`. No new dependencies.

## Global Constraints

- Depends on Part 3 (`docs/superpowers/plans/2026-07-02-llm-training-pipeline-03-reward-model-and-ppo.md`) having been executed: `data/checkpoints/llm_training_pipeline/preference_pairs.json` and `ppo_model.pt` must exist (the latter only for Part 3 of this notebook's SFT-vs-PPO-vs-DPO comparison), the HTML `<nav>` must already include `#s5`/`#s6`, and `concepts_qa.md` must already have 14 sections. Task 3 Step 1 verifies this before proceeding.
- Every notebook code cell must run top-to-bottom without error; every `assert`-based test cell must pass. Verified per-task via `jupyter nbconvert --to notebook --execute`.
- No placeholder content: HTML/Markdown sections written in this plan are the actual final content, not outlines.
- Checkpoints and generated datasets go under `data/checkpoints/llm_training_pipeline/` (gitignored — do not commit files from that directory).
- Reflection "Question" markdown cells are left blank — do not pre-fill answers.
- Hyperparameters below (`beta`, LR, step count) are fixed defaults, not the output of a calibration run — same caveat as Part 3's Global Constraints: adjust and note deviations if a verification step fails, don't loosen assertions.
- `dpo_model.pt` checkpoint shape stays `{'model_state_dict': ..., 'config': ...}`, identical to every other stage's checkpoint.

---

## File Map

| File | Responsibility |
|------|---------------|
| `docs/llm_training_pipeline_reference.html` | Modify — add Section 7 (DPO) + nav link |
| `docs/llm_training_pipeline/concepts_qa.md` | Modify — add Q&A sections 15-16 |
| `notebooks/build_llm_pipeline_04_dpo_notebook.py` | Create — builder script for notebook 4 |
| `notebooks/llm_training_pipeline/04_dpo.ipynb` | Generated — DPO loss, training loop, PPO/DPO/SFT comparison |
| `src/llm_pipeline/data.py` | Modify — add `tokenize_prompt_response` (generalizes Part 2's `tokenize_sft_example` to arbitrary prompt/response strings) |
| `src/llm_pipeline/rlhf.py` | Modify — add `sequence_logprob`, `dpo_loss` |
| `data/checkpoints/llm_training_pipeline/dpo_model.pt` | Output checkpoint (gitignored) |

---

## Task 1: HTML Reference — Section 7 (DPO)

**Files:**
- Modify: `docs/llm_training_pipeline_reference.html`

- [ ] **Step 1: Add the nav link**

Find the `<nav>` block (after Part 3 has run, it ends with the `#s6` link):

```html
<nav>
  <a href="#s1">1. Pipeline Overview</a>
  <a href="#s2">2. Transformer Architecture</a>
  <a href="#s3">3. Pretraining</a>
  <a href="#s4">4. SFT</a>
  <a href="#s5">5. Reward Modeling</a>
  <a href="#s6">6. PPO / RLHF</a>
</nav>
```

Replace with:

```html
<nav>
  <a href="#s1">1. Pipeline Overview</a>
  <a href="#s2">2. Transformer Architecture</a>
  <a href="#s3">3. Pretraining</a>
  <a href="#s4">4. SFT</a>
  <a href="#s5">5. Reward Modeling</a>
  <a href="#s6">6. PPO / RLHF</a>
  <a href="#s7">7. DPO</a>
</nav>
```

- [ ] **Step 2: Insert Section 7**

Find the closing `</body>` tag. Immediately before it, insert:

```html
<!-- ============================================================ -->
<h2 id="s7">7. Direct Preference Optimization (DPO)</h2>

<p>PPO (Section 6) solves the KL-constrained reward-maximization objective indirectly: fit
a reward model, then run an RL loop to optimize a policy against it. DPO (Rafailov et al.
2023, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model")
observes that the <em>same</em> objective has a closed-form solution, letting it be
optimized directly from preference data with a single supervised-style loss — no reward
model, no rollouts, no value function.</p>

<h3>The Closed-Form Optimal Policy</h3>

<p>Start from exactly Section 6's objective, for a <em>fixed</em> reward function \(r\):</p>

\[\max_\pi \; \mathbb{E}_{y \sim \pi(\cdot|x)}\big[r(x, y)\big] - \beta \, \mathrm{KL}\big(\pi(\cdot|x) \,\|\, \pi_{ref}(\cdot|x)\big)\]

<div class="proof-toggle" onclick="toggleProof('proof-dpo-closed-form')">▶ Derivation of the closed-form optimum (click to expand)</div>
<div class="proof-body" id="proof-dpo-closed-form">
<p>Expand the KL term and rewrite the objective as a single expectation over \(\pi\):</p>
\[\mathbb{E}_{y\sim\pi}\Big[r(x,y) - \beta\log\frac{\pi(y|x)}{\pi_{ref}(y|x)}\Big]
= -\beta\,\mathbb{E}_{y\sim\pi}\Big[\log\frac{\pi(y|x)}{\pi_{ref}(y|x)\exp(r(x,y)/\beta)}\Big]\]
<p>Define \(Z(x) = \sum_y \pi_{ref}(y|x)\exp(r(x,y)/\beta)\) (a normalizing constant — it
does not depend on \(\pi\), only on \(x\), \(\pi_{ref}\), and \(r\)). Then
\(\pi_{ref}(y|x)\exp(r(x,y)/\beta) = Z(x)\cdot\frac{\pi_{ref}(y|x)\exp(r(x,y)/\beta)}{Z(x)}\),
and the bracketed fraction inside the log becomes
\(\pi(y|x) \big/ \big(Z(x)\cdot\pi^*(y|x)\big)\) where
\(\pi^*(y|x) := \pi_{ref}(y|x)\exp(r(x,y)/\beta)/Z(x)\) is a valid probability
distribution (it's non-negative and sums to 1 by construction). Substituting:</p>
\[-\beta\,\mathbb{E}_{y\sim\pi}\Big[\log\frac{\pi(y|x)}{\pi^*(y|x)}\Big] + \beta\log Z(x)
= -\beta\,\mathrm{KL}\big(\pi(\cdot|x) \,\|\, \pi^*(\cdot|x)\big) + \beta\log Z(x)\]
<p>\(Z(x)\) doesn't depend on \(\pi\), so maximizing this over \(\pi\) is exactly
minimizing \(\mathrm{KL}(\pi\|\pi^*)\), which is minimized (to zero) exactly when
\(\pi = \pi^*\). So the objective's unique maximizer is:</p>
\[\pi^*(y|x) = \frac{1}{Z(x)}\,\pi_{ref}(y|x)\,\exp\!\Big(\frac{r(x,y)}{\beta}\Big)\]
</div>

<h3>Solving for the Reward, and Why \(Z(x)\) Doesn't Matter</h3>

<p>Rearranging the closed-form solution for \(r\) in terms of \(\pi^*\):</p>

\[r(x, y) = \beta \log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)\]

<p>This says: <strong>any reward function's optimal policy implicitly defines that reward
function</strong>, up to the intractable per-prompt constant \(\beta\log Z(x)\) (an
implicit reward relationship, not a coincidence — hence the paper's title, "your language
model is secretly a reward model"). Substitute this expression for \(r\) directly into the
Bradley-Terry pairwise loss (Section 5), for the chosen/rejected pair \((y_w, y_l)\) sharing
the same prompt \(x\):</p>

\[r(x,y_w) - r(x,y_l) = \beta\Big[\log\frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)}\Big]\]

<p>The \(\beta\log Z(x)\) terms are identical for \(y_w\) and \(y_l\) (same prompt \(x\)) and
cancel exactly in the difference — this is <em>why</em> the intractable normalizer never
needs to be computed. Substituting into
\(\mathcal{L}_{RM} = -\log\sigma(r(x,y_w) - r(x,y_l))\) and replacing the unknown optimal
\(\pi^*\) with a directly-trained policy \(\pi_\theta\) gives the final DPO loss:</p>

\[\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\!\left(\beta\Big[\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big]\right)\right]\]

<p>No reward model is ever fit; no rollouts are ever sampled. \(\pi_\theta\) starts as a
copy of \(\pi_{ref}\) (the SFT policy) and is updated directly by gradient descent on this
loss, using only the same preference pairs a reward model would have been trained on.</p>

<h3>\(\beta\)'s Role</h3>

<p>\(\beta\) plays the identical regularization role here as in PPO's KL penalty — it comes
from the exact same objective, so this is expected, not a coincidence. Examine the
gradient of \(\mathcal{L}_{DPO}\) with respect to \(\theta\): it scales each example's
gradient contribution by \(\beta\,\sigma\big(-\beta[\Delta_w - \Delta_l]\big)\) where
\(\Delta = \log(\pi_\theta/\pi_{ref})\) — a small \(\beta\) shrinks gradient magnitude but
also makes the sigmoid's "how wrong is this pair currently" signal weaker (the model can
drift further in log-prob space before the loss saturates near zero), while a large
\(\beta\) makes the sigmoid saturate quickly, tightly holding \(\pi_\theta\) close to
\(\pi_{ref}\).</p>

<h3>When PPO Might Still Be Preferred</h3>

<p>DPO requires a fixed, pre-collected preference dataset — it cannot generate new
completions and get them scored during training the way PPO can via its live reward model.
If the reward signal is more naturally expressed as an explicit, queryable function (a
reward model that can score <em>any</em> completion an evolving policy produces, including
ones far outside the original preference dataset's distribution), PPO's online loop can
explore into and correct regions of output space the fixed DPO dataset never covered.
This matters more as training runs longer or the policy's distribution shifts further from
\(\pi_{ref}\) — DPO's preference pairs, generated once from the reference policy, become
progressively less representative of what the current policy would actually generate.</p>

<div class="keyfacts">
<strong>Key Facts — Section 7</strong>
<ul>
  <li>DPO solves the same KL-constrained objective as PPO — the derivation shows the
  optimal policy for <em>any</em> reward function has the closed form
  \(\pi^* \propto \pi_{ref}\exp(r/\beta)\).</li>
  <li>Inverting this relationship substitutes an implicit, policy-derived reward into the
  Bradley-Terry loss; the intractable partition function \(Z(x)\) cancels in the
  chosen-vs-rejected difference, which is what makes the loss tractable.</li>
  <li>DPO needs no reward model, no rollouts, and no value function — it is a single
  closed-form loss over static preference pairs.</li>
  <li>\(\beta\) plays the same regularization role as PPO's KL penalty, derived from the
  same underlying objective.</li>
  <li>PPO's online rollouts can still be preferable when the policy needs to explore beyond
  what a fixed, pre-collected preference dataset covers.</li>
</ul>
</div>
```

- [ ] **Step 3: Verify**

```bash
.venv/bin/python -c "
html = open('docs/llm_training_pipeline_reference.html', encoding='utf-8').read()
assert '<h2 id=\"s7\"' in html
assert 'href=\"#s7\"' in html
assert html.count('\\\\[') == html.count('\\\\]')
print('HTML section 7 OK,', len(html), 'bytes')
"
```

Expected: `HTML section 7 OK, <N> bytes`

- [ ] **Step 4: Commit**

```bash
git add docs/llm_training_pipeline_reference.html
git commit -m "docs: add DPO section to LLM training pipeline reference"
```

---

## Task 2: Concepts Q&A — Sections 15-16 (DPO)

**Files:**
- Modify: `docs/llm_training_pipeline/concepts_qa.md`

- [ ] **Step 1: Replace the trailing placeholder note with Sections 15-16**

Find the final line of the file (left by Part 3):

```markdown
*(Sections for DPO, evaluation, and RLVR/GRPO are appended here as the
corresponding notebooks are built.)*
```

Replace it with:

```markdown
## 15. The DPO derivation, restated as a single chain of substitutions

The full chain, compressed to its essential steps (the HTML reference has the
complete algebra):

1. **Start:** the KL-constrained RL objective,
   `max_pi E[r(x,y)] - beta*KL(pi || pi_ref)`, is the *same* objective PPO
   optimizes — DPO doesn't change what's being optimized, only how.
2. **Closed form:** for any *fixed* reward `r`, this objective's unique
   maximizer is `pi*(y|x) = pi_ref(y|x) * exp(r(x,y)/beta) / Z(x)`, where
   `Z(x)` is an intractable per-prompt normalizer. This is a general fact
   about KL-regularized objectives (it's the same form as a Boltzmann/Gibbs
   distribution reweighting a prior by an exponentiated energy — here the
   "energy" is the reward, "temperature" is `beta`, and the "prior" is
   `pi_ref`).
3. **Invert:** solve the closed form for `r` in terms of `pi*`:
   `r(x,y) = beta*log(pi*(y|x)/pi_ref(y|x)) + beta*log Z(x)`. This says any
   reward function is recoverable (up to the `Z(x)` constant) from its own
   optimal policy — the paper's "secretly a reward model" framing.
4. **Substitute into Bradley-Terry:** plug this expression for `r` into
   `-log sigmoid(r(x,y_w) - r(x,y_l))`. The `beta*log Z(x)` term is identical
   for `y_w` and `y_l` (same prompt `x`) and cancels in the subtraction —
   this is the crux of why the loss becomes computable without ever touching
   `Z(x)`.
5. **Relabel:** replace the (unknown, only-hypothetical) optimal `pi*` with a
   directly-parameterized, directly-trained `pi_theta`. The result is
   `dpo_loss` in `src/llm_pipeline/rlhf.py` — a function of only
   `log pi_theta` and `log pi_ref` evaluated on the observed preference pairs,
   no reward model or sampling required anywhere in the chain.

The practical payoff of steps 2-4: an RL objective (needing rollouts,
exploration, a value function to reduce variance) has been converted into a
supervised classification loss over a fixed dataset — as easy to optimize as
the reward model's Bradley-Terry loss itself, except it trains the *policy*
directly instead of a separate scoring function.

---

## 16. Why does `beta` do the same job in DPO as in PPO — precedent and practical tuning

Both PPO's `-beta*KL(pi||pi_ref)` penalty term and DPO's `beta` inside the
sigmoid trace back to the *same* objective (Q&A 15, step 1) — this isn't two
different hyperparameters that happen to share a name, it's the same
regularization strength appearing in two different optimization procedures
for the same underlying problem. Rafailov et al. 2023 report DPO is markedly
less sensitive to `beta` than PPO is to its equivalent KL coefficient, because
DPO's loss landscape doesn't have PPO's additional sources of instability
(value function miscalibration, advantage estimation variance, importance-
sampling ratio blowup) layered on top of the KL trade-off — `beta` is the
*only* knob controlling how far the policy is allowed to move, rather than one
of several interacting ones.

Practical tuning intuition (used to pick this pipeline's `beta=0.1`, a
commonly-cited DPO default): too small and the model can drive the sigmoid's
argument arbitrarily large for any pair, in principle allowing the policy to
memorize the specific chosen/rejected pairs in the training set rather than
learning a generalizable preference direction (an overfitting risk analogous
to reward-model overoptimization in Q&A 14, but manifesting as memorization
of the fixed preference dataset instead of exploitation of a live queryable
reward model); too large and gradient updates become too small to shift the
policy's output distribution in the number of steps this pipeline trains for.

---

*(Sections for evaluation and RLVR/GRPO are appended here as the
corresponding notebooks are built.)*
```

- [ ] **Step 2: Verify**

```bash
.venv/bin/python -c "
text = open('docs/llm_training_pipeline/concepts_qa.md', encoding='utf-8').read()
n_sections = text.count('\n## ')
assert n_sections == 16, f'expected 16 sections, found {n_sections}'
assert '## 15. The DPO derivation' in text
assert '## 16. Why does' in text
print('Q&A doc OK,', len(text), 'bytes,', n_sections, 'sections')
"
```

Expected: `Q&A doc OK, <N> bytes, 16 sections`

- [ ] **Step 3: Commit**

```bash
git add docs/llm_training_pipeline/concepts_qa.md
git commit -m "docs: add DPO concepts Q&A (closed-form derivation chain, beta precedent)"
```

---

## Task 3: Notebook Bootstrap

**Files:**
- Create: `notebooks/build_llm_pipeline_04_dpo_notebook.py`

**Interfaces:**
- Produces: a `cells` list and `md()`/`code()` helpers that Tasks 4-6 append to; a final `nbf.write(nb, OUTPUT_PATH)` call.

- [ ] **Step 1: Verify Part 3's artifacts exist**

```bash
.venv/bin/python -c "
import os
base = 'data/checkpoints/llm_training_pipeline'
for f in ['sft_model.pt', 'ppo_model.pt', 'preference_pairs.json', 'tinystories_bpe-vocab.json', 'tinystories_bpe-merges.txt']:
    p = os.path.join(base, f)
    assert os.path.exists(p), f'missing {p} — run Part 3 plan first'
print('Part 1-3 artifacts present')
"
```

Expected: `Part 1-3 artifacts present`. If this fails, stop and execute
`docs/superpowers/plans/2026-07-02-llm-training-pipeline-03-reward-model-and-ppo.md` first.

- [ ] **Step 2: Write the builder script skeleton**

Create `notebooks/build_llm_pipeline_04_dpo_notebook.py`:

```python
"""
Generates notebooks/llm_training_pipeline/04_dpo.ipynb from cell definitions.
Run: python3 notebooks/build_llm_pipeline_04_dpo_notebook.py
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.0"}
}

cells = []

def md(text): return nbf.v4.new_markdown_cell(text.strip())
def code(text): return nbf.v4.new_code_cell(text.strip())

# ─── INTRO ───────────────────────────────────────────────────────────────────
cells.append(md("""
# LLM Training Pipeline — Part 4: Direct Preference Optimization (DPO)

Stage 4 of 6. Loads `sft_model.pt` and the `preference_pairs.json` dataset Part 3 built
(the same pairs the reward model was trained on) and trains a policy directly against them
with the closed-form DPO loss — no reward model, no rollouts, no value function. Produces
`dpo_model.pt`, then compares SFT vs PPO vs DPO on held-out prompts.

**How to use this notebook:**
- Read each theory section; keep `docs/llm_training_pipeline_reference.html`
  open in another tab (Section 7) for the full derivation.
- Code and tests are already implemented and verified — run cells top to
  bottom. Answer the **Question** cells yourself.

**Parts:**
1. DPO Loss
2. DPO Training Loop
3. SFT vs PPO vs DPO Comparison
"""))

# ─── SETUP ───────────────────────────────────────────────────────────────────
cells.append(code("""
import time, math, os, json, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tokenizers import ByteLevelBPETokenizer

import sys
sys.path.insert(0, '../..')
from src.llm_pipeline.model import GPTConfig, GPTModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

CKPT_DIR = "../../data/checkpoints/llm_training_pipeline"
torch.manual_seed(0)

tokenizer = ByteLevelBPETokenizer(
    f"{CKPT_DIR}/tinystories_bpe-vocab.json",
    f"{CKPT_DIR}/tinystories_bpe-merges.txt",
)
EOT_ID = tokenizer.token_to_id('<|endoftext|>')

sft_ckpt = torch.load(f"{CKPT_DIR}/sft_model.pt", weights_only=False)
sft_cfg = sft_ckpt['config']
sft_model = GPTModel(sft_cfg).to(device)
sft_model.load_state_dict(sft_ckpt['model_state_dict'])
sft_model.eval()
BLOCK_SIZE = sft_cfg.block_size
print(f"Loaded sft_model.pt — {sum(p.numel() for p in sft_model.parameters()):,} params")

with open(f"{CKPT_DIR}/preference_pairs.json") as f:
    preference_pairs = json.load(f)
print(f"Loaded {len(preference_pairs)} preference pairs from Part 3")
"""))

# Parts 1-3 are appended here by Tasks 4-6.

# ─── WRITE ───────────────────────────────────────────────────────────────────
nb['cells'] = cells
OUTPUT_PATH = "llm_training_pipeline/04_dpo.ipynb"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    nbf.write(nb, f)
print(f"Wrote {OUTPUT_PATH} with {len(cells)} cells")
```

- [ ] **Step 3: Generate and execute**

```bash
cd notebooks && ../.venv/bin/python build_llm_pipeline_04_dpo_notebook.py && cd ..
.venv/bin/python -m nbconvert --to notebook --execute --inplace \
  notebooks/llm_training_pipeline/04_dpo.ipynb
```

Expected: `Wrote llm_training_pipeline/04_dpo.ipynb with 2 cells`, then notebook execution
completes without error, printing `Loaded sft_model.pt — 13,817,856 params` and `Loaded <N>
preference pairs from Part 3`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/build_llm_pipeline_04_dpo_notebook.py notebooks/llm_training_pipeline/04_dpo.ipynb
git commit -m "feat: bootstrap llm_training_pipeline notebook 4 (intro + setup)"
```

---

## Task 4: Notebook Part 1 — DPO Loss

**Files:**
- Modify: `notebooks/build_llm_pipeline_04_dpo_notebook.py`

**Interfaces:**
- Consumes: `tokenizer`, `EOT_ID`, `BLOCK_SIZE`, `device` from Task 3.
- Produces (notebook runtime namespace): `tokenize_prompt_response(prompt, response, tokenizer, eot_id, block_size) -> (LongTensor[block_size], LongTensor[block_size])`, `sequence_logprob(model, input_ids, labels) -> Tensor[B]`, `dpo_loss(policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta) -> scalar Tensor`.

- [ ] **Step 1: Append the Part 1 cells**

```python
# ─── PART 1: DPO LOSS ────────────────────────────────────────────────────────
cells.append(md("""
---
## Part 1: DPO Loss

`tokenize_prompt_response` generalizes Part 2's `tokenize_sft_example` to arbitrary prompt
and response strings (not just the SFT topic template) — the mask boundary rule is
identical: a target token is masked (`-100`) iff it falls inside the prompt or padding
region. `sequence_logprob` sums the log-probability of the response tokens only, giving
`log pi(y|x)` for a whole completion. `dpo_loss` implements the closed-form loss from
`docs/llm_training_pipeline_reference.html#s7` directly.
"""))

cells.append(code("""
def tokenize_prompt_response(prompt, response, tokenizer, eot_id, block_size):
    prompt_ids = tokenizer.encode(prompt).ids
    completion_ids = tokenizer.encode(response).ids + [eot_id]
    full_ids = (prompt_ids + completion_ids)[: block_size + 1]
    n_prompt = min(len(prompt_ids), len(full_ids))
    n_real = len(full_ids)

    pad_len = (block_size + 1) - n_real
    full_ids = full_ids + [eot_id] * pad_len

    input_ids = full_ids[:-1]
    targets_raw = full_ids[1:]

    labels = []
    for i in range(block_size):
        target_pos = i + 1
        if target_pos < n_prompt or target_pos >= n_real:
            labels.append(-100)
        else:
            labels.append(targets_raw[i])

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def sequence_logprob(model, input_ids, labels):
    \"\"\"Returns (B,): sum of log pi(token) over only the non-masked (response)
    positions in each sequence — log pi(y|x) for the whole completion.\"\"\"
    logits, _ = model(input_ids)
    logprobs = F.log_softmax(logits, dim=-1)
    mask = labels != -100
    safe_labels = labels.clone()
    safe_labels[~mask] = 0
    token_logprobs = logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * mask
    return token_logprobs.sum(dim=-1)


def dpo_loss(policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta=0.1):
    pi_logratios = policy_chosen_lp - policy_rejected_lp
    ref_logratios = ref_chosen_lp - ref_rejected_lp
    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()
"""))

cells.append(code("""
# TEST 1: DPO loss against a hand-computed toy example, plus a monotonicity sanity check
policy_chosen_lp = torch.tensor([-2.0])
policy_rejected_lp = torch.tensor([-3.0])
ref_chosen_lp = torch.tensor([-2.5])
ref_rejected_lp = torch.tensor([-2.5])
beta = 0.5

loss = dpo_loss(policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta)
# by hand: pi_logratios = -2.0 - (-3.0) = 1.0; ref_logratios = -2.5 - (-2.5) = 0.0
# logits = 0.5 * (1.0 - 0.0) = 0.5; loss = -log(sigmoid(0.5)), computed independently below
expected = -math.log(1.0 / (1.0 + math.exp(-0.5)))
assert abs(loss.item() - expected) < 1e-5, f"{loss.item()} != {expected}"
print(f"TEST 1a PASSED — DPO loss matches hand-computed value ({loss.item():.4f})")

# Monotonicity: loss should be lower when the policy prefers chosen over rejected
# *more strongly relative to the reference* than in a case where it prefers the opposite.
loss_good = dpo_loss(torch.tensor([-1.0]), torch.tensor([-3.0]), torch.tensor([-2.0]), torch.tensor([-2.0]), beta=0.5)
loss_bad = dpo_loss(torch.tensor([-3.0]), torch.tensor([-1.0]), torch.tensor([-2.0]), torch.tensor([-2.0]), beta=0.5)
assert loss_good.item() < loss_bad.item(), "DPO loss should be lower when the policy prefers chosen over rejected relative to the reference"
print(f"TEST 1b PASSED — loss_good ({loss_good.item():.4f}) < loss_bad ({loss_bad.item():.4f})")
"""))

cells.append(code("""
# TEST 2: tokenize_prompt_response mask boundary (same rule as SFT's tokenize_sft_example)
prompt, response = "Write a short story about dog:\\n", "A dog ran fast."
prompt_len = len(tokenizer.encode(prompt).ids)
input_ids, labels = tokenize_prompt_response(prompt, response, tokenizer, EOT_ID, BLOCK_SIZE)
assert input_ids.shape == (BLOCK_SIZE,) and labels.shape == (BLOCK_SIZE,)
assert torch.all(labels[: prompt_len - 1] == -100), "prompt-region targets not fully masked"
assert labels[prompt_len - 1].item() != -100, "first response token incorrectly masked"
print(f"TEST 2 PASSED — mask boundary correct (prompt_len={prompt_len})")
"""))

cells.append(md("""
### Question 1

`sequence_logprob` sums log-probabilities over the response tokens rather than averaging
them. Suppose `y_w` (chosen) is a much longer response than `y_l` (rejected) for the same
prompt. Could summing (rather than averaging) systematically bias which response the DPO
loss favors, independent of which one is actually better? What would change if
`sequence_logprob` divided by the number of response tokens instead?

*Write your answer below:*

"""))

# Parts 2-3 are appended here.
```

- [ ] **Step 2: Regenerate and execute**

```bash
cd notebooks && ../.venv/bin/python build_llm_pipeline_04_dpo_notebook.py && cd ..
.venv/bin/python -m nbconvert --to notebook --execute --inplace \
  notebooks/llm_training_pipeline/04_dpo.ipynb
```

Expected: no errors; output includes `TEST 1a PASSED`, `TEST 1b PASSED`, `TEST 2 PASSED`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/build_llm_pipeline_04_dpo_notebook.py notebooks/llm_training_pipeline/04_dpo.ipynb
git commit -m "feat: llm_training_pipeline notebook 4 part 1 — DPO loss"
```

---

## Task 5: Notebook Part 2 — DPO Training Loop

**Files:**
- Modify: `notebooks/build_llm_pipeline_04_dpo_notebook.py`

**Interfaces:**
- Consumes: `sft_model`, `sft_cfg`, `BLOCK_SIZE`, `tokenizer`, `EOT_ID`, `CKPT_DIR`, `device` from Task 3; `preference_pairs` from Task 3; `tokenize_prompt_response`, `sequence_logprob`, `dpo_loss` from Task 4.
- Produces: `data/checkpoints/llm_training_pipeline/dpo_model.pt`.

- [ ] **Step 1: Append the Part 2 cells**

```python
# ─── PART 2: DPO TRAINING LOOP ───────────────────────────────────────────────
cells.append(md("""
---
## Part 2: DPO Training Loop

`dpo_policy` starts as a copy of `sft_model` and is the only model updated; `ref_model` is
a frozen copy of `sft_model`, exactly the same reference used by PPO in Part 3 — same
starting point, different optimization procedure. Trains for 300 steps at `beta=0.1`.
"""))

cells.append(code("""
dpo_policy = copy.deepcopy(sft_model).to(device)
ref_model = copy.deepcopy(sft_model).to(device)
for p in ref_model.parameters():
    p.requires_grad_(False)
ref_model.eval()

held_out_pairs = preference_pairs[-30:]
train_pairs = preference_pairs[:-30]
print(f"{len(train_pairs)} training pairs, {len(held_out_pairs)} held-out pairs")

def make_dpo_batch(pairs, batch_size):
    idx = torch.randint(0, len(pairs), (batch_size,))
    chosen = [tokenize_prompt_response(pairs[i]['prompt'], pairs[i]['chosen'], tokenizer, EOT_ID, BLOCK_SIZE) for i in idx]
    rejected = [tokenize_prompt_response(pairs[i]['prompt'], pairs[i]['rejected'], tokenizer, EOT_ID, BLOCK_SIZE) for i in idx]
    c_ids = torch.stack([c[0] for c in chosen]).to(device)
    c_labels = torch.stack([c[1] for c in chosen]).to(device)
    r_ids = torch.stack([r[0] for r in rejected]).to(device)
    r_labels = torch.stack([r[1] for r in rejected]).to(device)
    return c_ids, c_labels, r_ids, r_labels
"""))

cells.append(code("""
dpo_steps = 300
dpo_lr = 5e-6
dpo_batch_size = 16
beta = 0.1

opt = torch.optim.AdamW(dpo_policy.parameters(), lr=dpo_lr)
losses = []
t0 = time.time()
for step in range(dpo_steps):
    c_ids, c_labels, r_ids, r_labels = make_dpo_batch(train_pairs, dpo_batch_size)

    policy_chosen_lp = sequence_logprob(dpo_policy, c_ids, c_labels)
    policy_rejected_lp = sequence_logprob(dpo_policy, r_ids, r_labels)
    with torch.no_grad():
        ref_chosen_lp = sequence_logprob(ref_model, c_ids, c_labels)
        ref_rejected_lp = sequence_logprob(ref_model, r_ids, r_labels)

    loss = dpo_loss(policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(dpo_policy.parameters(), 1.0)
    opt.step()
    losses.append(loss.item())
    if step % 50 == 0 or step == dpo_steps - 1:
        print(f"step {step:4d} | loss {loss.item():.3f} | elapsed {time.time()-t0:.0f}s")
print(f"DPO training elapsed: {time.time()-t0:.1f}s")
"""))

cells.append(code("""
plt.figure(figsize=(8, 4))
plt.plot(losses, alpha=0.6, label="per-step DPO loss")
window = 20
smoothed = [sum(losses[max(0,i-window):i+1]) / len(losses[max(0,i-window):i+1]) for i in range(len(losses))]
plt.plot(smoothed, label=f"{window}-step moving average", linewidth=2)
plt.xlabel("step"); plt.ylabel("DPO loss"); plt.title("DPO training loss")
plt.legend(); plt.tight_layout(); plt.show()
"""))

cells.append(code("""
# TEST 3: loss decreased, and the implicit reward margin widens on held-out pairs
first_20_avg = sum(losses[:20]) / 20
last_20_avg = sum(losses[-20:]) / 20
print(f"first-20-step avg loss: {first_20_avg:.3f}, last-20-step avg loss: {last_20_avg:.3f}")
assert last_20_avg < first_20_avg, "DPO loss did not decrease over training"

@torch.no_grad()
def dpo_reward_margin(policy, ref, pairs):
    \"\"\"Mean of (policy_chosen_lp - policy_rejected_lp) - (ref_chosen_lp - ref_rejected_lp)
    over a set of pairs — the implicit reward margin DPO is optimizing (Section 7).\"\"\"
    margins = []
    for p in pairs:
        c_ids, c_labels = tokenize_prompt_response(p['prompt'], p['chosen'], tokenizer, EOT_ID, BLOCK_SIZE)
        r_ids, r_labels = tokenize_prompt_response(p['prompt'], p['rejected'], tokenizer, EOT_ID, BLOCK_SIZE)
        c_ids, c_labels = c_ids.unsqueeze(0).to(device), c_labels.unsqueeze(0).to(device)
        r_ids, r_labels = r_ids.unsqueeze(0).to(device), r_labels.unsqueeze(0).to(device)
        pc = sequence_logprob(policy, c_ids, c_labels).item()
        pr = sequence_logprob(policy, r_ids, r_labels).item()
        rc = sequence_logprob(ref, c_ids, c_labels).item()
        rr = sequence_logprob(ref, r_ids, r_labels).item()
        margins.append((pc - pr) - (rc - rr))
    return sum(margins) / len(margins)

margin_before = dpo_reward_margin(sft_model, ref_model, held_out_pairs)
margin_after = dpo_reward_margin(dpo_policy, ref_model, held_out_pairs)
print(f"held-out implicit reward margin — before (sft_model): {margin_before:.3f}, after (dpo_policy): {margin_after:.3f}")
assert abs(margin_before) < 1e-3, "margin computed against the reference itself should be ~0 (sanity check)"
assert margin_after > margin_before, "DPO did not increase the implicit reward margin on held-out pairs"
print("TEST 3 PASSED — DPO loss decreased and held-out implicit reward margin increased")
"""))

cells.append(md("""
### Question 2

`margin_before` is computed by comparing `sft_model` against `ref_model` — but `ref_model`
*is* a copy of `sft_model`'s weights (Part 2's setup cell). Why does `TEST 3` assert this
margin is approximately zero rather than exactly zero, and why is checking it at all a
useful sanity check on `dpo_reward_margin` itself, independent of whether DPO training
worked?

*Write your answer below:*

"""))

cells.append(code("""
ckpt_path = f"{CKPT_DIR}/dpo_model.pt"
torch.save({'model_state_dict': dpo_policy.state_dict(), 'config': sft_cfg}, ckpt_path)
print(f"Saved DPO checkpoint to {ckpt_path}")
"""))

# Part 3 is appended here.
```

- [ ] **Step 2: Regenerate and execute (this step takes a few minutes)**

```bash
cd notebooks && ../.venv/bin/python build_llm_pipeline_04_dpo_notebook.py && cd ..
.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 \
  notebooks/llm_training_pipeline/04_dpo.ipynb
```

Expected: no errors; output includes `TEST 3 PASSED` and `Saved DPO checkpoint to
../../data/checkpoints/llm_training_pipeline/dpo_model.pt`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/build_llm_pipeline_04_dpo_notebook.py notebooks/llm_training_pipeline/04_dpo.ipynb
git commit -m "feat: llm_training_pipeline notebook 4 part 2 — DPO training loop"
```

---

## Task 6: Notebook Part 3 — SFT vs PPO vs DPO Comparison

**Files:**
- Modify: `notebooks/build_llm_pipeline_04_dpo_notebook.py`

**Interfaces:**
- Consumes: `sft_model`, `sft_cfg`, `tokenizer`, `CKPT_DIR`, `device` from Task 3; `dpo_policy` from Task 5.
- Produces: no new saved artifacts (analysis-only part).

- [ ] **Step 1: Append the Part 3 cells**

```python
# ─── PART 3: SFT vs PPO vs DPO COMPARISON ────────────────────────────────────
cells.append(md("""
---
## Part 3: SFT vs PPO vs DPO Comparison

Loads `ppo_model.pt` (Part 3's output) alongside `sft_model` and `dpo_policy`, and compares
qualitative completions plus sentiment-score distributions across all three on held-out
topics — the same sentiment scorer used to build the preference dataset, so scores are
directly comparable to Part 3's.
"""))

cells.append(code("""
from transformers import pipeline as hf_pipeline
from src.llm_pipeline.data import TOPIC_KEYWORDS, format_sft_prompt

sentiment_pipe = hf_pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if device == 'cuda' else -1,
)

def sentiment_score(text):
    result = sentiment_pipe(text[:512])[0]
    sign = 1.0 if result['label'] == 'POSITIVE' else -1.0
    return sign * result['score']

ppo_ckpt = torch.load(f"{CKPT_DIR}/ppo_model.pt", weights_only=False)
ppo_model = GPTModel(ppo_ckpt['config']).to(device)
ppo_model.load_state_dict(ppo_ckpt['model_state_dict'])
ppo_model.eval()
dpo_policy.eval()
print("Loaded ppo_model.pt for comparison")
"""))

cells.append(code("""
held_out_topics = TOPIC_KEYWORDS[-10:]

@torch.no_grad()
def generate_completion(model, prompt, max_new_tokens=40):
    prompt_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    out = model.generate(prompt_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
    return tokenizer.decode(out[0, prompt_ids.shape[1]:].tolist())

sft_scores, ppo_scores, dpo_scores = [], [], []
for topic in held_out_topics:
    prompt = format_sft_prompt(topic)
    sft_c = generate_completion(sft_model, prompt)
    ppo_c = generate_completion(ppo_model, prompt)
    dpo_c = generate_completion(dpo_policy, prompt)
    sft_scores.append(sentiment_score(sft_c))
    ppo_scores.append(sentiment_score(ppo_c))
    dpo_scores.append(sentiment_score(dpo_c))
    print(f"=== topic: {topic} ===")
    print("SFT:", sft_c)
    print("PPO:", ppo_c)
    print("DPO:", dpo_c)
    print()

print(f"mean sentiment — SFT: {sum(sft_scores)/len(sft_scores):+.3f}, "
      f"PPO: {sum(ppo_scores)/len(ppo_scores):+.3f}, "
      f"DPO: {sum(dpo_scores)/len(dpo_scores):+.3f}")
"""))

cells.append(code("""
plt.figure(figsize=(7, 4))
plt.boxplot([sft_scores, ppo_scores, dpo_scores], labels=["SFT", "PPO", "DPO"])
plt.ylabel("sentiment score")
plt.title("Held-out completion sentiment — SFT vs PPO vs DPO")
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
plt.tight_layout(); plt.show()
"""))

cells.append(code("""
# TEST 4: DPO must raise mean sentiment above the SFT-only baseline. PPO's comparison is
# reported, not hard-asserted: PPO optimizes a *learned* reward model (Section 5) rather
# than the oracle sentiment scorer directly, so it can legitimately reward-hack into
# text the reward model still scores well but that a held-out oracle judges as no
# better (or worse) than SFT — this is a real, observable instance of the exact failure
# mode Section 5 and Question 3 (below) ask you to reason about, not a bug to hide.
mean_sft = sum(sft_scores) / len(sft_scores)
mean_ppo = sum(ppo_scores) / len(ppo_scores)
mean_dpo = sum(dpo_scores) / len(dpo_scores)
print(f"mean sentiment — SFT: {mean_sft:+.3f}, PPO: {mean_ppo:+.3f}, DPO: {mean_dpo:+.3f}")
assert mean_dpo > mean_sft, "DPO did not raise mean sentiment above SFT on held-out topics"
if mean_ppo > mean_sft:
    print("PPO also raised mean sentiment above SFT-only on held-out topics.")
else:
    print("PPO did NOT raise mean sentiment above SFT-only here — read the PPO completions "
          "above and see Question 3: this is a live instance of reward hacking (Section 5), "
          "not a failed run.")
print("TEST 4 PASSED — DPO raises mean held-out sentiment above SFT-only")
"""))

cells.append(md("""
### Question 3

PPO and DPO start from the identical `sft_model` checkpoint and target the identical
underlying objective, but reach it through very different training procedures (Section 6
vs Section 7). Looking at the sentiment distributions and the generations themselves, do
PPO and DPO converge to similarly-shifted output distributions, or do they diverge in some
noticeable way? In particular, if PPO's mean held-out sentiment did *not* exceed SFT's:
read the actual PPO completions printed above — do they read as coherent stories, or do
you see repetition/broken grammar that a sentiment classifier still scores positively
(because it only reads word-level valence, not coherence)? Given everything covered in
Sections 5-7 (reward hacking, KL budgets, DPO's static-dataset limitation), what's your
best guess for *why* PPO and DPO might diverge like this?

*Write your answer below:*

"""))
```

- [ ] **Step 2: Regenerate and execute**

```bash
cd notebooks && ../.venv/bin/python build_llm_pipeline_04_dpo_notebook.py && cd ..
.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 \
  notebooks/llm_training_pipeline/04_dpo.ipynb
```

Expected: no errors; output includes `TEST 4 PASSED`. PPO's held-out sentiment may or may
not exceed SFT's — both outcomes are informative and neither blocks this step; only DPO's
comparison is a hard assertion (see the code above for why).

- [ ] **Step 3: Commit**

```bash
git add notebooks/build_llm_pipeline_04_dpo_notebook.py notebooks/llm_training_pipeline/04_dpo.ipynb
git commit -m "feat: llm_training_pipeline notebook 4 part 3 — SFT vs PPO vs DPO comparison"
```

---

## Task 7: Consolidate into `src/llm_pipeline/data.py` and `src/llm_pipeline/rlhf.py`

**Files:**
- Modify: `src/llm_pipeline/data.py`
- Modify: `src/llm_pipeline/rlhf.py`

**Interfaces:**
- Consumes: the validated function bodies from Task 4 (copied verbatim).
- Produces: `from src.llm_pipeline.data import tokenize_prompt_response`; `from src.llm_pipeline.rlhf import sequence_logprob, dpo_loss`.

- [ ] **Step 1: Append `tokenize_prompt_response` to `src/llm_pipeline/data.py`**

Add to the end of `src/llm_pipeline/data.py`:

```python


def tokenize_prompt_response(prompt: str, response: str, tokenizer, eot_id: int, block_size: int):
    """Generalizes tokenize_sft_example to arbitrary prompt/response strings (not
    just the SFT topic template). Same prompt-loss-masking convention: a target
    token is masked (-100) iff it falls inside the prompt or padding region."""
    prompt_ids = tokenizer.encode(prompt).ids
    completion_ids = tokenizer.encode(response).ids + [eot_id]
    full_ids = (prompt_ids + completion_ids)[: block_size + 1]
    n_prompt = min(len(prompt_ids), len(full_ids))
    n_real = len(full_ids)

    pad_len = (block_size + 1) - n_real
    full_ids = full_ids + [eot_id] * pad_len

    input_ids = full_ids[:-1]
    targets_raw = full_ids[1:]

    labels = []
    for i in range(block_size):
        target_pos = i + 1
        if target_pos < n_prompt or target_pos >= n_real:
            labels.append(-100)
        else:
            labels.append(targets_raw[i])

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )
```

- [ ] **Step 2: Append `sequence_logprob` and `dpo_loss` to `src/llm_pipeline/rlhf.py`**

Add to the end of `src/llm_pipeline/rlhf.py`:

```python


def sequence_logprob(model, input_ids, labels):
    """Returns (B,): sum of log pi(token) over only the non-masked (response)
    positions in each sequence — log pi(y|x) for the whole completion."""
    logits, _ = model(input_ids)
    logprobs = F.log_softmax(logits, dim=-1)
    mask = labels != -100
    safe_labels = labels.clone()
    safe_labels[~mask] = 0
    token_logprobs = logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * mask
    return token_logprobs.sum(dim=-1)


def dpo_loss(policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta=0.1):
    pi_logratios = policy_chosen_lp - policy_rejected_lp
    ref_logratios = ref_chosen_lp - ref_rejected_lp
    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()
```

- [ ] **Step 3: Smoke-test the additions**

```bash
.venv/bin/python -c "
import torch
from src.llm_pipeline.model import GPTConfig, GPTModel
from src.llm_pipeline.data import tokenize_prompt_response
from src.llm_pipeline.rlhf import sequence_logprob, dpo_loss
from tokenizers import ByteLevelBPETokenizer

cfg = GPTConfig(vocab_size=8000, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
model = GPTModel(cfg)

class DummyTok:
    def encode(self, s):
        class R:
            ids = [ord(c) % 100 for c in s]
        return R()

tok = DummyTok()
ids, labels = tokenize_prompt_response('hi there', 'a response', tok, eot_id=99, block_size=32)
assert ids.shape == (32,) and labels.shape == (32,)

lp = sequence_logprob(model, ids.unsqueeze(0), labels.unsqueeze(0))
assert lp.shape == (1,)

loss = dpo_loss(torch.tensor([-1.0]), torch.tensor([-2.0]), torch.tensor([-1.5]), torch.tensor([-1.5]), beta=0.1)
assert loss.item() > 0
print('src.llm_pipeline.data / rlhf DPO helpers smoke test OK')
"
```

Expected: `src.llm_pipeline.data / rlhf DPO helpers smoke test OK` with no assertion errors.

- [ ] **Step 4: Commit**

```bash
git add src/llm_pipeline/data.py src/llm_pipeline/rlhf.py
git commit -m "refactor: consolidate notebook 4's DPO utilities into src/llm_pipeline"
```

---

## Self-Review Checklist

**1. Spec coverage** (against `docs/superpowers/specs/2026-07-02-llm-training-pipeline-learning-materials-design.md`):
- HTML reference Section 7 (DPO — full derivation from the KL-constrained objective to the closed-form loss, direct contrast with PPO's complexity): Task 1. ✓
- Concepts Q&A additions for DPO: Task 2. ✓
- Notebook 4 (`04_dpo.ipynb`): load `sft_model.pt` + Part 3's preference pairs, implement the DPO loss with a hand-computed toy-example test, train, compare DPO vs PPO vs SFT-only generations and sentiment-score distributions on held-out prompts, saves `dpo_model.pt`: Tasks 3-6. ✓
- Consolidation into `src/llm_pipeline/data.py` and `src/llm_pipeline/rlhf.py`: Task 7. ✓

**2. Placeholder scan:** No "TBD"/"TODO"/"similar to Task N" patterns; every code step contains complete, runnable code; every HTML/Markdown step contains complete final content.

**3. Type/interface consistency:** `tokenize_prompt_response(prompt, response, tokenizer, eot_id, block_size) -> (LongTensor, LongTensor)` has an identical body in the notebook (Task 4) and `src/llm_pipeline/data.py` (Task 7) — it deliberately duplicates Part 2's `tokenize_sft_example` masking logic under a more general signature rather than modifying that function, since `tokenize_sft_example`'s narrower `(topic, story)` signature is still used by Part 2's own tests and is left untouched. `sequence_logprob(model, input_ids, labels) -> Tensor[B]` and `dpo_loss(...) -> scalar Tensor` are identical between the notebook (Task 4) and `src/llm_pipeline/rlhf.py` (Task 7). `dpo_model.pt`'s checkpoint shape matches every prior stage's.
