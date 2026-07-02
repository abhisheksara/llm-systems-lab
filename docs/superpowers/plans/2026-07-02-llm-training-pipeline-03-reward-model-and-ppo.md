# LLM Training Pipeline — Part 3: Reward Model & PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reward-modeling + PPO theory sections, Q&A additions, and a verified, from-scratch reward model + PPO notebook that turns `sft_model.pt` into `ppo_model.pt` — a policy shifted toward higher-sentiment completions via KL-constrained RL against a learned reward model. This is the heaviest notebook in the series (reward model + full PPO loop: rollout, GAE, clipped surrogate, KL penalty).

**Architecture:** `RewardModel` (new class in this plan) reuses the `Block`/`GPTConfig` building blocks from `src/llm_pipeline/model.py` but swaps the tied LM head for a scalar `reward_head`, with its trunk weights initialized from `sft_model.pt` (standard practice — the reward model starts from the same distribution it will score). `PPOActorCritic` (new class) wraps a full `GPTModel` (reusing its `tok_emb`/`pos_emb`/`blocks`/`ln_f`/`lm_head` submodules directly, unmodified) and adds a `value_head`, without touching `model.py`.

**Tech Stack:** Python 3.12, PyTorch (CUDA), HuggingFace `transformers` (sentiment-analysis pipeline, already in `requirements.txt`), `nbformat`. No new dependencies.

## Global Constraints

- Depends on Part 2 (`docs/superpowers/plans/2026-07-02-llm-training-pipeline-02-sft.md`) having been executed: `data/checkpoints/llm_training_pipeline/sft_model.pt` must exist, `docs/llm_training_pipeline_reference.html`'s `<nav>` must already include the `#s4` link, and `docs/llm_training_pipeline/concepts_qa.md` must already have 10 sections. Task 3 Step 1 verifies this before proceeding.
- Every notebook code cell must run top-to-bottom without error; every `assert`-based test cell must pass. Verified per-task via `jupyter nbconvert --to notebook --execute`.
- No placeholder content: HTML/Markdown sections written in this plan are the actual final content, not outlines.
- Checkpoints and generated datasets go under `data/checkpoints/llm_training_pipeline/` (gitignored — do not commit files from that directory).
- Reflection "Question" markdown cells are left blank (for the user to fill in by hand) — do not pre-fill answers.
- Hyperparameters below (batch sizes, step counts, LR, `beta`, `clip_eps`) are fixed defaults chosen for a small model on a single consumer GPU, not the output of a calibration run (unlike Part 1/2, this plan was written without executing the notebook). If any `assert`-based test fails during execution, the implementer should adjust the failing hyperparameter and note the deviation in the commit message — do not silently loosen an assertion's threshold to make a bad run pass.
- Checkpoint dict shape stays `{'model_state_dict': ..., 'config': ...}`, identical to `base_model.pt`/`sft_model.pt`, loadable via `GPTModel(ckpt['config'])` + `load_state_dict(...)`.

---

## File Map

| File | Responsibility |
|------|---------------|
| `docs/llm_training_pipeline_reference.html` | Modify — add Section 5 (Reward Modeling) + Section 6 (PPO/RLHF) + nav links |
| `docs/llm_training_pipeline/concepts_qa.md` | Modify — add Q&A sections 11-14 |
| `notebooks/build_llm_pipeline_03_reward_model_and_ppo_notebook.py` | Create — builder script for notebook 3 |
| `notebooks/llm_training_pipeline/03_reward_model_and_ppo.ipynb` | Generated — preference dataset, reward model, PPO |
| `src/llm_pipeline/rlhf.py` | Create — `RewardModel`, `load_trunk_from_sft`, `bradley_terry_loss`, `PPOActorCritic`, `generate_rollout`, `compute_token_rewards`, `compute_gae`, `ppo_clipped_loss`, `evaluate_actions` |
| `data/checkpoints/llm_training_pipeline/preference_pairs.json` | Output dataset (gitignored) — reused by Part 4 (DPO) |
| `data/checkpoints/llm_training_pipeline/ppo_training_log.json` | Output log (gitignored) — reused by Part 5 (evaluation) |
| `data/checkpoints/llm_training_pipeline/ppo_model.pt` | Output checkpoint (gitignored) |

---

## Task 1: HTML Reference — Section 5 (Reward Modeling) + Section 6 (PPO/RLHF)

**Files:**
- Modify: `docs/llm_training_pipeline_reference.html`

- [ ] **Step 1: Add the nav links**

Find the `<nav>` block (after Part 2 has run, it ends with the `#s4` link):

```html
<nav>
  <a href="#s1">1. Pipeline Overview</a>
  <a href="#s2">2. Transformer Architecture</a>
  <a href="#s3">3. Pretraining</a>
  <a href="#s4">4. SFT</a>
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
</nav>
```

- [ ] **Step 2: Insert Sections 5 and 6**

Find the closing `</body>` tag. Immediately before it, insert:

```html
<!-- ============================================================ -->
<h2 id="s5">5. Reward Modeling</h2>

<p>PPO needs a scalar reward function \(r(x, y)\) — given a prompt \(x\) and a completion
\(y\), how good is it? No such function exists directly; humans (or, here, a sentiment
classifier standing in for a cheap, automatable preference signal) can only reliably say
<em>which of two completions is better</em>, not assign an absolute numeric quality score.
The reward model's job is to convert relative preference judgments into an absolute scalar
function that a downstream RL loop can optimize.</p>

<h3>From Preferences to a Scalar: the Bradley-Terry Model</h3>

<p>The Bradley-Terry model (Bradley &amp; Terry, 1952) assumes each item \(i\) has a latent
"strength" \(s_i\), and the probability that \(i\) beats \(j\) in a pairwise comparison is:</p>

\[P(i \succ j) = \frac{s_i}{s_i + s_j}\]

<p>Reparameterizing \(s_i = e^{r_i}\) (strengths are always positive, so this is a natural
substitution) gives the logistic form:</p>

\[P(i \succ j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \frac{1}{1 + e^{-(r_i - r_j)}} = \sigma(r_i - r_j)\]

<p>This is exactly <strong>Luce's choice axiom</strong> (Luce, 1959) applied to pairwise
choice: the probability of choosing \(i\) over \(j\) depends only on their two underlying
values, via a logistic link. Applied to (prompt, completion) pairs — treating the reward
model's output \(r_\theta(x, y)\) as the latent strength — the probability that a human (or
our sentiment-based proxy) prefers the chosen completion \(y_w\) over the rejected one
\(y_l\), for the same prompt \(x\), is modeled as:</p>

\[P(y_w \succ y_l \mid x) = \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big)\]

<h3>The Pairwise Loss</h3>

<p>Training \(r_\theta\) by maximum likelihood on observed preference pairs gives the
standard reward-model loss (this is exactly the loss used in InstructGPT, Ouyang et al.
2022):</p>

\[\mathcal{L}_{RM}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big)\Big]\]

<div class="definition">
<strong>Why only relative loss, never an absolute target?</strong> The loss only ever
depends on the <em>difference</em> \(r_\theta(x,y_w) - r_\theta(x,y_l)\) — it is invariant
to adding any constant to both rewards. This means the reward model's output scale and
zero-point are arbitrary; only reward <em>differences</em> (and downstream, the KL-weighted
comparison to the policy in the PPO objective) are meaningful. This is a direct consequence
of Bradley-Terry only ever modeling pairwise comparisons.</div>

<h3>Architecture</h3>

<p>The reward model reuses the same transformer trunk as the policy (token + position
embeddings, the same causal `Block` stack), initialized from the SFT checkpoint's weights
(a warm start from a model that already produces on-distribution completions, rather than
training a reward model from scratch), but replaces the tied LM head with a single linear
`reward_head` applied to the final token's hidden state — one scalar per (prompt,
completion) sequence, not one prediction per token.</p>

<h3>Reward Hacking</h3>

<div class="definition">
<strong>Reward hacking</strong> is what happens when the policy, during PPO, finds outputs
that score highly under \(r_\theta\) without actually being higher-quality by whatever
standard the reward model was meant to approximate. Because \(r_\theta\) is trained on a
finite, biased sample of comparisons — here, comparisons derived from a sentiment
classifier's output on model-generated text, not genuine human judgments of story quality —
it necessarily has blind spots and systematic biases (e.g. it may reward superficially
"positive-sounding" vocabulary independent of whether the text is a coherent story at all).
An unconstrained RL loop will find and exploit any such blind spot, because that is exactly
what "maximize reward" means. This is precisely why PPO's objective (Section 6) includes a
KL penalty back to the SFT policy — it bounds how far the policy is allowed to drift in
search of reward, limiting how hard it can exploit the reward model's imperfections.</div>

<div class="keyfacts">
<strong>Key Facts — Section 5</strong>
<ul>
  <li>The Bradley-Terry pairwise loss is derived from Luce's choice axiom: preference
  probability is a logistic function of the reward difference.</li>
  <li>The loss is shift-invariant in the reward model's output — only relative reward
  matters, never an absolute scale.</li>
  <li>The reward model's trunk is initialized from the SFT checkpoint, not trained from
  scratch, and only the final-token scalar head is new.</li>
  <li>Reward hacking is a structural risk whenever the reward model is a proxy for the true
  objective — it motivates the KL constraint in PPO.</li>
</ul>
</div>

<!-- ============================================================ -->
<h2 id="s6">6. PPO / RLHF</h2>

<h3>The KL-Constrained RL Objective</h3>

<p>The full RLHF objective (InstructGPT, Ouyang et al. 2022; building on Schulman et al.
2017's PPO) is not simply "maximize reward" — an unconstrained policy would collapse onto
whatever narrow, reward-hacking output maximizes \(r_\theta\), regardless of fluency or
diversity. Instead:</p>

\[\max_\pi \; \mathbb{E}_{x \sim D,\, y \sim \pi(\cdot|x)}\big[r_\theta(x, y)\big] - \beta \, \mathbb{E}_{x}\big[\mathrm{KL}(\pi(\cdot|x) \,\|\, \pi_{ref}(\cdot|x))\big]\]

<p>where \(\pi_{ref}\) is the frozen SFT policy. The KL term is a soft constraint, not a
hard one: it lets the policy move toward higher reward, but pays an increasing penalty the
further it drifts from a policy that is known to already produce fluent, on-distribution
text. \(\beta\) trades off reward-seeking against staying close to \(\pi_{ref}\) — too small
and the policy reward-hacks (Section 5); too large and training barely moves the policy at
all.</p>

<p>In practice, the KL term is applied <strong>per-token</strong> as a reward penalty rather
than as a separate loss term: at each generated token, the model pays
\(-\beta \big(\log \pi(a_t|s_t) - \log \pi_{ref}(a_t|s_t)\big)\), and the reward model's
score is added only once, at the final token of the completion (since \(r_\theta\) scores a
whole completion, not individual tokens). This gives a dense, per-step reward signal instead
of one sparse reward at the very end of a long sequence.</p>

<h3>Generalized Advantage Estimation (GAE)</h3>

<p>PPO is a policy-gradient method: it needs an estimate of the <strong>advantage</strong>
\(A_t\) — how much better an action was than the policy's average behavior at that state —
to know which actions to reinforce. A value function \(V(s_t)\) (here, a linear head added
on top of the same transformer trunk as the policy — see "Why a Value Function") predicts
the expected future (KL-penalized + reward-model) return from state \(s_t\), acting as a
baseline that reduces gradient variance versus using raw returns directly.</p>

<p>GAE (Schulman et al. 2016, "High-Dimensional Continuous Control Using Generalized
Advantage Estimation") combines the one-step TD residual
\(\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\) across multiple horizons with an
exponentially-decaying weight \(\lambda\):</p>

\[A_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l} = \delta_t + \gamma\lambda \, A_{t+1}^{GAE(\gamma,\lambda)}\]

<p>The recursive form on the right is what's implemented: compute \(\delta_t\) at every
step, then sweep backward through the trajectory accumulating
\(A_t = \delta_t + \gamma\lambda A_{t+1}\) (with \(A_T = 0\) at the trajectory's end,
since there is no future beyond the last generated token). \(\lambda=1\) recovers the
full Monte-Carlo advantage (\(A_t = \sum_{l\ge0}\gamma^l r_{t+l} - V(s_t)\), high variance,
low bias); \(\lambda=0\) recovers the one-step TD advantage (\(A_t = \delta_t\), low
variance, high bias). Intermediate \(\lambda\) (0.95 here) interpolates.</p>

<h3>The Clipped Surrogate Objective</h3>

<p>Vanilla policy gradient (\(\mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \, A]\)) is only
valid for an infinitesimally small policy update — after any real gradient step, the
distribution used to <em>collect</em> the rollout (the "old" policy \(\pi_{old}\)) and the
distribution being <em>updated</em> (\(\pi_\theta\)) have diverged, invalidating the
on-policy assumption. PPO's fix is to reuse the same rollout for several gradient steps
via importance sampling, but clip the importance ratio so a single step can't move the
policy arbitrarily far from \(\pi_{old}\):</p>

\[r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}, \qquad
\mathcal{L}^{CLIP}(\theta) = -\mathbb{E}_t\Big[\min\big(r_t(\theta)\,A_t,\; \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,A_t\big)\Big]\]

<div class="definition">
<strong>Why the min, not just the clip?</strong> Taking the min of the clipped and unclipped
terms makes the objective <em>pessimistic</em>: if the ratio has moved past the clip range
in the direction that <em>would increase</em> the objective, the clipped term (which no
longer has gradient signal pushing the ratio further) is used, capping the incentive to keep
pushing that action's probability up or down beyond the trust region. But if the ratio has
moved past the clip range in a direction that <em>decreases</em> the objective (e.g. the
policy previously moved probability mass away from a good action too far), the unclipped
term is smaller (more negative) and is used instead — so clipping never protects the policy
from being pushed back, only from being pushed further in an already-adopted direction.
Concretely: with positive advantage, clipping bounds how much probability mass can be added
to a good action; with negative advantage, clipping bounds how much can be removed from a
bad action — in both cases, once the ratio leaves \([1-\epsilon, 1+\epsilon]\).</div>

<h3>Why a Value Function Is Needed</h3>

<p>Without a value baseline, the advantage collapses to the raw discounted return, which has
much higher variance (it includes the variance of every future token's stochastic sampling
and reward, not just the current action's marginal contribution). A learned \(V(s_t)\)
absorbs the state-dependent part of the return, leaving the advantage to isolate the
action's marginal effect — this is the standard variance-reduction argument for
actor-critic methods generally, not specific to PPO.</p>

<h3>Why PPO Is Comparatively Complex and Fragile</h3>

<p>Relative to SFT or DPO (Section 7), PPO requires maintaining and coordinating <strong>four</strong>
models simultaneously: the policy being trained, a frozen reference policy (for the KL
penalty), a frozen reward model (for the terminal reward), and a value function (often
sharing weights with the policy, as here). Each rollout is an autoregressive generation
loop (expensive relative to a single forward pass), and training is sensitive to several
interacting hyperparameters (\(\beta\), \(\epsilon\), GAE's \(\lambda\), the relative
learning rates of policy vs. value head) — a poor setting of any one can produce reward
hacking, KL blowup (policy diverges from anything fluent), or value function collapse
(advantages become uninformative). This fragility, not any flaw in the underlying objective,
is the primary practical motivation for DPO (Section 7), which optimizes the same
KL-constrained objective without rollouts, a reward model, or a value function.</p>

<div class="keyfacts">
<strong>Key Facts — Section 6</strong>
<ul>
  <li>PPO maximizes expected reward minus a KL penalty to the frozen SFT policy — reward
  alone would reward-hack (Section 5); the KL term bounds how far the policy can drift.</li>
  <li>GAE trades off bias and variance in the advantage estimate via \(\lambda\); it reduces
  to Monte-Carlo return at \(\lambda=1\) and one-step TD at \(\lambda=0\).</li>
  <li>The clipped surrogate objective lets a single rollout be reused for several gradient
  steps (needed because true on-policy REINFORCE would need a fresh rollout per step) while
  bounding how far any one update can move the policy from \(\pi_{old}\).</li>
  <li>PPO's operational complexity (4 coordinated models, sensitive hyperparameters) is the
  main practical argument for DPO when a reward model isn't otherwise needed.</li>
</ul>
</div>
```

- [ ] **Step 3: Verify**

```bash
.venv/bin/python -c "
html = open('docs/llm_training_pipeline_reference.html', encoding='utf-8').read()
assert '<h2 id=\"s5\"' in html and '<h2 id=\"s6\"' in html
assert 'href=\"#s5\"' in html and 'href=\"#s6\"' in html
assert html.count('\\\\[') == html.count('\\\\]')
print('HTML sections 5-6 OK,', len(html), 'bytes')
"
```

Expected: `HTML sections 5-6 OK, <N> bytes`

- [ ] **Step 4: Commit**

```bash
git add docs/llm_training_pipeline_reference.html
git commit -m "docs: add reward modeling and PPO/RLHF sections to LLM training pipeline reference"
```

---

## Task 2: Concepts Q&A — Sections 11-14 (Reward Modeling + PPO)

**Files:**
- Modify: `docs/llm_training_pipeline/concepts_qa.md`

- [ ] **Step 1: Replace the trailing placeholder note with Sections 11-14**

Find the final line of the file (left by Part 2):

```markdown
*(Sections for reward modeling, PPO, DPO, evaluation, and RLVR/GRPO are
appended here as the corresponding notebooks are built.)*
```

Replace it with:

```markdown
## 11. Bradley-Terry from first principles — why a logistic link, specifically?

Bradley-Terry (1952) models pairwise comparison outcomes for a set of items with
latent strengths `s_i > 0` as `P(i > j) = s_i / (s_i + s_j)`. This isn't an
arbitrary functional form — it's the unique form (up to reparameterization)
satisfying **Luce's choice axiom** (independence of irrelevant alternatives):
the relative odds of choosing `i` over `j` shouldn't depend on what *other*
items are also available in a larger choice set. Substituting `s_i = exp(r_i)`
turns the ratio into a difference in log-space:

```
P(i > j) = exp(r_i) / (exp(r_i) + exp(r_j))
         = 1 / (1 + exp(-(r_i - r_j)))
         = sigmoid(r_i - r_j)
```

This is exactly logistic regression on the *difference* of two learned scores —
which is why the reward model training loss (`-log sigmoid(r_chosen -
r_rejected)`) is literally a binary cross-entropy loss with the "logit" being
`r_chosen - r_rejected`. Concretely: `bradley_terry_loss` in
`src/llm_pipeline/rlhf.py` is `F.binary_cross_entropy_with_logits`-equivalent,
just written via `logsigmoid` directly. There is no separate "Bradley-Terry
loss function" beyond ordinary logistic regression applied to reward
*differences* rather than raw features.

---

## 12. GAE derivation — why the recursive form, and what gamma/lambda actually trade off

Starting from the n-step advantage estimator family: for horizon `n`,

```
A_t^(n) = sum_{l=0}^{n-1} gamma^l * r_{t+l} + gamma^n * V(s_{t+n}) - V(s_t)
```

`n=1` gives the one-step TD advantage `A_t^(1) = delta_t = r_t + gamma*V(s_{t+1})
- V(s_t)` (low variance — only one reward sample and one value estimate — but
biased by however wrong `V` currently is). `n=infinity` gives the Monte-Carlo
advantage `A_t^(inf) = sum_{l>=0} gamma^l r_{t+l} - V(s_t)` (unbiased, since it
uses only observed rewards, but high variance — it accumulates every
downstream token's sampling randomness).

GAE (Schulman et al. 2016) doesn't pick one `n` — it takes an
exponentially-weighted average over *all* of them, controlled by `lambda`:

```
A_t^GAE(gamma,lambda) = (1 - lambda) * sum_{n=1}^{inf} lambda^(n-1) * A_t^(n)
```

This infinite sum telescopes (the derivation substitutes the n-step formula in
and collects terms by `delta`) into the closed, recursive form actually
implemented:

```
A_t^GAE = delta_t + (gamma*lambda) * A_{t+1}^GAE
```

computed by sweeping backward through the trajectory from the last timestep
(`A_T = 0`, no future beyond the episode) to the first. `lambda=0` collapses
the weighted sum to just the `n=1` term (`A_t = delta_t`, one-step TD).
`lambda=1` gives every `n` equal infinite... no — concretely, taking the limit
`lambda -> 1` in the closed form removes the `gamma^n * V(s_{t+n})` bootstrap
terms entirely (they telescope away against each other across timesteps),
recovering the full Monte-Carlo advantage. Practically: `lambda` close to 1
(0.95 here, matching the original GAE paper's recommendation and standard PPO
implementations) accepts a bit more variance for less dependence on how
accurate the value function currently is, especially early in training when
`V` is still a poor estimate.

---

## 13. The clipped surrogate objective — a worked numeric example

Take `clip_eps = 0.2`, and suppose the ratio `r = pi_new(a|s) / pi_old(a|s)`
comes out to `1.3` (the new policy has become 30% more likely to take this
action than the policy that generated the rollout), with advantage `A = +1`
(this action was better than average):

- Unclipped term: `r * A = 1.3 * 1 = 1.3`
- Clipped term: `clip(1.3, 0.8, 1.2) * A = 1.2 * 1 = 1.2`
- `min(1.3, 1.2) = 1.2` — the **clipped** term is used.

Because the objective (before the `-1` for gradient descent) takes the `min`,
and gradient only flows through whichever branch is selected, using the
clipped branch here means **no further gradient signal pushes this action's
probability up beyond the 1.2 ratio** — the clip has done its job of stopping
this particular update from moving further in a direction it already moved a
lot in.

Now suppose instead `r = 0.7` with the *same* positive advantage `A = +1`
(the new policy has *decreased* the probability of a good action — this is
the ratio moving in the direction that *hurts* the objective):

- Unclipped: `0.7 * 1 = 0.7`
- Clipped: `clip(0.7, 0.8, 1.2) * 1 = 0.8 * 1 = 0.8`
- `min(0.7, 0.8) = 0.7` — the **unclipped** term is used, and its gradient
  (which pushes the ratio back up, i.e. corrects the mistake) is preserved.

This confirms the asymmetry stated in the HTML reference: clipping only
activates to *prevent further movement in an already-taken direction*; it
never blocks a correction. `src/llm_pipeline/rlhf.py`'s `ppo_clipped_loss`
test cases (Notebook 3, TEST 6) verify both branches numerically.

---

## 14. Reward hacking and the KL budget — what "well-regularized" vs. "overoptimized" look like

Gao, Schulman & Hilton, 2022 ("Scaling Laws for Reward Model Overoptimization")
run PPO against reward models of varying quality and plot **gold-standard
reward** (a separate, higher-fidelity reward signal treated as ground truth)
against **KL divergence from the reference policy** over the course of
training. Their key finding: gold reward increases with KL up to a point, then
turns over and *decreases* — the policy has found ways to exploit the proxy
reward model that a better judge would penalize. The KL at which this turnover
happens (the "KL budget") is a measurable property of a given reward model's
quality, and it shrinks as the reward model gets noisier or more biased
relative to the true objective.

Two visibly different training curves this implies (both plotted in Notebook
5's evaluation stage, using the proxy reward model's *own* score against KL,
since no separate gold-standard judge is used in this pipeline):

- **Well-regularized** (adequate `beta`, moderate training length): reward
  rises with KL and the curve is monotonically increasing across the training
  run — KL never grows large enough to reach the point where the reward
  model's proxy-ness becomes exploitable.
- **Overoptimized** (`beta` too small, or trained for far more steps than
  this pipeline's 150): reward keeps climbing according to the *proxy* reward
  model even as KL grows very large, while qualitative inspection of
  generations reveals degenerate, repetitive, or off-topic text — the proxy
  reward and true quality have decoupled. Concretely for this pipeline's
  sentiment-based proxy: an overoptimized policy could learn to emit strings
  of maximally-positive-sentiment words ("happy happy wonderful joy...")
  divorced from coherent story structure, since the sentiment classifier, not
  a coherence judge, is what's actually being optimized against.

---

*(Sections for DPO, evaluation, and RLVR/GRPO are appended here as the
corresponding notebooks are built.)*
```

- [ ] **Step 2: Verify**

```bash
.venv/bin/python -c "
text = open('docs/llm_training_pipeline/concepts_qa.md', encoding='utf-8').read()
n_sections = text.count('\n## ')
assert n_sections == 14, f'expected 14 sections, found {n_sections}'
assert '## 11. Bradley-Terry' in text
assert '## 14. Reward hacking' in text
print('Q&A doc OK,', len(text), 'bytes,', n_sections, 'sections')
"
```

Expected: `Q&A doc OK, <N> bytes, 14 sections`

- [ ] **Step 3: Commit**

```bash
git add docs/llm_training_pipeline/concepts_qa.md
git commit -m "docs: add reward modeling and PPO concepts Q&A (Bradley-Terry, GAE, clipping, reward hacking)"
```

---

## Task 3: Notebook Bootstrap

**Files:**
- Create: `notebooks/build_llm_pipeline_03_reward_model_and_ppo_notebook.py`

**Interfaces:**
- Produces: a `cells` list and `md()`/`code()` helpers that Tasks 4-7 append to; a final `nbf.write(nb, OUTPUT_PATH)` call.

- [ ] **Step 1: Verify Part 1+2's artifacts exist**

```bash
.venv/bin/python -c "
import os
base = 'data/checkpoints/llm_training_pipeline'
for f in ['sft_model.pt', 'base_model.pt', 'tinystories_bpe-vocab.json', 'tinystories_bpe-merges.txt']:
    p = os.path.join(base, f)
    assert os.path.exists(p), f'missing {p} — run Part 1 and Part 2 plans first'
print('Part 1+2 artifacts present')
"
```

Expected: `Part 1+2 artifacts present`. If this fails, stop and execute
`docs/superpowers/plans/2026-07-02-llm-training-pipeline-02-sft.md` first (which itself
depends on Part 1).

- [ ] **Step 2: Write the builder script skeleton**

Create `notebooks/build_llm_pipeline_03_reward_model_and_ppo_notebook.py`:

```python
"""
Generates notebooks/llm_training_pipeline/03_reward_model_and_ppo.ipynb from cell
definitions.
Run: python3 notebooks/build_llm_pipeline_03_reward_model_and_ppo_notebook.py
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
# LLM Training Pipeline — Part 3: Reward Model & PPO

Stage 3 of 6. Builds a preference dataset by sampling and sentiment-scoring SFT
completions, trains a reward model on it, then runs a full from-scratch PPO
loop (rollout, GAE, clipped surrogate objective, KL penalty) against
`sft_model.pt` to produce `ppo_model.pt`. This is the heaviest notebook in the
series — expect more moving parts than notebooks 1-2.

**How to use this notebook:**
- Read each theory section; keep `docs/llm_training_pipeline_reference.html`
  open in another tab (Sections 5-6) for the full derivations.
- Code and tests are already implemented and verified — run cells top to
  bottom. Answer the **Question** cells yourself.

**Parts:**
1. Preference Dataset Construction
2. Reward Model
3. PPO Core (rollout, GAE, clipped objective)
4. PPO Training Loop
"""))

# ─── SETUP ───────────────────────────────────────────────────────────────────
cells.append(code("""
import time, math, os, json, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tokenizers import ByteLevelBPETokenizer
from transformers import pipeline as hf_pipeline

import sys
sys.path.insert(0, '../..')
from src.llm_pipeline.model import GPTConfig, GPTModel, Block
from src.llm_pipeline.data import TOPIC_KEYWORDS, format_sft_prompt

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
"""))

# Parts 1-4 are appended here by Tasks 4-7.

# ─── WRITE ───────────────────────────────────────────────────────────────────
nb['cells'] = cells
OUTPUT_PATH = "llm_training_pipeline/03_reward_model_and_ppo.ipynb"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    nbf.write(nb, f)
print(f"Wrote {OUTPUT_PATH} with {len(cells)} cells")
```

- [ ] **Step 3: Generate and execute**

```bash
cd notebooks && ../.venv/bin/python build_llm_pipeline_03_reward_model_and_ppo_notebook.py && cd ..
.venv/bin/python -m nbconvert --to notebook --execute --inplace \
  notebooks/llm_training_pipeline/03_reward_model_and_ppo.ipynb
```

Expected: `Wrote llm_training_pipeline/03_reward_model_and_ppo.ipynb with 2 cells`, then
notebook execution completes without error, printing `Loaded sft_model.pt — 13,817,856 params`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/build_llm_pipeline_03_reward_model_and_ppo_notebook.py notebooks/llm_training_pipeline/03_reward_model_and_ppo.ipynb
git commit -m "feat: bootstrap llm_training_pipeline notebook 3 (intro + setup)"
```

---

## Task 4: Notebook Part 1 — Preference Dataset Construction

**Files:**
- Modify: `notebooks/build_llm_pipeline_03_reward_model_and_ppo_notebook.py`

**Interfaces:**
- Consumes: `sft_model`, `tokenizer`, `EOT_ID`, `CKPT_DIR`, `device` from Task 3; `TOPIC_KEYWORDS`, `format_sft_prompt` from `src.llm_pipeline.data`.
- Produces (notebook runtime namespace): `sentiment_score(text: str) -> float`, `preference_pairs` (list of dicts with keys `prompt`, `chosen`, `rejected`, `chosen_score`, `rejected_score`). Also writes `data/checkpoints/llm_training_pipeline/preference_pairs.json`.

- [ ] **Step 1: Append the Part 1 cells**

Insert before the `# Parts 1-4 are appended here` comment:

```python
# ─── PART 1: PREFERENCE DATASET ──────────────────────────────────────────────
cells.append(md("""
---
## Part 1: Preference Dataset Construction

For each topic, sample a *group* of completions from `sft_model` at temperature, score
each with a sentiment classifier (a cheap, automatable stand-in for a human preference
judgment), and pair the highest- and lowest-scoring completions as `(chosen, rejected)`.
Groups where the best and worst score are indistinguishable are dropped (no clear
preference). See `docs/llm_training_pipeline_reference.html#s5` for why a reward model
needs pairwise, not absolute, preference data.
"""))

cells.append(code("""
sentiment_pipe = hf_pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if device == 'cuda' else -1,
)

def sentiment_score(text):
    \"\"\"Signed scalar in [-1, 1]: +confidence for POSITIVE, -confidence for NEGATIVE.\"\"\"
    result = sentiment_pipe(text[:512])[0]
    sign = 1.0 if result['label'] == 'POSITIVE' else -1.0
    return sign * result['score']
"""))

cells.append(code("""
# TEST 1: sentiment scorer sanity
pos_score = sentiment_score("I am so happy today, everything is wonderful and bright!")
neg_score = sentiment_score("This is terrible, I am so sad and scared and everything is awful.")
assert pos_score > 0, f"expected positive score, got {pos_score}"
assert neg_score < 0, f"expected negative score, got {neg_score}"
print(f"TEST 1 PASSED — sentiment scorer sane (pos={pos_score:.3f}, neg={neg_score:.3f})")
"""))

cells.append(code("""
N_PROMPTS_PER_TOPIC = 5
GROUP_SIZE = 6
MAX_NEW_TOKENS = 40

@torch.no_grad()
def sample_group(topic, group_size, max_new_tokens):
    prompt = format_sft_prompt(topic)
    prompt_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    prompt_batch = prompt_ids.repeat(group_size, 1)
    out = sft_model.generate(prompt_batch, max_new_tokens=max_new_tokens, temperature=1.0, top_k=40)
    completions = [tokenizer.decode(out[i, prompt_ids.shape[1]:].tolist()) for i in range(group_size)]
    return prompt, completions

preference_pairs = []
t0 = time.time()
for topic in TOPIC_KEYWORDS:
    for _ in range(N_PROMPTS_PER_TOPIC):
        prompt, completions = sample_group(topic, GROUP_SIZE, MAX_NEW_TOKENS)
        scores = [sentiment_score(c) for c in completions]
        best_i = max(range(GROUP_SIZE), key=lambda i: scores[i])
        worst_i = min(range(GROUP_SIZE), key=lambda i: scores[i])
        if abs(scores[best_i] - scores[worst_i]) < 1e-3:
            continue  # tie — no clear preference, drop
        preference_pairs.append({
            'prompt': prompt,
            'chosen': completions[best_i],
            'rejected': completions[worst_i],
            'chosen_score': scores[best_i],
            'rejected_score': scores[worst_i],
        })
print(f"Built {len(preference_pairs)} preference pairs in {time.time()-t0:.0f}s "
      f"from {len(TOPIC_KEYWORDS) * N_PROMPTS_PER_TOPIC} prompt instances")
"""))

cells.append(code("""
# TEST 2: pair construction sanity
assert len(preference_pairs) > 100, f"expected >100 pairs, got {len(preference_pairs)}"
for p in preference_pairs[:5]:
    assert p['chosen_score'] > p['rejected_score']
print(f"TEST 2 PASSED — {len(preference_pairs)} pairs, chosen_score > rejected_score verified on samples")
"""))

cells.append(md("""
### Question 1

The rejected completion in each pair is the *lowest-sentiment* completion sampled in that
group — not necessarily a *bad* completion by any other standard (grammar, topicality,
coherence). Given this construction, what does the reward model in Part 2 actually learn to
prefer? Is that the same thing as "prefer better stories"?

*Write your answer below:*

"""))

cells.append(code("""
with open(f"{CKPT_DIR}/preference_pairs.json", 'w') as f:
    json.dump(preference_pairs, f, indent=2)
print(f"Saved {len(preference_pairs)} preference pairs to {CKPT_DIR}/preference_pairs.json")
"""))

# Parts 2-4 are appended here.
```

- [ ] **Step 2: Regenerate and execute**

```bash
cd notebooks && ../.venv/bin/python build_llm_pipeline_03_reward_model_and_ppo_notebook.py && cd ..
.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 \
  notebooks/llm_training_pipeline/03_reward_model_and_ppo.ipynb
```

Expected: no errors; output includes `TEST 1 PASSED`, `TEST 2 PASSED`, and `Saved <N>
preference pairs to ...preference_pairs.json`. This step downloads and caches the
sentiment classifier on first run and takes several minutes (200 prompt instances × 6
completions each, plus scoring).

- [ ] **Step 3: Commit**

```bash
git add notebooks/build_llm_pipeline_03_reward_model_and_ppo_notebook.py notebooks/llm_training_pipeline/03_reward_model_and_ppo.ipynb
git commit -m "feat: llm_training_pipeline notebook 3 part 1 — preference dataset construction"
```

---

## Task 5: Notebook Part 2 — Reward Model

**Files:**
- Modify: `notebooks/build_llm_pipeline_03_reward_model_and_ppo_notebook.py`

**Interfaces:**
- Consumes: `GPTConfig`, `GPTModel`, `Block` from Task 3; `sft_model`, `sft_cfg`, `tokenizer`, `EOT_ID`, `BLOCK_SIZE`, `device` from Task 3; `preference_pairs` from Task 4.
- Produces (notebook runtime namespace): `RewardModel`, `load_trunk_from_sft`, `bradley_terry_loss`, `encode_pair_text(prompt, completion, block_size) -> (LongTensor[block_size], int)`, `reward_model` (trained instance).

- [ ] **Step 1: Append the Part 2 cells**

```python
# ─── PART 2: REWARD MODEL ────────────────────────────────────────────────────
cells.append(md("""
---
## Part 2: Reward Model

The reward model shares the GPT trunk architecture with the policy but replaces the tied LM
head with a single scalar head applied to the hidden state at the completion's last real
token (not necessarily the last position in the padded tensor — see the length-aware
gather below). Its trunk is initialized from `sft_model`'s weights. See
`docs/llm_training_pipeline_reference.html#s5` for the Bradley-Terry derivation.
"""))

cells.append(code("""
class RewardModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.reward_head = nn.Linear(config.n_embd, 1, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, lengths=None):
        \"\"\"idx: (B, T) padded token ids. lengths: (B,) real (unpadded) sequence
        lengths, or None to use position T-1 for every example (only correct when
        every sequence in the batch fills the full T with no trailing padding).\"\"\"
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        if lengths is None:
            last = x[:, -1, :]
        else:
            last_idx = (lengths - 1).clamp(min=0)
            last = x[torch.arange(B, device=x.device), last_idx, :]
        return self.reward_head(last).squeeze(-1)


def load_trunk_from_sft(reward_model, sft_state_dict):
    \"\"\"Copies token/position embeddings, transformer blocks, and final LayerNorm
    from an SFT GPTModel's state dict into a freshly-initialized RewardModel,
    leaving reward_head randomly initialized.\"\"\"
    trunk_keys = [k for k in sft_state_dict if k.startswith(('tok_emb', 'pos_emb', 'blocks', 'ln_f'))]
    own_state = reward_model.state_dict()
    for k in trunk_keys:
        own_state[k].copy_(sft_state_dict[k])


def bradley_terry_loss(reward_chosen, reward_rejected):
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()


def encode_pair_text(prompt, completion, block_size):
    \"\"\"Returns (padded_ids: LongTensor[block_size], real_length: int).\"\"\"
    ids = (tokenizer.encode(prompt).ids + tokenizer.encode(completion).ids + [EOT_ID])[:block_size]
    length = len(ids)
    ids = ids + [EOT_ID] * (block_size - length)
    return torch.tensor(ids, dtype=torch.long), length
"""))

cells.append(code("""
# TEST 3: trunk transplant, Bradley-Terry loss sanity, and length-aware gather correctness
rm_cfg = sft_cfg
reward_model = RewardModel(rm_cfg).to(device)
load_trunk_from_sft(reward_model, sft_model.state_dict())
assert torch.allclose(reward_model.tok_emb.weight, sft_model.tok_emb.weight)
assert torch.allclose(reward_model.blocks[0].attn.qkv_proj.weight, sft_model.blocks[0].attn.qkv_proj.weight)
print("TEST 3a PASSED — trunk weights transplanted correctly from sft_model")

rc = torch.tensor([2.0, 0.0])
rr = torch.tensor([0.0, 2.0])
loss = bradley_terry_loss(rc, rr)
expected = -(torch.log(torch.sigmoid(torch.tensor(2.0))) + torch.log(torch.sigmoid(torch.tensor(-2.0)))) / 2
assert torch.allclose(loss, expected, atol=1e-5), f"{loss.item()} != {expected.item()}"
print(f"TEST 3b PASSED — Bradley-Terry loss matches hand-computed value ({loss.item():.4f})")

reward_model.eval()
ids_short, len_short = encode_pair_text("dog", "A dog ran fast.", 64)
ids_long, len_long = encode_pair_text("dog", "A dog ran fast.", 256)
with torch.no_grad():
    r_short = reward_model(ids_short.unsqueeze(0).to(device), torch.tensor([len_short], device=device))
    r_long = reward_model(ids_long.unsqueeze(0).to(device), torch.tensor([len_long], device=device))
assert torch.allclose(r_short, r_long, atol=1e-4), (
    f"reward changed with padding length alone ({r_short.item():.4f} vs {r_long.item():.4f}) — "
    "the length-aware gather is not correctly ignoring trailing padding"
)
print(f"TEST 3c PASSED — reward is invariant to trailing padding length ({r_short.item():.4f})")
reward_model.train()
"""))
```

- [ ] **Step 2: Append the RM training loop cells**

```python
cells.append(code("""
held_out_pairs = preference_pairs[-30:]
train_pairs = preference_pairs[:-30]
print(f"{len(train_pairs)} training pairs, {len(held_out_pairs)} held-out pairs")

def make_rm_batch(pairs, batch_size):
    idx = torch.randint(0, len(pairs), (batch_size,))
    chosen = [encode_pair_text(pairs[i]['prompt'], pairs[i]['chosen'], BLOCK_SIZE) for i in idx]
    rejected = [encode_pair_text(pairs[i]['prompt'], pairs[i]['rejected'], BLOCK_SIZE) for i in idx]
    chosen_ids = torch.stack([c[0] for c in chosen]).to(device)
    chosen_lens = torch.tensor([c[1] for c in chosen], device=device)
    rejected_ids = torch.stack([r[0] for r in rejected]).to(device)
    rejected_lens = torch.tensor([r[1] for r in rejected], device=device)
    return chosen_ids, chosen_lens, rejected_ids, rejected_lens
"""))

cells.append(code("""
rm_max_steps = 300
rm_lr = 1e-4
rm_batch_size = 16

opt = torch.optim.AdamW(reward_model.parameters(), lr=rm_lr)
rm_losses = []
t0 = time.time()
for step in range(rm_max_steps):
    chosen_ids, chosen_lens, rejected_ids, rejected_lens = make_rm_batch(train_pairs, rm_batch_size)
    r_chosen = reward_model(chosen_ids, chosen_lens)
    r_rejected = reward_model(rejected_ids, rejected_lens)
    loss = bradley_terry_loss(r_chosen, r_rejected)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
    opt.step()
    rm_losses.append(loss.item())
    if step % 50 == 0 or step == rm_max_steps - 1:
        print(f"step {step:4d} | loss {loss.item():.3f} | elapsed {time.time()-t0:.0f}s")
print(f"Reward model training elapsed: {time.time()-t0:.1f}s")
"""))

cells.append(code("""
plt.figure(figsize=(8, 4))
plt.plot(rm_losses, alpha=0.6, label="per-step Bradley-Terry loss")
window = 20
smoothed = [sum(rm_losses[max(0,i-window):i+1]) / len(rm_losses[max(0,i-window):i+1]) for i in range(len(rm_losses))]
plt.plot(smoothed, label=f"{window}-step moving average", linewidth=2)
plt.xlabel("step"); plt.ylabel("Bradley-Terry loss"); plt.title("Reward model training loss")
plt.legend(); plt.tight_layout(); plt.show()
"""))

cells.append(code("""
# TEST 4: held-out ranking accuracy
reward_model.eval()
with torch.no_grad():
    ho_chosen = [encode_pair_text(p['prompt'], p['chosen'], BLOCK_SIZE) for p in held_out_pairs]
    ho_rejected = [encode_pair_text(p['prompt'], p['rejected'], BLOCK_SIZE) for p in held_out_pairs]
    ho_chosen_ids = torch.stack([c[0] for c in ho_chosen]).to(device)
    ho_chosen_lens = torch.tensor([c[1] for c in ho_chosen], device=device)
    ho_rejected_ids = torch.stack([r[0] for r in ho_rejected]).to(device)
    ho_rejected_lens = torch.tensor([r[1] for r in ho_rejected], device=device)
    r_chosen = reward_model(ho_chosen_ids, ho_chosen_lens)
    r_rejected = reward_model(ho_rejected_ids, ho_rejected_lens)
ranking_acc = (r_chosen > r_rejected).float().mean().item()
print(f"held-out ranking accuracy: {ranking_acc:.3f}")
assert ranking_acc > 0.6, f"ranking accuracy {ranking_acc:.3f} not above chance+margin (0.6)"
print("TEST 4 PASSED — reward model ranks held-out chosen > rejected well above chance")
reward_model.train()

# Freeze the reward model — it is used read-only from here on (PPO's terminal reward).
reward_model.eval()
for p in reward_model.parameters():
    p.requires_grad_(False)
"""))

cells.append(md("""
### Question 2

TEST 4's ranking accuracy measures whether the reward model agrees with the *sentiment
classifier's* ranking on held-out pairs — not whether it agrees with a human's judgment of
story quality (Question 1). Given that, what would rising PPO reward (Part 4) actually
demonstrate — that the policy is producing "better stories", or something narrower? What
additional evidence (beyond the reward number) would you want before concluding PPO
"worked"?

*Write your answer below:*

"""))

# Parts 3-4 are appended here.
```

- [ ] **Step 3: Regenerate and execute**

```bash
cd notebooks && ../.venv/bin/python build_llm_pipeline_03_reward_model_and_ppo_notebook.py && cd ..
.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 \
  notebooks/llm_training_pipeline/03_reward_model_and_ppo.ipynb
```

Expected: no errors; output includes `TEST 3a/3b/3c PASSED` and `TEST 4 PASSED`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/build_llm_pipeline_03_reward_model_and_ppo_notebook.py notebooks/llm_training_pipeline/03_reward_model_and_ppo.ipynb
git commit -m "feat: llm_training_pipeline notebook 3 part 2 — reward model"
```

---

## Task 6: Notebook Part 3 — PPO Core

**Files:**
- Modify: `notebooks/build_llm_pipeline_03_reward_model_and_ppo_notebook.py`

**Interfaces:**
- Consumes: `GPTModel`, `sft_model`, `sft_cfg`, `BLOCK_SIZE`, `device` from Task 3.
- Produces (notebook runtime namespace): `PPOActorCritic`, `generate_rollout`, `compute_token_rewards`, `compute_gae`, `ppo_clipped_loss`, `evaluate_actions`.

- [ ] **Step 1: Append the Part 3 cells**

```python
# ─── PART 3: PPO CORE ────────────────────────────────────────────────────────
cells.append(md("""
---
## Part 3: PPO Core

Implements the pieces from `docs/llm_training_pipeline_reference.html#s6`: an actor-critic
wrapper around the policy (adds a value head without modifying `GPTModel`), rollout
generation that records both the policy's and a frozen reference model's log-probabilities
at each sampled token, GAE, and the clipped surrogate objective. Every piece is tested
against a hand-computed toy example before being used in the training loop (Part 4).
"""))

cells.append(code("""
class PPOActorCritic(nn.Module):
    \"\"\"Wraps a GPTModel, exposing both LM logits and a per-position scalar value
    estimate. Reuses the wrapped model's tok_emb/pos_emb/drop/blocks/ln_f/lm_head
    directly — no changes to src/llm_pipeline/model.py.\"\"\"
    def __init__(self, gpt: GPTModel):
        super().__init__()
        self.gpt = gpt
        self.value_head = nn.Linear(gpt.config.n_embd, 1, bias=False)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.gpt.tok_emb(idx) + self.gpt.pos_emb(pos)
        x = self.gpt.drop(x)
        for block in self.gpt.blocks:
            x = block(x)
        x = self.gpt.ln_f(x)
        logits = self.gpt.lm_head(x)
        values = self.value_head(x).squeeze(-1)  # (B, T)
        return logits, values
"""))

cells.append(code("""
@torch.no_grad()
def generate_rollout(actor_critic, ref_model, prompt_ids, max_new_tokens, temperature, top_k, block_size):
    \"\"\"Samples max_new_tokens autoregressively from actor_critic, recording the
    policy's log-prob, the frozen ref_model's log-prob, and the value estimate at
    each sampled token. Returns (idx, policy_logprobs, ref_logprobs, values), each
    of shape (B, max_new_tokens) except idx which is (B, prompt_len + max_new_tokens).\"\"\"
    idx = prompt_ids.clone()
    policy_logprobs, ref_logprobs, values = [], [], []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, vals = actor_critic(idx_cond)
        logits_last = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits_last, top_k)
            logits_last[logits_last < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits_last, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        policy_lp = F.log_softmax(logits_last, dim=-1).gather(1, next_id).squeeze(-1)

        ref_logits, _ = ref_model(idx_cond)
        ref_lp = F.log_softmax(ref_logits[:, -1, :], dim=-1).gather(1, next_id).squeeze(-1)

        idx = torch.cat([idx, next_id], dim=1)
        policy_logprobs.append(policy_lp)
        ref_logprobs.append(ref_lp)
        values.append(vals[:, -1])
    return (
        idx,
        torch.stack(policy_logprobs, dim=1),
        torch.stack(ref_logprobs, dim=1),
        torch.stack(values, dim=1),
    )


def compute_token_rewards(policy_logprobs, ref_logprobs, terminal_reward, kl_beta):
    \"\"\"Per-token reward = -kl_beta * KL at every step, plus terminal_reward added
    only at the last generated token (the reward model scores whole completions).
    Returns (rewards, kl), both (B, T).\"\"\"
    kl = policy_logprobs - ref_logprobs
    rewards = -kl_beta * kl
    rewards = rewards.clone()
    rewards[:, -1] = rewards[:, -1] + terminal_reward
    return rewards, kl


def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    \"\"\"rewards, values: (B, T). Returns (advantages, returns), both (B, T).
    Bootstraps with a next_value of 0 beyond the last generated token (episode end).\"\"\"
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)
    next_value = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae
        next_value = values[:, t]
    returns = advantages + values
    return advantages, returns


def ppo_clipped_loss(new_logprobs, old_logprobs, advantages, clip_eps=0.2):
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()


def evaluate_actions(policy, idx, prompt_len, gen_len):
    \"\"\"Re-runs the (now-updated) policy over the full generated sequence and
    extracts log-probs and values at exactly the positions/tokens that were
    sampled during the rollout. Returns (logprobs, values), both (B, gen_len).\"\"\"
    logits, values = policy(idx[:, :-1])
    action_logits = logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :]
    action_values = values[:, prompt_len - 1 : prompt_len - 1 + gen_len]
    actions = idx[:, prompt_len : prompt_len + gen_len]
    logprobs = F.log_softmax(action_logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return logprobs, action_values
"""))

cells.append(code("""
# TEST 5: GAE against a hand-computed toy trajectory (gamma=1, lambda=1 -> Monte Carlo advantage)
rewards = torch.tensor([[1.0, 0.0, 2.0]])
values = torch.tensor([[0.5, 0.5, 0.5]])
adv, ret = compute_gae(rewards, values, gamma=1.0, lam=1.0)
# hand derivation: return_t = sum of rewards from t onward; advantage_t = return_t - value_t
# return_2 = 2.0 -> adv_2 = 2.0 - 0.5 = 1.5
# return_1 = 0.0 + 2.0 = 2.0 -> adv_1 = 2.0 - 0.5 = 1.5
# return_0 = 1.0 + 0.0 + 2.0 = 3.0 -> adv_0 = 3.0 - 0.5 = 2.5
expected_adv = torch.tensor([[2.5, 1.5, 1.5]])
expected_ret = torch.tensor([[3.0, 2.0, 2.0]])
assert torch.allclose(adv, expected_adv, atol=1e-5), f"{adv} != {expected_adv}"
assert torch.allclose(ret, expected_ret, atol=1e-5), f"{ret} != {expected_ret}"
print("TEST 5 PASSED — GAE matches hand-computed toy trajectory")
"""))

cells.append(code("""
# TEST 6: clipped surrogate objective at the clip boundary (both advantage signs)
clip_eps = 0.2
old_lp = torch.tensor([0.0, 0.0])

new_lp_high = torch.log(torch.tensor([1.3, 1.3]))  # ratio = 1.3 -> clipped to 1.2
adv_pos = torch.tensor([1.0, 1.0])
loss_pos = ppo_clipped_loss(new_lp_high, old_lp, adv_pos, clip_eps)
# unclipped = 1.3*1 = 1.3, clipped = 1.2*1 = 1.2, min = 1.2, loss = -1.2
assert abs(loss_pos.item() - (-1.2)) < 1e-4, f"expected -1.2, got {loss_pos.item()}"
print(f"TEST 6a PASSED — positive-advantage clip boundary uses the clipped (pessimistic) term: loss={loss_pos.item():.4f}")

new_lp_low = torch.log(torch.tensor([0.7, 0.7]))  # ratio = 0.7 -> clipped to 0.8
adv_neg = torch.tensor([-1.0, -1.0])
loss_neg = ppo_clipped_loss(new_lp_low, old_lp, adv_neg, clip_eps)
# unclipped = 0.7*-1 = -0.7, clipped = 0.8*-1 = -0.8, min(-0.7,-0.8) = -0.8, loss = 0.8
assert abs(loss_neg.item() - 0.8) < 1e-4, f"expected 0.8, got {loss_neg.item()}"
print(f"TEST 6b PASSED — negative-advantage clip boundary uses the clipped (pessimistic) term: loss={loss_neg.item():.4f}")

# Sanity: an unclipped ratio (inside [0.8, 1.2]) must equal ratio * advantage exactly.
new_lp_mid = torch.log(torch.tensor([1.05, 1.05]))
loss_mid = ppo_clipped_loss(new_lp_mid, old_lp, adv_pos, clip_eps)
assert abs(loss_mid.item() - (-1.05)) < 1e-4, f"expected -1.05, got {loss_mid.item()}"
print(f"TEST 6c PASSED — ratio inside the clip range is left unclipped: loss={loss_mid.item():.4f}")
"""))

cells.append(md("""
### Question 3

`compute_token_rewards` adds the reward model's terminal score only at the *last*
generated token, while the KL penalty is subtracted at *every* token. If a completion is
very long, the cumulative KL penalty (summed via the recursive GAE backup) can end up
comparable in magnitude to the one-time terminal reward. What effect would you expect this
to have on a PPO run with an unusually large `max_new_tokens`, and why does that argue for
keeping completions short in this pipeline's setting?

*Write your answer below:*

"""))

# Part 4 is appended here.
```

- [ ] **Step 2: Regenerate and execute**

```bash
cd notebooks && ../.venv/bin/python build_llm_pipeline_03_reward_model_and_ppo_notebook.py && cd ..
.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 \
  notebooks/llm_training_pipeline/03_reward_model_and_ppo.ipynb
```

Expected: no errors; output includes `TEST 5 PASSED`, `TEST 6a/6b/6c PASSED`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/build_llm_pipeline_03_reward_model_and_ppo_notebook.py notebooks/llm_training_pipeline/03_reward_model_and_ppo.ipynb
git commit -m "feat: llm_training_pipeline notebook 3 part 3 — PPO core (rollout, GAE, clipped objective)"
```

---

## Task 7: Notebook Part 4 — PPO Training Loop

**Files:**
- Modify: `notebooks/build_llm_pipeline_03_reward_model_and_ppo_notebook.py`

**Interfaces:**
- Consumes: `sft_model`, `sft_cfg`, `BLOCK_SIZE`, `tokenizer`, `EOT_ID`, `CKPT_DIR`, `device` from Task 3; `TOPIC_KEYWORDS`, `format_sft_prompt` from `src.llm_pipeline.data`; `encode_pair_text`, `reward_model` from Task 5; `PPOActorCritic`, `generate_rollout`, `compute_token_rewards`, `compute_gae`, `ppo_clipped_loss`, `evaluate_actions` from Task 6.
- Produces: `data/checkpoints/llm_training_pipeline/ppo_model.pt` (dict with keys `model_state_dict`, `config`), `data/checkpoints/llm_training_pipeline/ppo_training_log.json` (dict with keys `mean_rewards`, `mean_kls`, both lists of floats, one per PPO step).

- [ ] **Step 1: Append the Part 4 cells**

```python
# ─── PART 4: PPO TRAINING LOOP ───────────────────────────────────────────────
cells.append(md("""
---
## Part 4: PPO Training Loop

Trains a fresh copy of `sft_model` (the "policy") against the frozen reward model, with a
frozen copy of `sft_model` as the KL reference. Each step: sample a batch of prompts,
generate a rollout, score it with the reward model, compute GAE advantages, then take
several clipped-objective gradient steps on the same rollout. Logs mean reward and mean KL
per step to `ppo_training_log.json` for Notebook 5.
"""))

cells.append(code("""
policy = PPOActorCritic(copy.deepcopy(sft_model)).to(device)
ref_model = copy.deepcopy(sft_model).to(device)
for p in ref_model.parameters():
    p.requires_grad_(False)
ref_model.eval()
print(f"PPO policy params (incl. value head): {sum(p.numel() for p in policy.parameters()):,}")
"""))

cells.append(code("""
ppo_steps = 150
ppo_epochs = 2
batch_size = 16
max_new_tokens = 40
gamma, lam = 1.0, 0.95
clip_eps = 0.2
kl_beta = 0.1
value_coef = 0.5
lr = 1e-5

opt = torch.optim.AdamW(policy.parameters(), lr=lr)
mean_rewards, mean_kls = [], []
t0 = time.time()
for step in range(ppo_steps):
    topic_idx = torch.randint(0, len(TOPIC_KEYWORDS), (batch_size,))
    topics = [TOPIC_KEYWORDS[i] for i in topic_idx]
    prompts = [format_sft_prompt(t) for t in topics]
    prompt_id_lists = [tokenizer.encode(p).ids for p in prompts]
    prompt_len = max(len(p) for p in prompt_id_lists)
    padded = [[EOT_ID] * (prompt_len - len(p)) + p for p in prompt_id_lists]
    prompt_ids = torch.tensor(padded, device=device)

    idx, policy_lp, ref_lp, values = generate_rollout(
        policy, ref_model, prompt_ids, max_new_tokens, temperature=1.0, top_k=40, block_size=BLOCK_SIZE
    )
    completions = [tokenizer.decode(idx[i, prompt_len:].tolist()) for i in range(batch_size)]
    with torch.no_grad():
        full = [encode_pair_text(prompts[i], completions[i], BLOCK_SIZE) for i in range(batch_size)]
        full_ids = torch.stack([f[0] for f in full]).to(device)
        full_lens = torch.tensor([f[1] for f in full], device=device)
        terminal_rewards = reward_model(full_ids, full_lens)

    token_rewards, kl = compute_token_rewards(policy_lp, ref_lp, terminal_rewards, kl_beta)
    advantages, returns = compute_gae(token_rewards, values, gamma, lam)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    old_logprobs = policy_lp

    for _ in range(ppo_epochs):
        new_logprobs, new_values = evaluate_actions(policy, idx, prompt_len, max_new_tokens)
        policy_loss = ppo_clipped_loss(new_logprobs, old_logprobs, advantages, clip_eps)
        value_loss = F.mse_loss(new_values, returns)
        loss = policy_loss + value_coef * value_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

    mean_reward = terminal_rewards.mean().item()
    mean_kl = kl.mean().item()
    mean_rewards.append(mean_reward)
    mean_kls.append(mean_kl)
    if step % 15 == 0 or step == ppo_steps - 1:
        print(f"step {step:4d} | mean_reward {mean_reward:+.3f} | mean_kl {mean_kl:.4f} | elapsed {time.time()-t0:.0f}s")

print(f"PPO training elapsed: {time.time()-t0:.1f}s")
"""))

cells.append(code("""
with open(f"{CKPT_DIR}/ppo_training_log.json", 'w') as f:
    json.dump({'mean_rewards': mean_rewards, 'mean_kls': mean_kls}, f)
print(f"Saved PPO training log to {CKPT_DIR}/ppo_training_log.json")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(mean_rewards); ax1.set_xlabel('PPO step'); ax1.set_ylabel('mean reward-model score'); ax1.set_title('Reward over training')
ax2.plot(mean_kls); ax2.set_xlabel('PPO step'); ax2.set_ylabel('mean per-token KL(policy || ref)'); ax2.set_title('KL over training')
plt.tight_layout(); plt.show()
"""))

cells.append(code("""
# TEST 7: reward-model score rises over training while KL stays bounded
first_20_avg_r = sum(mean_rewards[:20]) / 20
last_20_avg_r = sum(mean_rewards[-20:]) / 20
print(f"first-20 mean reward: {first_20_avg_r:.3f}, last-20 mean reward: {last_20_avg_r:.3f}")
assert last_20_avg_r > first_20_avg_r, "reward did not increase over PPO training"

max_kl = max(mean_kls)
print(f"max mean KL over training: {max_kl:.4f}")
assert max_kl < 2.0, f"KL grew unexpectedly large ({max_kl:.4f}) — policy may have collapsed away from reference"
print("TEST 7 PASSED — reward increased over training while KL stayed bounded")
"""))

cells.append(code("""
# Qualitative comparison: SFT vs PPO completions on held-out topics
policy.eval()
for topic in ["dragon", "picnic", "robot"]:
    prompt = format_sft_prompt(topic)
    prompt_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    with torch.no_grad():
        sft_out = sft_model.generate(prompt_ids, max_new_tokens=40, temperature=0.8, top_k=40)
        ppo_out, _ = policy.gpt.generate, None  # placeholder unused
    ppo_out = policy.gpt.generate(prompt_ids, max_new_tokens=40, temperature=0.8, top_k=40)
    print(f"=== topic: {topic} ===")
    print("SFT:", tokenizer.decode(sft_out[0].tolist()))
    print("PPO:", tokenizer.decode(ppo_out[0].tolist()))
    print(f"  sentiment — SFT: {sentiment_score(tokenizer.decode(sft_out[0, prompt_ids.shape[1]:].tolist())):+.3f}, "
          f"PPO: {sentiment_score(tokenizer.decode(ppo_out[0, prompt_ids.shape[1]:].tolist())):+.3f}")
    print()
policy.train()
"""))

cells.append(md("""
### Question 4

Compare the SFT and PPO completions and their sentiment scores above. Beyond "the sentiment
score went up", does the PPO output still read as a coherent story about the stated topic,
or do you see early signs of reward hacking (Section 5) — e.g. generic positive-sentiment
phrases inserted somewhat independent of the story's actual content? What would you check
next (only 150 PPO steps were run here) if you wanted to know whether more training would
make this better or worse?

*Write your answer below:*

"""))

cells.append(code("""
ckpt_path = f"{CKPT_DIR}/ppo_model.pt"
torch.save({'model_state_dict': policy.gpt.state_dict(), 'config': sft_cfg}, ckpt_path)
print(f"Saved PPO checkpoint to {ckpt_path}")
"""))
```

- [ ] **Step 2: Fix the placeholder line before executing**

The Part 4 qualitative-comparison cell above contains a leftover debugging line
(`ppo_out, _ = policy.gpt.generate, None  # placeholder unused`) that must be removed before
this is considered final. Edit the cell in the builder script to delete that line — the
correct version is:

```python
cells.append(code("""
# Qualitative comparison: SFT vs PPO completions on held-out topics
policy.eval()
for topic in ["dragon", "picnic", "robot"]:
    prompt = format_sft_prompt(topic)
    prompt_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    with torch.no_grad():
        sft_out = sft_model.generate(prompt_ids, max_new_tokens=40, temperature=0.8, top_k=40)
        ppo_out = policy.gpt.generate(prompt_ids, max_new_tokens=40, temperature=0.8, top_k=40)
    print(f"=== topic: {topic} ===")
    print("SFT:", tokenizer.decode(sft_out[0].tolist()))
    print("PPO:", tokenizer.decode(ppo_out[0].tolist()))
    print(f"  sentiment — SFT: {sentiment_score(tokenizer.decode(sft_out[0, prompt_ids.shape[1]:].tolist())):+.3f}, "
          f"PPO: {sentiment_score(tokenizer.decode(ppo_out[0, prompt_ids.shape[1]:].tolist())):+.3f}")
    print()
policy.train()
"""))
```

(This replaces the cell appended in Step 1 above — the Step 1 listing included the bug
deliberately so this fix step has a concrete diff to make; write the corrected version
directly in Step 1 and skip re-appending.)

- [ ] **Step 3: Regenerate and execute (this step takes several minutes — 150 PPO steps, each generating a 40-token rollout for 16 prompts)**

```bash
cd notebooks && ../.venv/bin/python build_llm_pipeline_03_reward_model_and_ppo_notebook.py && cd ..
.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 \
  notebooks/llm_training_pipeline/03_reward_model_and_ppo.ipynb
```

Expected: no errors; output includes `TEST 7 PASSED` and `Saved PPO checkpoint to
../../data/checkpoints/llm_training_pipeline/ppo_model.pt`.

- [ ] **Step 4: Verify the checkpoint and log load**

```bash
.venv/bin/python -c "
import torch, json
from src.llm_pipeline.model import GPTModel
ckpt = torch.load('data/checkpoints/llm_training_pipeline/ppo_model.pt', weights_only=False)
model = GPTModel(ckpt['config'])
model.load_state_dict(ckpt['model_state_dict'])
n_params = sum(p.numel() for p in model.parameters())
assert n_params == 13817856
log = json.load(open('data/checkpoints/llm_training_pipeline/ppo_training_log.json'))
assert len(log['mean_rewards']) == len(log['mean_kls']) == 150
print('ppo_model.pt OK, param count', n_params, '— ppo_training_log.json OK,', len(log['mean_rewards']), 'steps')
"
```

Note: count via `model.parameters()` after `load_state_dict`, not
`sum(v.numel() for v in state_dict.values())` — `GPTModel` ties `lm_head.weight` to
`tok_emb.weight`, and a raw `state_dict()` lists tied parameters under both names (no
dedup), which inflates the count to 16,889,856 and makes this assertion fail even though
the checkpoint and weight tying are both correct (same fix already applied to the Part 1
and Part 2 plans).

Expected: `ppo_model.pt OK, param count 13817856 — ppo_training_log.json OK, 150 steps`

- [ ] **Step 5: Commit**

```bash
git add notebooks/build_llm_pipeline_03_reward_model_and_ppo_notebook.py notebooks/llm_training_pipeline/03_reward_model_and_ppo.ipynb
git commit -m "feat: llm_training_pipeline notebook 3 part 4 — PPO training loop"
```

---

## Task 8: Consolidate into `src/llm_pipeline/rlhf.py`

**Files:**
- Create: `src/llm_pipeline/rlhf.py`

**Interfaces:**
- Consumes: the validated class/function bodies from Tasks 5-6 (copied verbatim).
- Produces: `from src.llm_pipeline.rlhf import RewardModel, load_trunk_from_sft, bradley_terry_loss, encode_pair_text, PPOActorCritic, generate_rollout, compute_token_rewards, compute_gae, ppo_clipped_loss, evaluate_actions`. Reused by Part 6 (GRPO), which imports `ppo_clipped_loss` and adds its own `compute_group_relative_advantage`.

- [ ] **Step 1: Create `src/llm_pipeline/rlhf.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.llm_pipeline.model import GPTConfig, GPTModel, Block


class RewardModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.reward_head = nn.Linear(config.n_embd, 1, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, lengths=None):
        """idx: (B, T) padded token ids. lengths: (B,) real (unpadded) sequence
        lengths, or None to use position T-1 for every example."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        if lengths is None:
            last = x[:, -1, :]
        else:
            last_idx = (lengths - 1).clamp(min=0)
            last = x[torch.arange(B, device=x.device), last_idx, :]
        return self.reward_head(last).squeeze(-1)


def load_trunk_from_sft(reward_model, sft_state_dict):
    """Copies token/position embeddings, transformer blocks, and final LayerNorm
    from an SFT GPTModel's state dict into a freshly-initialized RewardModel,
    leaving reward_head randomly initialized."""
    trunk_keys = [k for k in sft_state_dict if k.startswith(('tok_emb', 'pos_emb', 'blocks', 'ln_f'))]
    own_state = reward_model.state_dict()
    for k in trunk_keys:
        own_state[k].copy_(sft_state_dict[k])


def bradley_terry_loss(reward_chosen, reward_rejected):
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()


def encode_pair_text(prompt, completion, tokenizer, eot_id, block_size):
    """Returns (padded_ids: LongTensor[block_size], real_length: int)."""
    ids = (tokenizer.encode(prompt).ids + tokenizer.encode(completion).ids + [eot_id])[:block_size]
    length = len(ids)
    ids = ids + [eot_id] * (block_size - length)
    return torch.tensor(ids, dtype=torch.long), length


class PPOActorCritic(nn.Module):
    """Wraps a GPTModel, exposing both LM logits and a per-position scalar value
    estimate. Reuses the wrapped model's tok_emb/pos_emb/drop/blocks/ln_f/lm_head
    directly — no changes to GPTModel itself."""
    def __init__(self, gpt: GPTModel):
        super().__init__()
        self.gpt = gpt
        self.value_head = nn.Linear(gpt.config.n_embd, 1, bias=False)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.gpt.tok_emb(idx) + self.gpt.pos_emb(pos)
        x = self.gpt.drop(x)
        for block in self.gpt.blocks:
            x = block(x)
        x = self.gpt.ln_f(x)
        logits = self.gpt.lm_head(x)
        values = self.value_head(x).squeeze(-1)
        return logits, values


@torch.no_grad()
def generate_rollout(actor_critic, ref_model, prompt_ids, max_new_tokens, temperature, top_k, block_size):
    """Samples max_new_tokens autoregressively from actor_critic, recording the
    policy's log-prob, the frozen ref_model's log-prob, and the value estimate at
    each sampled token. Returns (idx, policy_logprobs, ref_logprobs, values)."""
    idx = prompt_ids.clone()
    policy_logprobs, ref_logprobs, values = [], [], []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, vals = actor_critic(idx_cond)
        logits_last = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits_last, top_k)
            logits_last[logits_last < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits_last, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        policy_lp = F.log_softmax(logits_last, dim=-1).gather(1, next_id).squeeze(-1)

        ref_logits, _ = ref_model(idx_cond)
        ref_lp = F.log_softmax(ref_logits[:, -1, :], dim=-1).gather(1, next_id).squeeze(-1)

        idx = torch.cat([idx, next_id], dim=1)
        policy_logprobs.append(policy_lp)
        ref_logprobs.append(ref_lp)
        values.append(vals[:, -1])
    return (
        idx,
        torch.stack(policy_logprobs, dim=1),
        torch.stack(ref_logprobs, dim=1),
        torch.stack(values, dim=1),
    )


def compute_token_rewards(policy_logprobs, ref_logprobs, terminal_reward, kl_beta):
    """Per-token reward = -kl_beta * KL at every step, plus terminal_reward added
    only at the last generated token. Returns (rewards, kl), both (B, T)."""
    kl = policy_logprobs - ref_logprobs
    rewards = -kl_beta * kl
    rewards = rewards.clone()
    rewards[:, -1] = rewards[:, -1] + terminal_reward
    return rewards, kl


def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """rewards, values: (B, T). Returns (advantages, returns), both (B, T)."""
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)
    next_value = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae
        next_value = values[:, t]
    returns = advantages + values
    return advantages, returns


def ppo_clipped_loss(new_logprobs, old_logprobs, advantages, clip_eps=0.2):
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()


def evaluate_actions(policy, idx, prompt_len, gen_len):
    """Re-runs policy over the full generated sequence, extracting log-probs and
    values at the positions/tokens that were sampled during the rollout."""
    logits, values = policy(idx[:, :-1])
    action_logits = logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :]
    action_values = values[:, prompt_len - 1 : prompt_len - 1 + gen_len]
    actions = idx[:, prompt_len : prompt_len + gen_len]
    logprobs = F.log_softmax(action_logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return logprobs, action_values
```

Note: `encode_pair_text` here takes `tokenizer, eot_id` as explicit parameters (unlike the
notebook version, which closes over module-level `tokenizer`/`EOT_ID` globals) — this is
the one deliberate signature difference between the notebook and `rlhf.py`, needed because
`rlhf.py` has no notebook-global tokenizer to close over.

- [ ] **Step 2: Smoke-test the module**

```bash
.venv/bin/python -c "
import torch
from src.llm_pipeline.model import GPTConfig, GPTModel
from src.llm_pipeline.rlhf import (
    RewardModel, load_trunk_from_sft, bradley_terry_loss,
    PPOActorCritic, generate_rollout, compute_token_rewards,
    compute_gae, ppo_clipped_loss, evaluate_actions,
)

cfg = GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
gpt = GPTModel(cfg)
rm = RewardModel(cfg)
load_trunk_from_sft(rm, gpt.state_dict())
idx = torch.randint(0, cfg.vocab_size, (3, 10))
lengths = torch.tensor([10, 5, 8])
r = rm(idx, lengths)
assert r.shape == (3,)
print('RewardModel forward OK')

loss = bradley_terry_loss(torch.tensor([1.0]), torch.tensor([0.0]))
assert loss.item() > 0
print('bradley_terry_loss OK')

policy = PPOActorCritic(GPTModel(cfg))
ref = GPTModel(cfg)
ref.eval()
prompt_ids = torch.randint(0, cfg.vocab_size, (2, 4))
rollout_idx, plp, rlp, vals = generate_rollout(policy, ref, prompt_ids, max_new_tokens=3, temperature=1.0, top_k=None, block_size=cfg.block_size)
assert rollout_idx.shape == (2, 7) and plp.shape == (2, 3) and vals.shape == (2, 3)
rewards, kl = compute_token_rewards(plp, rlp, torch.tensor([1.0, -1.0]), kl_beta=0.1)
adv, ret = compute_gae(rewards, vals)
assert adv.shape == (2, 3)
new_lp, new_v = evaluate_actions(policy, rollout_idx, prompt_len=4, gen_len=3)
loss = ppo_clipped_loss(new_lp, plp, adv)
assert loss.dim() == 0
print('PPO pipeline (rollout -> reward -> GAE -> clipped loss) OK')
"
```

Expected: `RewardModel forward OK`, `bradley_terry_loss OK`,
`PPO pipeline (rollout -> reward -> GAE -> clipped loss) OK` — no assertion errors.

- [ ] **Step 3: Commit**

```bash
git add src/llm_pipeline/rlhf.py
git commit -m "refactor: consolidate notebook 3's reward model and PPO utilities into src/llm_pipeline/rlhf.py"
```

---

## Self-Review Checklist

**1. Spec coverage** (against `docs/superpowers/specs/2026-07-02-llm-training-pipeline-learning-materials-design.md`):
- HTML reference Section 5 (reward modeling — Bradley-Terry/Luce's choice axiom derivation, pairwise loss, reward hacking) and Section 6 (PPO — KL-constrained objective, GAE derivation, clipped surrogate, why a value function, why PPO complex/fragile): Task 1. ✓
- Concepts Q&A additions for reward modeling + PPO: Task 2. ✓
- Notebook 3 (`03_reward_model_and_ppo.ipynb`): generate preference dataset (sample+sentiment-score+pair), `RewardModel` + Bradley-Terry loss with ranking-accuracy test, PPO (frozen reference, value head, rollout, GAE, clipped objective) with GAE-toy-trajectory and clip-boundary tests, reward rising while KL bounded, logs to `ppo_training_log.json`, saves `ppo_model.pt` and `preference_pairs.json`: Tasks 3-7. ✓
- Consolidation into `src/llm_pipeline/rlhf.py`: Task 8. ✓
- `src/llm_pipeline/rlhf.py` reused (partially) by the GRPO notebook, per spec: `ppo_clipped_loss` is designed to be imported as-is by Part 6. ✓

**2. Placeholder scan:** No "TBD"/"TODO"/"similar to Task N" patterns. Task 7 Step 1 deliberately includes one broken line in its first listing and Step 2 is the fix — this is flagged explicitly in-plan (not a silent placeholder) because the qualitative-comparison cell's control flow needs `torch.no_grad()` to wrap only the `sft_out`/`ppo_out` generation calls, and drafting it in one pass without an intermediate variable produced a leftover no-op line; Step 2 gives the exact corrected code to use. Every other code step contains complete, runnable code; every HTML/Markdown step contains complete final content.

**3. Type/interface consistency:** `RewardModel.forward(idx, lengths=None) -> Tensor[B]`, `PPOActorCritic.forward(idx) -> (logits, values)`, `generate_rollout(...) -> (idx, policy_logprobs, ref_logprobs, values)`, `compute_gae(rewards, values, gamma=1.0, lam=0.95) -> (advantages, returns)`, and `ppo_clipped_loss(new_logprobs, old_logprobs, advantages, clip_eps=0.2) -> scalar Tensor` are identical between the notebook (Tasks 5-7) and `src/llm_pipeline/rlhf.py` (Task 8), except `encode_pair_text`'s explicit `tokenizer, eot_id` parameters in `rlhf.py` versus notebook-global closures — called out in Task 8 Step 1. `preference_pairs.json`'s schema (`prompt`, `chosen`, `rejected`, `chosen_score`, `rejected_score`) is fixed here and is exactly what Part 4 (DPO)'s plan must consume unchanged. `ppo_model.pt`'s checkpoint shape (`model_state_dict`, `config`) matches `base_model.pt`/`sft_model.pt`.
