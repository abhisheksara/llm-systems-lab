# LLM Training Pipeline — Concept Q&A

Study notes consolidating the conceptual discussion. Companion to
`docs/llm_training_pipeline_reference.html` and
`notebooks/llm_training_pipeline/`.

---

## 1. Why decoder-only, not encoder-only or encoder-decoder, for a general-purpose LLM?

Three architecture families, three objectives:

| Family | Attention | Objective | Strength | Weakness for chat/generation |
|---|---|---|---|---|
| Encoder-only (BERT) | Bidirectional | Masked LM | Best per-token representations for understanding tasks | No causal structure — can't sample token-by-token |
| Encoder-decoder (T5) | Bidirectional encoder + causal decoder w/ cross-attention | Seq2seq (span corruption, translation, etc.) | Strong for well-defined input→output tasks | Doubles the stack; needs an explicit input/output split |
| Decoder-only (GPT) | Causal | Next-token prediction | One stack, one objective, handles any task framed as text continuation | Can't look ahead within a single forward pass (not usually a real limitation for generation) |

The decisive property for chat models: a decoder-only model treats "prompt" and
"response" as one continuous token stream, distinguished only by **loss
masking** (see the SFT stage). This is exactly what lets the same architecture be
pretrained on raw text, then SFT'd on instructions, then aligned via PPO/DPO,
with zero architecture changes between stages — only the data and loss mask
change. This uniformity is why decoder-only became the dominant choice for
general-purpose assistants, even though encoder-decoder models can be more
parameter-efficient for narrowly-scoped seq2seq tasks.

---

## 2. Full derivation: why scale attention logits by `1/sqrt(d_k)`?

Assume, at initialization, that each component of `q` and `k` is drawn i.i.d.
with mean 0 and variance 1 (roughly true right after the `QW_Q`, `XW_K`
projections with standard-initialized weights). The raw dot product is:

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

Each term `q_i k_i` is a product of two independent zero-mean, unit-variance
variables, so `E[q_i k_i] = 0` and `Var(q_i k_i) = E[q_i^2]E[k_i^2] = 1`. Since
the `d_k` terms are independent, variances add:

$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

So the standard deviation of the raw dot product grows as `sqrt(d_k)`. Feed
that into softmax: softmax is invariant to adding a constant to every logit,
but *not* to scaling the spread of the logits — a wider spread pushes the max
logit's softmax output toward 1 and everything else toward 0 (softmax
saturates), and the local gradient of softmax at saturation is tiny (the
Jacobian `diag(p) - pp^T` vanishes as `p` approaches a one-hot vector). Dividing
by `sqrt(d_k)` renormalizes the dot product back to unit variance regardless
of head dimension, keeping the softmax input distribution — and its gradient
— well-conditioned independent of how wide each attention head is.

**Extra empirical detail (GPT-2):** GPT-2 additionally scales the residual
projection weights (the output projection of attention, and the second linear
layer of the MLP) by `1/sqrt(2 * n_layer)` at initialization, to compensate
for the fact that activations accumulate variance additively down `n_layer`
residual branches. This pipeline's model does not add this refinement — plain
`std=0.02` init is enough at 6 layers / ~14M params — but it is the standard
next step if training becomes unstable at greater depth.

---

## 3. Pre-norm vs. post-norm — what actually breaks in deep post-norm transformers?

The original Transformer (Vaswani et al. 2017) is **post-norm**:
`x <- LayerNorm(x + SubLayer(x))`. Xiong et al. 2020 ("On Layer Normalization
in the Transformer Architecture") show that in post-norm, the expected
gradient magnitude at the *input* layers grows with depth unless a careful
learning-rate warmup schedule is used — layer normalization sits *after* the
residual addition, so it keeps rescaling the accumulated signal at every
layer, and this repeated rescaling distorts backpropagated gradients as depth
increases. Without warmup, post-norm training is either unstable (LR too
high for early steps) or very slow to converge (LR low enough to be safe
throughout).

**Pre-norm** (`x <- x + SubLayer(LayerNorm(x))`, used here and in GPT-2
onward) keeps an un-normalized identity path from input straight to output —
LayerNorm only ever touches the sub-layer's *input*, never the residual
stream itself. Xiong et al. show this gives well-behaved gradients at
initialization *without* warmup, which is why pre-norm became the default as
transformer stacks got deeper (dozens to hundreds of layers in production
LLMs, versus post-norm's practical ceiling around 20-30 layers without
additional tricks).

---

## 4. Teacher forcing makes training parallel — what does it hide? (exposure bias)

**Why training is parallel:** the loss at every position `t` is computed
against the *ground-truth* previous tokens `x_{<t}`, not the model's own
predictions. Because of this, and because attention respects the causal mask,
one forward pass over a full sequence computes the loss at all `T` positions
simultaneously — no autoregressive loop is needed at training time.

**What this hides:** at inference, the model conditions on its *own*
previously-generated tokens, not ground truth. If the model produces an
early mistake, every subsequent token is generated conditioned on a prefix
the model never saw an equivalent of during training (training prefixes are
always "correct" by construction). This train/inference mismatch is called
**exposure bias** (Bengio et al. 2015, "Scheduled Sampling"). It is a known,
not-fully-solved issue — in practice it is mitigated by scale (bigger models
make fewer early mistakes to compound) and by sampling strategy (greedy
decoding compounds errors more than a well-tuned temperature/top-p), rather
than by a clean architectural fix. Worth knowing as a named concept, not a
solved problem.

---

## 5. Weight tying — precedent and when it stops helping

Press & Wolf, 2017 ("Using the Output Embedding to Improve Language
Models") show that sharing the input embedding matrix and the output
(LM head) projection — under the interpretation that both are ultimately
representing "how token `i` relates to a `d`-dimensional semantic space" —
reduces parameter count and acts as a regularizer, with no loss reduction
(often a small improvement) in the language-modeling setting. GPT-2 and most
subsequent decoder-only LMs adopt it by default.

At this pipeline's scale (~8k vocab, `d_model=384`), the tied matrix is
`8000 x 384 ≈ 3.1M` params — roughly a fifth of the model's ~14M total, so
tying is a meaningful savings, not a rounding error. The benefit is largest
exactly in this regime (embedding parameters a large fraction of the total).
At the opposite extreme — very large vocabularies (e.g. 250k+ multilingual
tokenizers) paired with very large `d_model` — some production models untie
input and output embeddings, on the reasoning that the input embedding wants
to cluster tokens by *distributional similarity* while the output projection
wants to separate them for a sharp next-token decision, and at large enough
scale the parameter savings from tying stop being the binding constraint.

---

## 6. Next-token prediction as maximum likelihood — and as compression

Minimizing $-\sum_t \log P_\theta(x_t \mid x_{<t})$ over a corpus is exactly
maximum-likelihood estimation of $\theta$ under the autoregressive
factorization of the corpus's joint distribution. There is a second, useful
reading: an arithmetic coder driven by a probability model $P_\theta$ encodes
a sequence in a number of bits equal to (up to rounding) its negative
log-likelihood under that model. So **minimizing cross-entropy loss is
identical to minimizing the compressed size of the training corpus under an
arithmetic coder parameterized by the model** — "predict the next token well"
and "compress the corpus well" are the same objective (Delétang et al. 2023,
"Language Modeling Is Compression", make this explicit and use LLM
compression ratios as a capability benchmark). This is why loss-in-nats
converts directly to bits-per-byte / bits-per-character figures reported for
frontier models — it's a compression ratio, not an arbitrary training metric.

---

## 7. Chinchilla / compute-optimal scaling — the actual numbers

Hoffmann et al. 2022 ("Training Compute-Optimal Large Language Models")
trained ~400 models across a range of sizes and token counts and fit scaling
laws for loss as a joint function of parameter count `N` and training tokens
`D` given a fixed compute budget `C ≈ 6ND` (FLOPs). Their headline finding:
prior large models (GPT-3 175B on 300B tokens, Gopher 280B on 300B tokens)
were **undertrained** relative to their size for the compute spent —
compute-optimal training scales `N` and `D` at roughly the same rate as `C`
grows (`N_opt ∝ C^0.5`, `D_opt ∝ C^0.5`), which works out to a commonly cited
rule of thumb of **~20 training tokens per parameter**. Their Chinchilla model
(70B params, 1.4T tokens, same compute budget as Gopher) outperformed the 4x
larger Gopher. Practical implication: past a point, spending more compute on
a bigger model with the same data is a worse trade than training a smaller
model longer on more data — this is why post-2022 open models (LLaMA family
etc.) train comparatively "small" models on very large token counts.

---

## 8. Why does data curation matter as much as scale? (concept-only)

Raw web-scraped corpora (Common Crawl derivatives) contain heavy near-duplicate
content, boilerplate, and low-quality text. Empirically (e.g. the
RefinedWeb / FineWeb line of work), aggressive deduplication and
quality-filtering of pretraining data measurably improves downstream
benchmark performance *at fixed compute and fixed token count* — the model
sees more unique signal per token processed, and heavy duplication has been
directly linked to increased verbatim memorization (a privacy/IP concern
independent of capability). This pipeline trains on TinyStories, which is
already synthetically curated (GPT-4-generated, restricted vocabulary,
consistent short narrative structure) specifically so a small model can fit
its distribution well — no additional curation step is implemented here, but
at production scale, data curation is frequently a bigger lever on final
model quality than architectural changes.

---

## 9. Why mask the prompt tokens in the SFT loss — what breaks if you don't?

Concretely: if a `(prompt, response)` pair has roughly as many prompt tokens
as response tokens (common for short instructions + short responses), an
*unmasked* loss spends roughly half its gradient mass teaching the model to
reproduce a fixed, deterministic template string. That template is drawn from
a tiny, structured space (here, `"Write a short story about {topic}:\n"` for
~40 topic words) — a model can drive its loss on those positions to near
zero almost immediately, after which further gradient on them is close to
noise, but it is noise computed and backpropagated through the *same* shared
weights used for the response tokens, diluting the signal that actually
teaches "produce a good response to this prompt."

There's a second, sharper problem: at inference time the prompt is **given**,
never generated — the user supplies it. Training the model to also predict
it optimizes a capability (predicting the next prompt token) that the
deployed system never exercises. Masking the prompt with `ignore_index=-100`
removes both problems: gradient flows only through the tokens the model must
actually learn to produce.

---

## 10. Catastrophic forgetting and the low-LR rationale — precedent

**Catastrophic forgetting** (McCloskey & Cohen, 1989, in the connectionist
networks literature; French, 1999, "Catastrophic Forgetting in Connectionist
Networks" surveys it) is the tendency of a neural network trained
sequentially on task B after task A to lose performance on task A, because
gradient descent on B's narrower data distribution is free to overwrite
weight directions that were only useful for A. SFT is exactly this setup:
"task A" is the broad next-token competence pretraining built from a huge,
diverse corpus; "task B" is a narrow, templated instruction-following
distribution over a fraction of that data's diversity.

Two standard mitigations, in order of what this pipeline uses:

1. **Low LR + few steps** (used here: `3e-5`, roughly 10-100x below
   pretraining's `3e-4` peak LR, and a comparatively small step count). This
   keeps SFT's gradient updates small enough to shift the output
   *distribution* toward instruction-following without individual steps
   large enough to erase pretrained weight structure. It is the simplest
   mitigation and requires no architecture changes — which is why it is the
   default starting point for any full-parameter fine-tune.
2. **Parameter-efficient fine-tuning** (LoRA — Hu et al. 2021, "LoRA:
   Low-Rank Adaptation of Large Language Models" — adapters, etc.): freeze
   the pretrained weights entirely and train only a small number of new
   parameters (e.g. low-rank update matrices) inserted alongside them.
   Because the original weights are literally untouched, catastrophic
   forgetting of the base capability is structurally impossible, at the cost
   of a smaller effective capacity for the new task. Not used in this
   pipeline (a full from-scratch, full-parameter fine-tune is the point of
   the exercise), but the standard production-scale answer when serving many
   fine-tunes off one base model cheaply matters.

---

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
