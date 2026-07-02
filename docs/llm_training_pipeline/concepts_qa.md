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

*(Sections for SFT, reward modeling, PPO, DPO, evaluation, and RLVR/GRPO are
appended here as the corresponding notebooks are built.)*
