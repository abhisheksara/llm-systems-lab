# Positional Embeddings & RoPE — Concept Q&A

Study notes consolidating the conceptual discussion. Companion to
`docs/positional_embeddings_reference.html` and
`notebooks/positional_embeddings_tutorial.ipynb`.

---

## 1. What is the `10000` in sinusoidal encoding?

The formula:

$$PE(pos, 2i) = \sin\!\left(\frac{pos}{\mathbf{10000}^{2i/d}}\right), \qquad
  PE(pos, 2i+1) = \cos\!\left(\frac{pos}{\mathbf{10000}^{2i/d}}\right)$$

`10000` is the **`base`** — the single free hyperparameter. It sets the **range of
wavelengths**. As the dimension index $i$ runs from $0$ to $d/2-1$, the wavelength sweeps a
geometric progression:

| dim pair $i$ | wavelength | role |
|---|---|---|
| $0$ | $2\pi$ (≈6 positions) | fastest — distinguishes adjacent tokens |
| $d/2-1$ | $2\pi \cdot 10000$ (≈62,800 positions) | slowest — coarse long-range position |

Intuition: `base` ≈ "the longest distance I can encode before the slowest dimension wraps
around and repeats." Choose it large enough that the lowest-frequency sinusoid does **not**
complete a full cycle within your sequences, so every position gets a **unique** code.

It is a **heuristic constant, not derived** — picked big enough for expected sequence
lengths. Strongest evidence it's just a tunable knob: **NTK-aware / YaRN context extension
works by changing this base**. Bigger base → longer reach, coarser fine-grained resolution.

---

## 2. Why sinusoidal? Precedent? Naive alternatives?

### Precedent
The prior art was **learned positional embeddings** (a trainable vector per position), used
in ConvS2S (Gehring et al. 2017). The Transformer authors tried that too and found it gave
**nearly identical results**. They chose sinusoidal for one hypothesized reason
(extrapolation — see §3). Deeper precedent: this is **Fourier features**, an old
signal-processing idea (represent a scalar with sinusoids of many frequencies).

### Naive alternatives and their failure modes

| Strategy | Problem |
|---|---|
| **Raw integer** $pos = 0,1,2,\dots$ | Unbounded magnitude — position 5000 dwarfs the embedding values, destabilizes training. No relative structure in the dot product. |
| **Normalized** $pos/N$ | Bounded $[0,1]$ but **length-dependent**: "0.5" = position 5 in a length-10 sequence, 50 in length-100. Same position ≠ same code. One scalar = tiny capacity. |
| **Learned table** (vector per position) | Works in-distribution but **hard length ceiling**, zero extrapolation, $O(L \cdot d)$ params, every position learned independently. |
| **Binary encoding** of $pos$ across bits | Good intuition — multi-resolution, unique codes — but **discrete/jagged**: non-differentiable jumps, abrupt boundaries (0111 → 1000). |

### Key insight: sinusoidal = "continuous binary"

A binary counter has bit 0 flip every position, bit 1 every 2, bit 2 every 4… — frequencies
in geometric progression. Sinusoidal PE replaces each discrete **bit** with a smooth
**sinusoid** whose frequency drops geometrically (that's what `10000` controls). You keep
what's good about binary and fix what's bad:

- **Bounded** values in $[-1, 1]$ → same scale as embeddings, stable to add
- **Multi-resolution** → fast dims separate nearby tokens, slow dims separate far ones
- **Defined for any position** → no ceiling
- **Smooth** → nearby positions get nearby codes (differentiable)
- **Linear-shift property** → $PE(pos+k)$ is a rotation of $PE(pos)$ → relative attention is
  learnable (see §5–§6)
- **Zero parameters**

It's the unique choice that is simultaneously bounded, length-independent, multi-resolution,
smooth, and relative-position-friendly.

---

## 3. The authors said sinusoidal "may extrapolate to longer sequences." How?

Two layers: the **mechanism they bet on**, and **why it mostly doesn't pan out**.

### The intended mechanism

**(a) The encoding is defined for every position.** $PE(5000)$ is computable even if you
trained on length 512. A learned table has no row 5000. Minimum bar for extrapolation: there
is *a* value to feed in.

**(b) Position *similarity* depends only on distance, identically everywhere.** Take the dot
product of two encodings $p$ and $p+k$:

$$PE(p)\cdot PE(p+k) = \sum_i \big[\sin\omega_i p \,\sin\omega_i(p{+}k) + \cos\omega_i p\,\cos\omega_i(p{+}k)\big] = \sum_i \cos(\omega_i k)$$

The $p$ cancels — similarity is a function of the **offset $k$ alone**, and the *same*
function at positions $(3,5)$ as at $(1003,1005)$. Equivalently $PE(p+k) = R_k\, PE(p)$ with
$R_k$ depending only on $k$. The bet: a relative-offset pattern learned at small positions
should transfer to large ones, because the relative geometry is globally identical.

### Why it largely fails in practice

(The notebook experiment shows sinusoidal collapsing to ~0.29 at $L=64$ — about as bad as
the learned table.)

- **PE is *added* to the input, then mangled by learned $W_Q, W_K$.** The clean
  distance-only property holds for *raw* encodings. After the network entangles position with
  token content through arbitrary learned matrices, nothing *enforces* generalization. The
  relative structure is *available*, not *guaranteed*.
- **Low-frequency dims take never-seen values out of range.** Within length 512 the slow
  dimensions barely move; the network only saw a narrow slice of their range. At position
  5000 those inputs are **off-distribution** → garbage. (This is the same failure mode that
  later kills even RoPE's extrapolation and motivates Position Interpolation: keep angles in
  the trained range instead of extrapolating into the unknown.)
- Higher layers can reconstruct absolute-position features, which don't transfer.

That's why the authors wrote "**may**" — a hopeful hypothesis, not a proven property. The gap
between "relative structure is *available*" and "*enforced*" is exactly what RoPE closes, and
what PI/YaRN patch when RoPE runs out of trained range.

---

## 4. What positional encoding do SOTA models use?

**RoPE, essentially everywhere.** The mechanism you derive from scratch *is* the SOTA one.

### The standard
- **Llama 2/3/4, Mistral/Mixtral, Qwen 2/2.5/3, DeepSeek V2/V3/R1, Gemma, Phi, Yi,
  Command-R, Falcon-2…** — all RoPE.
- Closed models (GPT-4-class, Claude, Gemini) don't publish details; rotary or
  rotary-variants are the near-universal assumption.

### Layered on top (long-context recipes)
1. **Crank the base $\theta$.** The `10000` gets pushed up — Llama 3 uses
   $\theta = 500{,}000$; long-context models go to $1\text{M}$–$5\text{M}+$. Cheapest context
   lever.
2. **YaRN / NTK-aware scaling** (and Llama 3.1's frequency-dependent RoPE scaling) to reach
   128k–1M context.

### Mostly deprecated
- **ALiBi** (BLOOM, MPT, early Baichuan) — largely abandoned for RoPE.
- **Learned absolute** (BERT, GPT-2 era) — gone from new decoder LLMs.

### Frontier variants worth knowing
- **Decoupled RoPE in MLA (DeepSeek V2/V3).** Multi-head Latent Attention compresses the KV
  cache, but RoPE is incompatible with naive KV compression (rotation must be applied after
  caching). Fix: split each key into a compressed no-RoPE latent part **plus** a small
  separate RoPE-carrying part. A deployed-at-scale wrinkle caused by RoPE's structure.
- **NoPE (no positional encoding).** Causal decoders can infer position from the attention
  mask alone (Kazemnejad et al. 2023); some long-context recipes interleave a few NoPE layers
  among RoPE layers to improve length generalization.
- **Partial RoPE** — apply rotary to only a fraction of each head's dims (GPT-NeoX
  `rotary_pct`).
- **CoPE (Contextual Position Encoding, Meta 2024)** — position counted by content;
  research-stage.

**Interview line:** "RoPE is the standard; long context = high RoPE base + YaRN-style
scaling; notable frontier exceptions are DeepSeek's decoupled RoPE for MLA and the NoPE
experiments." *(Knowledge current to ~early 2026; closed-model internals are inferred.)*

---

## 5. "A position shift is a fixed rotation of the encoding" — so what?

This is the difference between *relative position is theoretically present in the input* and
*relative position is something attention can actually read out with the linear machinery it
already has*.

### The payoff: relative distance becomes linearly readable

**Step 1 — the plain dot product ($M = I$).** Start with the simplest case: the raw dot
product of two position encodings (this is the positional score when
$M = W_Q^\top W_K = I$). Group the $d$ dims into $d/2$ pairs indexed by $m$; pair $m$ of
$PE(i)$ is $(\sin\omega_m i,\ \cos\omega_m i)$. The dot product is the sum over pairs:

$$PE(i)\cdot PE(j) = \sum_m \big(\sin\omega_m i\,\sin\omega_m j + \cos\omega_m i\,\cos\omega_m j\big)$$

Apply the **cosine angle-subtraction identity** $\cos(A-B) = \cos A\cos B + \sin A\sin B$
with $A = \omega_m i$, $B = \omega_m j$ — each bracket becomes $\cos(\omega_m(i-j))$:

$$PE(i)\cdot PE(j) = \sum_m \cos\big(\omega_m (i-j)\big)$$

**Pure function of $i-j$** — absolute positions cancel. That trig identity *is* the "how";
it's the coordinate version of the rotation property $R_i^\top R_j = R_{j-i}$.

**Step 2 — what the learned $M$ adds.** Real attention scores the *projected* encodings,
$PE(i)^\top M\, PE(j)$ with $M = W_Q^\top W_K$. For an **arbitrary** $M$ this is **not**
purely relative. But the model has the *capacity* to make it relative: if $M$ is
block-diagonal with each 2×2 block a rotation $R(\phi_m)$, the same algebra gives

$$PE(i)^\top M\, PE(j) = \sum_m \cos\big(\omega_m(i-j) - \phi_m\big)$$

Still a function of $i-j$ only — now with a **learnable phase $\phi_m$ per frequency**.
Choosing the phases lets a head **peak at a chosen relative offset** $k$ ("attend to whatever
is $k$ tokens back"). So the honest claim is not "every $M$ is relative," but: relative-offset
attention is *expressible* with the linear $QK$ machinery, and the model can learn an $M$ that
realizes it.

The reason all of this works: rotations compose by **adding angles**, so
$R_i^\top R_j = R_{j-i}$. Position **subtraction becomes a clean angle difference** that the
dot product computes for free.

### Step 3 — but PE is *added* to the input: the four-term reality

The clean relative term above is misleading on its own, because the query/key are not
$PE(i)$ alone. The input is $e_i = x_i + PE(i)$ (token content **plus** position), projected
by the *same* $W_Q, W_K$. So with $q_i = W_Q(x_i + PE(i))$, $k_j = W_K(x_j + PE(j))$ and
$M = W_Q^\top W_K$, the score expands into **four** terms (the Transformer-XL decomposition,
Dai et al. 2019):

$$q_i^\top k_j =
\underbrace{x_i^\top M\, x_j}_{\text{(a) content–content}} +
\underbrace{x_i^\top M\, PE(j)}_{\text{(b) content–position}} +
\underbrace{PE(i)^\top M\, x_j}_{\text{(c) position–content}} +
\underbrace{PE(i)^\top M\, PE(j)}_{\text{(d) position–position}}$$

Only **(d)** is the purely relative term derived in Steps 1–2. The rest contaminate it:

- **(a)** pure semantic matching, no position.
- **(b), (c)** depend on **absolute** position ($j$ or $i$, not $i-j$) *and* on token content
  — e.g. (b) is "does this query's content care that the key is at absolute position $j$."
  Not relative, and exactly where the off-distribution low-frequency values (§3) leak in to
  break extrapolation.

So adding PE to the embeddings means: the relative signal is **one of four**, not isolated;
the cross terms re-inject **absolute, content-entangled** position dependence; and content
and position **share one subspace**, so a single $M$ must serve all four roles and they
interfere.

### Why RoPE is the clean fix

RoPE does **not** add position to the input. It rotates $q$ and $k$ *after* projection —
$\tilde q_i = R_i W_Q x_i$, $\tilde k_j = R_j W_K x_j$ — so the score is a **single** term:

$$\tilde q_i^\top \tilde k_j = (W_Q x_i)^\top R_i^\top R_j (W_K x_j)
  = (W_Q x_i)^\top R_{\,i-j}\,(W_K x_j)$$

Content × content, **modulated** by a pure relative rotation. No cross terms, no
content/position subspace collision, no absolute leakage. Eliminating terms (b) and (c) by
construction is a big part of why RoPE behaves and extrapolates better than added sinusoids.

### Contrast: the naive encoding lacks this

Integer index $PE(i) = i$:

$$PE(i)\cdot PE(j) = i \cdot j$$

Depends on the **product**, not the difference. Recovering $i-j$ from $i\cdot j$ needs
nonlinear arithmetic — attention's bilinear score can't do it.

So "a shift is a fixed rotation" is precisely the property that makes relative-position
attention **expressible** with the existing $QK$ mechanism — and RoPE upgrades it from
*possible* (sinusoidal, added to input, may or may not be learned) to *mandatory* (rotation
welded directly into $Q$ and $K$).

---

## 6. Is it the *linearity* of the rotation that makes position recoverable?

**No — "linear" is too weak.** That's the subtle but important correction.

### Linearity alone isn't the reason

The naive integer encoding's shift is *also* linear/affine:

$$PE(p) = p \;\Rightarrow\; PE(p+k) = p + k = PE(p) + k \quad (\text{"add } k\text{"})$$

If "the shift is a linear transform" were the magic, integer encoding would work — it
doesn't. So linearity is not the operative property.

### The real reason: it's a *rotation*, and attention reads via *inner products*

The distinguishing fact is that the shift is an **orthogonal** transform (a rotation), with:

$$R_i^\top R_j = R_{j-i}$$

Feed that into the specific operation attention performs — the inner product:

$$PE(i)\cdot PE(j) = (R_i u)^\top (R_j u) = u^\top R_i^\top R_j\, u = u^\top R_{j-i}\, u
  \;=\; \text{function of }(j-i)\text{ only.}$$

The absolute positions **cancel inside the dot product** — *because* $R^\top$ inverts the
rotation, which only orthogonal transforms do. The integer encoding's inner product is
$i \cdot j$: same "linear shift," completely different dot-product behavior.

### Precise statement

> The encoding transforms by a **rotation**, and attention reads position through an **inner
> product**. Rotation is exactly the structure under which the inner product collapses to a
> function of relative distance. So relative position is recoverable *by the one operation
> attention already computes* — no extra nonlinear machinery.

Two ingredients that must match: **rotation on the encoding side**, **inner product on the
attention side**. Linearity is necessary but not sufficient; **orthogonality**
(norm-preserving rotation) is the part that makes the absolute terms vanish. That is why RoPE
is *rotary* specifically — engineered so the $QK^\top$ inner product is *forced* through
$R_{j-i}$.
