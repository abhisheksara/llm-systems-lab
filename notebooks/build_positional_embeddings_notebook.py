"""
Generates positional_embeddings_tutorial.ipynb from cell definitions.
Run: python3 notebooks/build_positional_embeddings_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"}
}

cells = []

def md(text): return nbf.v4.new_markdown_cell(text.strip())
def code(text): return nbf.v4.new_code_cell(text.strip())

# ─── SETUP ───────────────────────────────────────────────────────────────────
cells.append(md(r"""
# Positional Embeddings & RoPE — Tutorial Notebook

Builds intuition for **why transformers need position**, derives and implements
**Rotary Position Embeddings (RoPE)** from scratch, and ends with a small experiment
you run yourself showing RoPE's **length-extrapolation** advantage.

Sources: Vaswani et al. 2017 (sinusoidal); Su et al. 2021 (RoPE / RoFormer);
Chen et al. 2023 (Position Interpolation); Peng et al. 2023 (YaRN). PDFs are in
`docs/positional_embeddings/papers/`.

**How to use this notebook:**
- Read each theory section, then fill in the `# YOUR CODE HERE` blocks.
- Test cells use `assert` — they pass silently or raise. Do not modify test cells.
- Keep the HTML reference (`docs/positional_embeddings_reference.html`) open in another tab.

**Parts:**
1. Sinusoidal positional encoding (from scratch + viz)
2. RoPE from scratch (`rope_freqs`, `apply_rope`)
3. Verifying the relative-position property
4. Cross-checking against the complex-number form
5. Frequency / wavelength visualization
6. Context extension: Position Interpolation mechanism
7. Experiment: sinusoidal vs RoPE vs learned-absolute on length extrapolation
"""))

cells.append(code(r"""
# Run first. CPU-only, no GPU needed. (Part 7 trains 3 tiny models: ~10-12 min on CPU.)
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'torch', 'matplotlib', 'numpy'])
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)
print("torch", torch.__version__)
"""))

# ─── PART 1 ──────────────────────────────────────────────────────────────────
cells.append(md(r"""
## Part 1 — Sinusoidal Positional Encoding

Self-attention is **permutation-equivariant**: without a position signal a transformer
sees a *bag of tokens* and "dog bites man" == "man bites dog". The original transformer
adds a fixed sinusoidal vector to each token embedding (§1–§2 of the reference):

$$PE(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad
  PE(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right).$$

Each pair of dims is a sinusoid of a different frequency, forming a geometric wavelength
ladder from $2\pi$ to $10000\cdot 2\pi$ — a continuous, multi-resolution position code.

**Exercise:** implement `sinusoidal_pe(seq_len, d)` returning a `(seq_len, d)` tensor.
"""))

cells.append(code(r"""
def sinusoidal_pe(seq_len, d, base=10000.0):
    # YOUR CODE HERE
    # 1. pos: column vector of positions 0..seq_len-1, shape (seq_len, 1)
    # 2. div: base ** (arange(0, d, 2) / d), shape (1, d/2)
    # 3. fill even dims with sin(pos/div), odd dims with cos(pos/div)
    raise NotImplementedError
"""))

cells.append(code(r"""
# TEST — do not modify
pe = sinusoidal_pe(50, 16)
assert pe.shape == (50, 16), pe.shape
# position 0 -> sin(0)=0 on even dims, cos(0)=1 on odd dims
assert torch.allclose(pe[0], torch.tensor([0., 1.] * 8), atol=1e-6)
plt.figure(figsize=(7, 4))
plt.imshow(sinusoidal_pe(100, 64).T, aspect='auto', cmap='RdBu')
plt.xlabel('position'); plt.ylabel('dimension'); plt.title('Sinusoidal PE')
plt.colorbar(); plt.tight_layout(); plt.show()
print("Part 1 OK")
"""))

# ─── PART 2 ──────────────────────────────────────────────────────────────────
cells.append(md(r"""
## Part 2 — RoPE From Scratch

Instead of *adding* a position vector, RoPE **rotates** the query and key vectors by an
angle proportional to their position (§4 of the reference). In 2D, rotating $W_q x_m$ by
$m\theta$ and $W_k x_n$ by $n\theta$ makes the dot product depend only on $m-n$, because
$R(\alpha)^\top R(\beta) = R(\beta-\alpha)$.

For $d$ dimensions we split into $d/2$ pairs, each with its own frequency
$\theta_i = \text{base}^{-2i/d}$, and rotate pair $i$ by $m\theta_i$. The efficient
"rotate-half" form (paired layout: dims $2i, 2i{+}1$ form pair $i$):

$$x'_{2i} = x_{2i}\cos m\theta_i - x_{2i+1}\sin m\theta_i, \qquad
  x'_{2i+1} = x_{2i}\sin m\theta_i + x_{2i+1}\cos m\theta_i.$$

**Exercise:** implement `rope_freqs(d)` and `apply_rope(x, positions)`. `apply_rope`
must work for `x` of shape `(..., T, dim)` with `positions` of shape `(T,)` — so it can
rotate both a plain `(T, d)` matrix and attention's `(B, heads, T, head_dim)` tensor.
"""))

cells.append(code(r"""
def rope_freqs(d, base=10000.0):
    # YOUR CODE HERE: return theta_i = base ** (-2i/d) for i = 0..d/2-1, shape (d/2,)
    raise NotImplementedError

def apply_rope(x, positions, base=10000.0):
    # YOUR CODE HERE
    # 1. dim = x.shape[-1]; theta = rope_freqs(dim); ang = positions[:,None] * theta[None,:]  (T, dim/2)
    # 2. cos, sin = ang.cos(), ang.sin(); reshape to broadcast over leading dims:
    #       shape = [1]*(x.dim()-2) + [T, dim//2]
    # 3. x1, x2 = x[..., 0::2], x[..., 1::2]   (the pairs)
    # 4. o1 = x1*cos - x2*sin ;  o2 = x1*sin + x2*cos
    # 5. interleave back: torch.stack((o1, o2), dim=-1).flatten(-2)
    raise NotImplementedError
"""))

cells.append(code(r"""
# TEST — do not modify
# d=2, pos=1, theta_0 = base**0 = 1  -> rotate (1,0) by 1 radian
out = apply_rope(torch.tensor([[1.0, 0.0]]), torch.tensor([1]))
exp = torch.tensor([[math.cos(1.0), math.sin(1.0)]])
assert torch.allclose(out, exp, atol=1e-6), (out, exp)
# rotation preserves norm
xr = torch.randn(5, 64)
assert torch.allclose(apply_rope(xr, torch.arange(5)).norm(dim=-1), xr.norm(dim=-1), atol=1e-5)
print("Part 2 OK")
"""))

# ─── PART 3 ──────────────────────────────────────────────────────────────────
cells.append(md(r"""
## Part 3 — The Relative-Position Property

The whole point of RoPE: the attention score between a query at position $m$ and a key at
position $n$ depends **only on $m-n$**, not on $m$ and $n$ separately. We verify it
numerically: fix random $q, k$, then for a given offset $\delta$ the score
$\langle \text{RoPE}(q, m), \text{RoPE}(k, m+\delta)\rangle$ should be the **same** for
every absolute $m$.
"""))

cells.append(code(r"""
# TEST — do not modify
torch.manual_seed(1)
d = 64
q, k = torch.randn(d), torch.randn(d)

def score(m, n):
    qm = apply_rope(q[None], torch.tensor([m]))[0]
    kn = apply_rope(k[None], torch.tensor([n]))[0]
    return (qm * kn).sum()

for delta in [0, 1, 3, 7, 15]:
    vals = torch.stack([score(m, m + delta) for m in range(20)])
    assert torch.allclose(vals, vals[0].expand_as(vals), atol=1e-4), (delta, vals[:3])
print("Part 3 OK — score depends only on relative offset (m-n)")
"""))

# ─── PART 4 ──────────────────────────────────────────────────────────────────
cells.append(md(r"""
## Part 4 — Cross-Check Against the Complex-Number Form

RoPE's clean derivation treats each 2D pair as a complex number and multiplies by
$e^{im\theta}$. PyTorch has native complex ops, so we can implement RoPE that way and
confirm it matches our rotate-half implementation bit-for-bit. This is exactly the
formulation LLaMA-style codebases use (up to a permutation of the pairing).
"""))

cells.append(code(r"""
# Reference implementation via complex multiplication (provided)
def apply_rope_complex(x, positions, base=10000.0):
    dim = x.shape[-1]
    theta = rope_freqs(dim, base).to(x.device)
    ang = positions.float()[:, None] * theta[None, :]            # (T, dim/2)
    rot = torch.polar(torch.ones_like(ang), ang)                 # (T, dim/2) complex
    rot = rot.reshape(*([1] * (x.dim() - 2)), *rot.shape)        # broadcast over leading dims
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], dim // 2, 2).contiguous())
    return torch.view_as_real(xc * rot).reshape(*x.shape)

# TEST — do not modify
xb = torch.randn(2, 4, 10, 64)        # (batch, heads, seq, head_dim)
pos = torch.arange(10)
assert torch.allclose(apply_rope(xb, pos), apply_rope_complex(xb, pos), atol=1e-5)
print("Part 4 OK — rotate-half == complex form")
"""))

# ─── PART 5 ──────────────────────────────────────────────────────────────────
cells.append(md(r"""
## Part 5 — Frequency / Wavelength Visualization

Each dimension pair rotates at its own rate $\theta_i$. Low-index pairs spin fast (encode
fine local position), high-index pairs spin slowly (encode coarse long-range position).
This spectrum is *why* context-extension methods (Part 6) treat dimensions differently.
"""))

cells.append(code(r"""
d = 64
positions = torch.arange(64)
theta = rope_freqs(d)
angles = positions[:, None] * theta[None, :]    # (pos, d/2)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for i in [0, 2, 8, 31]:                          # a few pairs, fast -> slow
    ax[0].plot(positions, torch.cos(angles[:, i]), label=f'pair {i} (theta={theta[i]:.4f})')
ax[0].set_xlabel('position'); ax[0].set_ylabel('cos(pos * theta_i)')
ax[0].set_title('Rotation per dimension pair'); ax[0].legend(fontsize=8)

ax[1].semilogy(range(d // 2), 2 * math.pi / theta)
ax[1].set_xlabel('pair index i'); ax[1].set_ylabel('wavelength (positions)')
ax[1].set_title('Geometric wavelength ladder')
plt.tight_layout(); plt.show()
print("Fast pairs = local detail, slow pairs = long-range position.")
"""))

# ─── PART 6 ──────────────────────────────────────────────────────────────────
cells.append(md(r"""
## Part 6 — Context Extension: the Position Interpolation Mechanism

A model trained to length $L$ has only ever seen relative angles $(m-n)\theta_i$ for
$|m-n| < L$. Run it on a longer window $L'$ and the slow dims reach **unseen** angles,
producing the *catastrophically high attention scores* Chen et al. describe.

**Position Interpolation (PI)** down-scales positions by $L/L'$ so the largest angle stays
inside the trained range — interpolating between known positions instead of extrapolating.
Below we visualize the angle of the slowest (most vulnerable) dimension under naive
extrapolation vs PI. PI stays within the trained band; naive extrapolation shoots past it.
"""))

cells.append(code(r"""
d = 64
L, Lp = 16, 64                       # trained length, extended length
theta_min = rope_freqs(d)[-1].item() # slowest dimension
pos = torch.arange(Lp)

vanilla = pos * theta_min                       # naive: use raw positions
pi      = pos * (L / Lp) * theta_min            # PI: scale positions by L/L'
trained_band = (L - 1) * theta_min              # max angle ever seen in training

plt.figure(figsize=(7, 4))
plt.plot(pos, vanilla, label='naive extrapolation')
plt.plot(pos, pi, label='Position Interpolation')
plt.axhline(trained_band, ls='--', color='gray', label='max angle seen in training')
plt.axvline(L, ls=':', color='red', label=f'train length L={L}')
plt.xlabel('position'); plt.ylabel('angle of slowest dim (rad)')
plt.title('Why PI works: it keeps angles in-distribution')
plt.legend(fontsize=8); plt.tight_layout(); plt.show()

print(f"At position {Lp-1}: naive={vanilla[-1]:.3f} rad, PI={pi[-1]:.3f} rad, "
      f"trained max={trained_band:.3f} rad")
print("NTK-aware / YaRN refine this by scaling per-frequency instead of uniformly (see reference).")
"""))

# ─── PART 7 ──────────────────────────────────────────────────────────────────
cells.append(md(r"""
## Part 7 — Experiment: Length Extrapolation (the payoff)

Now we *measure* the advantage. Task: a **delayed-copy** — at each position predict the
token that appeared `K=3` steps earlier. This is purely *relative*: the rule "look back 3"
is the same at every position and every length.

We train three identical tiny transformers, differing only in positional encoding —
**learned-absolute**, **sinusoidal**, **RoPE** — on sequences of length **16**, then test
on lengths **16 → 64**. A relative-aware encoding should keep working past the training
length; learned-absolute (whose position rows beyond 16 were never trained) should collapse.

This is the "ship *and* measure" habit from your career guide: don't just assert RoPE is
better — design a probe and read the numbers.
"""))

cells.append(code(r"""
# Model with pluggable positional encoding (provided — reuses YOUR apply_rope from Part 2)
class SelfAttention(nn.Module):
    def __init__(self, d, h, pe, base=10000.0):
        super().__init__()
        self.d, self.h, self.hd, self.pe, self.base = d, h, d // h, pe, base
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
    def forward(self, x, pos):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]               # (B, h, T, hd)
        if self.pe == 'rope':
            q, k = apply_rope(q, pos, self.base), apply_rope(k, pos, self.base)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        att = att.masked_fill(mask, float('-inf')).softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, self.d)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, d, h, pe, base):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = SelfAttention(d, h, pe, base)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x, pos):
        x = x + self.attn(self.ln1(x), pos)
        return x + self.mlp(self.ln2(x))

class TinyLM(nn.Module):
    def __init__(self, V, d=64, h=4, L=2, pe='rope', max_pos=256, base=10000.0):
        super().__init__()
        self.pe = pe; self.tok = nn.Embedding(V, d)
        if pe == 'learned':
            self.pos_emb = nn.Embedding(max_pos, d)
        elif pe == 'sinusoidal':
            self.register_buffer('sin_pe', sinusoidal_pe(max_pos, d, base))
        self.blocks = nn.ModuleList([Block(d, h, pe, base) for _ in range(L)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, V)
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx)
        if self.pe == 'learned':
            x = x + self.pos_emb(pos)[None]
        elif self.pe == 'sinusoidal':
            x = x + self.sin_pe[:T][None]
        for b in self.blocks:
            x = b(x, pos)
        return self.head(self.lnf(x))
"""))

cells.append(code(r"""
# Task data + train/eval (provided)
V, K, TRAIN_LEN = 20, 3, 16

def make_batch(B, T):
    x = torch.randint(0, V, (B, T))
    y = x.roll(K, dims=1).clone()      # y[:, t] = x[:, t-K]  (delayed copy)
    y[:, :K] = -100                    # ignore the first K wrapped positions
    return x, y

@torch.no_grad()
def accuracy(model, T, B=256):
    model.eval()
    x, y = make_batch(B, T)
    pred = model(x).argmax(-1)
    m = y != -100
    return (pred[m] == y[m]).float().mean().item()

def train(pe, steps=800, lr=3e-3):
    torch.manual_seed(0)
    model = TinyLM(V, pe=pe)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        model.train()
        x, y = make_batch(64, TRAIN_LEN)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

EVAL_LENS = [16, 32, 48, 64]
t0 = time.time(); results = {}
for pe in ['learned', 'sinusoidal', 'rope']:
    results[pe] = [accuracy(train(pe), T) for T in EVAL_LENS]
    print(f"{pe:11s} " + "  ".join(f"L={T}:{a:.2f}" for T, a in zip(EVAL_LENS, results[pe])))
print(f"total wall time: {time.time()-t0:.0f}s")
"""))

cells.append(code(r"""
# Plot: accuracy vs sequence length (trained only on length 16)
plt.figure(figsize=(7, 4.5))
for pe, marker in [('learned', 'o'), ('sinusoidal', 's'), ('rope', '^')]:
    plt.plot(EVAL_LENS, results[pe], marker=marker, label=pe)
plt.axvline(TRAIN_LEN, ls=':', color='red', label=f'train length = {TRAIN_LEN}')
plt.xlabel('evaluation sequence length'); plt.ylabel('delayed-copy accuracy')
plt.title('Length extrapolation: trained at 16, tested longer')
plt.ylim(0, 1.05); plt.legend(); plt.tight_layout(); plt.show()
"""))

cells.append(md(r"""
### What you should see

A representative run (your numbers will vary a little with the seed/hardware), all trained
only on length 16:

| encoding | L=16 | L=32 | L=48 | L=64 |
|----------|------|------|------|------|
| learned-absolute | 1.00 | 0.49 | 0.35 | 0.27 |
| sinusoidal | 1.00 | 0.50 | 0.37 | 0.29 |
| **RoPE** | 1.00 | **0.90** | **0.71** | **0.54** |

- **Learned-absolute** solves the training length but **collapses** past it — its position
  rows beyond 16 were never trained, so longer positions are random noise.
- **Sinusoidal** collapses almost as hard here: even though its encoding is *defined* at any
  length, the network learned to read position from an *added absolute* signal, and that
  decoding does not transfer to the unseen long-range part of the curve.
- **RoPE** degrades far more gracefully and stays well ahead: because position enters *only*
  as a relative rotation inside the dot product, the learned "look back 3" rule transfers
  to unseen lengths instead of being tied to absolute indices.

This is the small-scale shadow of the paper results in §7 of the reference: RoPE's relative
structure is what makes context extension (PI / NTK / YaRN) even possible. RoPE still drifts
eventually (0.54 at 4× the training length) — which is exactly the residual problem PI and
YaRN clean up.

**Try it yourself:** raise `EVAL_LENS` to 128, increase `K`, or apply Position
Interpolation (scale `pos` by `TRAIN_LEN / T`) inside the RoPE path and watch long-length
accuracy recover.
"""))

# ─── WRITE ───────────────────────────────────────────────────────────────────
nb.cells = cells
import os
out = os.path.join(os.path.dirname(__file__), "positional_embeddings_tutorial.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {out} ({len(cells)} cells)")
