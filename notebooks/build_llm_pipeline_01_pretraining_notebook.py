"""
Generates notebooks/llm_training_pipeline/01_transformer_and_pretraining.ipynb
from cell definitions.
Run: python3 notebooks/build_llm_pipeline_01_pretraining_notebook.py
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
# LLM Training Pipeline — Part 1: Transformer Architecture & Pretraining

Stage 1 of 6 in `notebooks/llm_training_pipeline/`. Builds a ~14M-parameter
decoder-only transformer from scratch and pretrains it on TinyStories.
Later notebooks (SFT, reward model + PPO, DPO, evaluation, RLVR/GRPO) load
the checkpoint this notebook produces.

**How to use this notebook:**
- Read each theory section; keep `docs/llm_training_pipeline_reference.html`
  open in another tab for the full derivations.
- Code and tests are already implemented and verified — run cells top to
  bottom. Answer the **Question** cells yourself; that is the reflective part
  of this notebook.

**Parts:**
1. BPE Tokenizer
2. Causal Self-Attention, MLP, Transformer Block
3. Full GPT Model
4. Data Loading & Packing
5. Pretraining Loop
"""))

# ─── SETUP ───────────────────────────────────────────────────────────────────
cells.append(code("""
import time, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

CKPT_DIR = "../../data/checkpoints/llm_training_pipeline"
os.makedirs(CKPT_DIR, exist_ok=True)
torch.manual_seed(0)
"""))

# ─── PART 1: TOKENIZER ───────────────────────────────────────────────────────
cells.append(md("""
---
## Part 1: BPE Tokenizer

Language models operate on integer token ids, not raw text. We train a
byte-level BPE tokenizer (same family as GPT-2's tokenizer) on a sample of
TinyStories, vocab size 8000. Byte-level BPE can represent *any* input
string (it falls back to raw bytes for unseen sequences), so there is no
"unknown token" problem.
"""))

cells.append(code("""
print("Loading TinyStories (train[:50000])...")
ds = load_dataset('roneneldan/TinyStories', split='train[:50000]')
texts = [x['text'] for x in ds]
print(f"{len(texts)} stories loaded")

tok_train_path = f"{CKPT_DIR}/tinystories_tok_train.txt"
with open(tok_train_path, 'w') as f:
    f.write('\\n'.join(texts[:20000]))

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=[tok_train_path], vocab_size=8000, min_frequency=2,
    special_tokens=['<|endoftext|>'],
)
EOT_ID = tokenizer.token_to_id('<|endoftext|>')
print(f"Vocab size: {tokenizer.get_vocab_size()}, EOT id: {EOT_ID}")
"""))

cells.append(code("""
# TEST 1: tokenizer roundtrip + vocab size
test_strings = [
    "Once upon a time, there was a little girl named Lily.",
    "The dog ran to the park and played with a ball.",
    "\\"I am happy,\\" said Tom. \\"Let's go home!\\"",
]
for s in test_strings:
    ids = tokenizer.encode(s).ids
    decoded = tokenizer.decode(ids)
    assert decoded.strip() == s.strip(), f"roundtrip mismatch: {s!r} -> {decoded!r}"
    print(f"  OK ({len(ids)} tokens): {s[:40]}...")

assert tokenizer.get_vocab_size() == 8000
print("TEST 1 PASSED — tokenizer roundtrip and vocab size verified")
"""))

cells.append(md("""
### Question 1

**Why does a byte-level BPE tokenizer never need an "unknown token"?** What
would happen with a purely word-level tokenizer (split on whitespace, one id
per unique word) applied to text containing a word it never saw during
tokenizer training?

*Write your answer below (double-click this cell to edit):*

"""))

# ─── PART 2: ATTENTION, MLP, BLOCK ───────────────────────────────────────────
cells.append(md("""
---
## Part 2: Causal Self-Attention, MLP, Transformer Block

See `docs/llm_training_pipeline_reference.html#s2` for the full derivation
of the `1/sqrt(d_k)` scaling and the pre-norm residual structure used below.
"""))

cells.append(code("""
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 8000
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(
            1, 1, config.block_size, config.block_size
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
"""))

cells.append(code("""
# TEST 2: shape + causality
torch.manual_seed(0)
test_cfg = GPTConfig(vocab_size=100, block_size=16, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
block = Block(test_cfg)
block.eval()

B, T, C = 4, 16, 32
x = torch.randn(B, T, C)
y = block(x)
assert y.shape == (B, T, C), f"expected {(B,T,C)}, got {y.shape}"
print(f"TEST 2a PASSED — Block output shape {tuple(y.shape)}")

# causality: perturbing token T-1 must not change output at positions < T-1
x2 = x.clone()
x2[:, -1, :] = torch.randn(B, C)
y1 = block(x)
y2 = block(x2)
assert torch.allclose(y1[:, :-1, :], y2[:, :-1, :], atol=1e-6), "causality violated in Block!"
print("TEST 2b PASSED — causal mask verified: perturbing the last token does not change earlier outputs")
"""))

cells.append(md("""
### Question 2

Attention mixes information **across** positions; the MLP processes each
position **independently**. Given that, why does the causality test above
only need to check the attention path — could a bug in the MLP ever leak
future-token information into an earlier position's output?

*Write your answer below:*

"""))

# ─── PART 3: FULL GPT MODEL ──────────────────────────────────────────────────
cells.append(md("""
---
## Part 3: Full GPT Model

Assembles token + position embeddings, the block stack, and a tied LM head.
See `docs/llm_training_pipeline_reference.html#s2` for why the embedding and
LM head weights are tied.
"""))

cells.append(code("""
class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"
        pos = torch.arange(T, device=idx.device).unsqueeze(0)

        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
"""))

cells.append(code("""
# TEST 3: param count, gradient flow, generate() shape
torch.manual_seed(0)
test_cfg = GPTConfig(vocab_size=100, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
model = GPTModel(test_cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f"param count: {n_params:,}")
assert n_params > 0

B, T = 4, 16
idx = torch.randint(0, test_cfg.vocab_size, (B, T))
targets = torch.randint(0, test_cfg.vocab_size, (B, T))
logits, loss = model(idx, targets)
assert logits.shape == (B, T, test_cfg.vocab_size)
print(f"TEST 3a PASSED — logits shape {tuple(logits.shape)}, loss {loss.item():.3f}")

loss.backward()
n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
n_total = sum(1 for p in model.parameters())
assert n_with_grad == n_total, f"only {n_with_grad}/{n_total} params received gradient"
print(f"TEST 3b PASSED — all {n_total} parameters received gradient")

out = model.generate(idx[:, :4], max_new_tokens=5)
assert out.shape == (B, 4 + 5)
print(f"TEST 3c PASSED — generate() output shape {tuple(out.shape)}")

print("TEST 3 PASSED")
"""))

cells.append(md("""
### Question 3

The `_init_weights` method initializes every `nn.Linear` and `nn.Embedding`
weight from `N(0, 0.02^2)`, independent of the layer's fan-in. A classic
"Xavier/Glorot" initialization instead scales the variance by `1/fan_in`.
Given that this model uses pre-norm residual blocks (LayerNorm re-normalizes
the input to every sub-layer), why might a fixed small std work fine here
even though it ignores fan-in?

*Write your answer below:*

"""))

# Parts 4-5 are appended here.

# ─── WRITE ───────────────────────────────────────────────────────────────────
nb['cells'] = cells
OUTPUT_PATH = "llm_training_pipeline/01_transformer_and_pretraining.ipynb"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    nbf.write(nb, f)
print(f"Wrote {OUTPUT_PATH} with {len(cells)} cells")
