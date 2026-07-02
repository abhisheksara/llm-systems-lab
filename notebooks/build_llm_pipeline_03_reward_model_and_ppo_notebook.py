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

# ─── WRITE ───────────────────────────────────────────────────────────────────
nb['cells'] = cells
OUTPUT_PATH = "llm_training_pipeline/03_reward_model_and_ppo.ipynb"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    nbf.write(nb, f)
print(f"Wrote {OUTPUT_PATH} with {len(cells)} cells")
