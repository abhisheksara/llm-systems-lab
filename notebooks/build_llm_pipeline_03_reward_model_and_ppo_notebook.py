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

# Parts 2-4 are appended here.

# ─── WRITE ───────────────────────────────────────────────────────────────────
nb['cells'] = cells
OUTPUT_PATH = "llm_training_pipeline/03_reward_model_and_ppo.ipynb"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    nbf.write(nb, f)
print(f"Wrote {OUTPUT_PATH} with {len(cells)} cells")
