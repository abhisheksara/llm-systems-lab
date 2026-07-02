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

# Parts 1-5 are appended here by Tasks 6-10.

# ─── WRITE ───────────────────────────────────────────────────────────────────
nb['cells'] = cells
OUTPUT_PATH = "llm_training_pipeline/01_transformer_and_pretraining.ipynb"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    nbf.write(nb, f)
print(f"Wrote {OUTPUT_PATH} with {len(cells)} cells")
