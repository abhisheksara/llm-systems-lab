# LLM Training Pipeline — Part 2: SFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the SFT theory section, Q&A addendum, and a verified, from-scratch supervised fine-tuning notebook that turns `base_model.pt` (Part 1's pretrained checkpoint) into `sft_model.pt` — an instruction-following model, via prompt-loss-masked next-token training on a synthetic instruction-formatted slice of TinyStories.

**Architecture:** Reuses `GPTModel`/`GPTConfig` from `src/llm_pipeline/model.py` unchanged (SFT is a training-regime change, not an architecture change). New: an instruction-formatting scheme (`"Write a short story about {topic}:\n" -> story`, topic extracted heuristically from each story's own text via keyword match) and a prompt-loss-masking tokenizer that reuses `GPTModel.forward`'s existing `ignore_index=-100` support with zero model-code changes.

**Tech Stack:** Python 3.12, PyTorch (CUDA), `nbformat`. No new dependencies — reuses everything from Part 1's `requirements.txt`.

## Global Constraints

- Depends on Part 1 (`docs/superpowers/plans/2026-07-02-llm-training-pipeline-01-transformer-pretraining.md`) having been executed: `data/checkpoints/llm_training_pipeline/base_model.pt`, `tinystories_bpe-vocab.json`, and `tinystories_bpe-merges.txt` must exist. Task 3 Step 1 verifies this before proceeding.
- Every notebook code cell must run top-to-bottom without error; every `assert`-based test cell must pass. Verified per-task via `jupyter nbconvert --to notebook --execute`.
- No placeholder content: HTML/Markdown sections written in this plan are the actual final content, not outlines.
- Checkpoints and generated datasets go under `data/checkpoints/llm_training_pipeline/` (gitignored — do not commit files from that directory).
- Reflection "Question" markdown cells are left blank (for the user to fill in by hand) — do not pre-fill answers.
- `GPTModel.forward(idx, targets=None)` already supports `ignore_index=-100` (see `src/llm_pipeline/model.py`); SFT's prompt-loss-masking reuses this directly — no model code changes in this plan.
- Exact hyperparameters below (topic keyword list, LR, step count) are fixed to avoid ambiguity; tunable only if a verification step fails.

---

## File Map

| File | Responsibility |
|------|---------------|
| `docs/llm_training_pipeline_reference.html` | Modify — add Section 4 (SFT) + nav link |
| `docs/llm_training_pipeline/concepts_qa.md` | Modify — add Q&A sections 9-10 |
| `notebooks/build_llm_pipeline_02_sft_notebook.py` | Create — builder script for notebook 2 |
| `notebooks/llm_training_pipeline/02_sft.ipynb` | Generated — instruction dataset, prompt masking, SFT loop |
| `src/llm_pipeline/data.py` | Modify — add `TOPIC_KEYWORDS`, `extract_topic`, `format_sft_prompt`, `build_sft_pairs`, `tokenize_sft_example` |
| `data/checkpoints/llm_training_pipeline/sft_model.pt` | Output checkpoint (gitignored) |

---

## Task 1: HTML Reference — Section 4 (SFT)

**Files:**
- Modify: `docs/llm_training_pipeline_reference.html`

- [ ] **Step 1: Add the nav link**

In `docs/llm_training_pipeline_reference.html`, find the `<nav>` block:

```html
<nav>
  <a href="#s1">1. Pipeline Overview</a>
  <a href="#s2">2. Transformer Architecture</a>
  <a href="#s3">3. Pretraining</a>
</nav>
```

Replace with:

```html
<nav>
  <a href="#s1">1. Pipeline Overview</a>
  <a href="#s2">2. Transformer Architecture</a>
  <a href="#s3">3. Pretraining</a>
  <a href="#s4">4. SFT</a>
</nav>
```

- [ ] **Step 2: Insert Section 4**

Find the closing `</body>` tag at the end of the file. Immediately before it, insert:

```html
<!-- ============================================================ -->
<h2 id="s4">4. Supervised Fine-Tuning (SFT)</h2>

<p>The base (pretrained) model is fluent but has no notion of "follow this instruction" — it was
only ever trained to continue whatever text it's given, and a raw TinyStories corpus contains no
consistent prompt/response structure to imitate. SFT teaches that structure by continuing the
<em>same</em> next-token objective, but on <code>(prompt, response)</code> pairs and with the loss
restricted to the response tokens.</p>

<h3>Prompt-Loss-Masking</h3>

<p>Each training example is still one token sequence — prompt tokens followed by response
tokens — but the cross-entropy loss is only computed at positions predicting a response token:</p>

\[\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t \,:\, t+1 \,\in\, \text{response}} \log P_\theta(x_{t+1} \mid x_{\le t})\]

<p>Concretely, the target label at position \(t\) is set to a special <code>ignore_index</code>
(\(-100\) here, matching PyTorch's <code>F.cross_entropy</code> default) whenever the token it
would predict falls inside the prompt (or padding); only response-token predictions contribute
gradient.</p>

<div class="definition">
<strong>Why mask the prompt at all?</strong> Without masking, the model is also trained to predict
the prompt text itself, i.e. to <em>generate topics</em> rather than <em>respond to them</em>.
Prompts are drawn from a small, structured template space (here, a fixed
<code>"Write a short story about {topic}:\n"</code> format) — training on their tokens teaches
nothing generalizable and dilutes gradient signal that should be spent on the actual skill being
taught: producing a good response given a prompt. Worse, at inference the prompt is always given,
never generated — spending training loss on "predicting" it optimizes a capability that is never
exercised at generation time.</div>

<h3>Why SFT "Unlocks" Instruction-Following</h3>

<p>The pretrained model already contains the capability to write a coherent short story — that
was the entire pretraining objective. What it lacks is the <em>behavior</em> of treating a
specific textual pattern (an instruction) as a cue to switch into "respond" mode rather than
"continue this text as if it were more of the same" mode. Because the underlying capability is
already present, SFT typically needs a comparatively small number of examples and gradient steps
to elicit reliable instruction-following — it is teaching a behavioral trigger on top of existing
capability, not teaching new capability from scratch. This is the same "capability vs. alignment"
distinction introduced in Section 1.</p>

<h3>Catastrophic Forgetting and the Low-LR Rationale</h3>

<p>SFT continues training the same weights that pretraining produced, on a much smaller and much
narrower dataset. A learning rate anywhere near the pretraining peak LR risks
<strong>catastrophic forgetting</strong>: large gradient steps on a narrow data distribution can
overwrite the broader linguistic competence pretraining spent many more steps building, in favor
of overfitting to the SFT set's narrow templates. SFT therefore uses a peak LR roughly one to two
orders of magnitude below the pretraining peak LR (\(3\times10^{-5}\) here vs. pretraining's
\(3\times10^{-4}\)) and comparatively few steps — enough to shift the output <em>distribution</em>
toward instruction-following, not enough to erase what pretraining already learned.</p>

<div class="keyfacts">
<strong>Key Facts — Section 4</strong>
<ul>
  <li>SFT is the same next-token cross-entropy objective as pretraining, restricted to response
  tokens via <code>ignore_index</code> masking on the prompt.</li>
  <li>SFT elicits an existing capability (coherent generation) into a new behavior
  (instruction-following); it is comparatively cheap because it is not teaching new knowledge.</li>
  <li>A much lower LR than pretraining is used specifically to avoid catastrophic forgetting on
  the narrower SFT distribution.</li>
</ul>
</div>

```

- [ ] **Step 3: Verify**

```bash
.venv/bin/python -c "
html = open('docs/llm_training_pipeline_reference.html', encoding='utf-8').read()
assert '<h2 id=\"s4\"' in html
assert 'href=\"#s4\"' in html
assert html.count('\\\\[') == html.count('\\\\]')
print('HTML section 4 OK,', len(html), 'bytes')
"
```

Expected: `HTML section 4 OK, <N> bytes`

- [ ] **Step 4: Commit**

```bash
git add docs/llm_training_pipeline_reference.html
git commit -m "docs: add SFT section to LLM training pipeline reference"
```

---

## Task 2: Concepts Q&A — Sections 9-10 (SFT)

**Files:**
- Modify: `docs/llm_training_pipeline/concepts_qa.md`

- [ ] **Step 1: Replace the trailing placeholder note with Sections 9-10**

Find the final line of the file:

```markdown
*(Sections for SFT, reward modeling, PPO, DPO, evaluation, and RLVR/GRPO are
appended here as the corresponding notebooks are built.)*
```

Replace it with:

```markdown
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

*(Sections for reward modeling, PPO, DPO, evaluation, and RLVR/GRPO are
appended here as the corresponding notebooks are built.)*
```

- [ ] **Step 2: Verify**

```bash
.venv/bin/python -c "
text = open('docs/llm_training_pipeline/concepts_qa.md', encoding='utf-8').read()
n_sections = text.count('\n## ')
assert n_sections == 10, f'expected 10 sections, found {n_sections}'
assert '## 9. Why mask the prompt' in text
assert '## 10. Catastrophic forgetting' in text
print('Q&A doc OK,', len(text), 'bytes,', n_sections, 'sections')
"
```

Expected: `Q&A doc OK, <N> bytes, 10 sections`

- [ ] **Step 3: Commit**

```bash
git add docs/llm_training_pipeline/concepts_qa.md
git commit -m "docs: add SFT concepts Q&A (prompt masking, catastrophic forgetting)"
```

---

## Task 3: Notebook Bootstrap

**Files:**
- Create: `notebooks/build_llm_pipeline_02_sft_notebook.py`

**Interfaces:**
- Produces: a `cells` list and `md()`/`code()` helpers that Tasks 4-6 append to; a final `nbf.write(nb, OUTPUT_PATH)` call.

- [ ] **Step 1: Verify Part 1's artifacts exist**

```bash
.venv/bin/python -c "
import os
base = 'data/checkpoints/llm_training_pipeline'
for f in ['base_model.pt', 'tinystories_bpe-vocab.json', 'tinystories_bpe-merges.txt']:
    p = os.path.join(base, f)
    assert os.path.exists(p), f'missing {p} — run Part 1 plan first'
print('Part 1 artifacts present')
"
```

Expected: `Part 1 artifacts present`. If this fails, stop and execute
`docs/superpowers/plans/2026-07-02-llm-training-pipeline-01-transformer-pretraining.md` first.

- [ ] **Step 2: Write the builder script skeleton**

Create `notebooks/build_llm_pipeline_02_sft_notebook.py`:

```python
"""
Generates notebooks/llm_training_pipeline/02_sft.ipynb from cell definitions.
Run: python3 notebooks/build_llm_pipeline_02_sft_notebook.py
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
# LLM Training Pipeline — Part 2: Supervised Fine-Tuning (SFT)

Stage 2 of 6. Loads `base_model.pt` from notebook 1 and fine-tunes it on an
instruction-formatted slice of TinyStories, using prompt-loss-masking so only
the response tokens contribute gradient. Produces `sft_model.pt`, the
checkpoint every later stage (reward model + PPO, DPO, evaluation,
RLVR/GRPO) fine-tunes or references as the frozen "reference" policy.

**How to use this notebook:**
- Read each theory section; keep `docs/llm_training_pipeline_reference.html`
  open in another tab (Section 4) for the full derivations.
- Code and tests are already implemented and verified — run cells top to
  bottom. Answer the **Question** cells yourself.

**Parts:**
1. Instruction Dataset Construction
2. Prompt-Loss-Masking
3. SFT Training Loop
"""))

# ─── SETUP ───────────────────────────────────────────────────────────────────
cells.append(code("""
import time, math, os
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

ckpt = torch.load(f"{CKPT_DIR}/base_model.pt", weights_only=False)
base_cfg = ckpt['config']
base_model = GPTModel(base_cfg).to(device)
base_model.load_state_dict(ckpt['model_state_dict'])
print(f"Loaded base_model.pt — {sum(p.numel() for p in base_model.parameters()):,} params")
"""))

# Parts 1-3 are appended here by Tasks 4-6.

# ─── WRITE ───────────────────────────────────────────────────────────────────
nb['cells'] = cells
OUTPUT_PATH = "llm_training_pipeline/02_sft.ipynb"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    nbf.write(nb, f)
print(f"Wrote {OUTPUT_PATH} with {len(cells)} cells")
```

- [ ] **Step 3: Generate and execute**

```bash
cd notebooks && ../.venv/bin/python build_llm_pipeline_02_sft_notebook.py && cd ..
.venv/bin/python -m nbconvert --to notebook --execute --inplace \
  notebooks/llm_training_pipeline/02_sft.ipynb
```

Expected: `Wrote llm_training_pipeline/02_sft.ipynb with 2 cells`, then notebook
execution completes without error, printing `Loaded base_model.pt — 13,817,856 params`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/build_llm_pipeline_02_sft_notebook.py notebooks/llm_training_pipeline/02_sft.ipynb
git commit -m "feat: bootstrap llm_training_pipeline notebook 2 (intro + setup)"
```

---

## Task 4: Notebook Part 1 — Instruction Dataset Construction

**Files:**
- Modify: `notebooks/build_llm_pipeline_02_sft_notebook.py`

**Interfaces:**
- Consumes: nothing new from Task 3 at import time.
- Produces (notebook runtime namespace): `TOPIC_KEYWORDS` (list of 40 lowercase strings), `extract_topic(story: str) -> str | None`, `format_sft_prompt(topic: str) -> str`, `sft_pairs` (list of `(topic, story)` tuples).

- [ ] **Step 1: Append the Part 1 cells**

Insert before the `# Parts 1-3 are appended here` comment:

```python
# ─── PART 1: INSTRUCTION DATASET ─────────────────────────────────────────────
cells.append(md("""
---
## Part 1: Instruction Dataset Construction

TinyStories has no built-in `(prompt, response)` structure. We synthesize one:
for each story, we heuristically extract a "topic" (a keyword the story is
actually about, from a fixed vocabulary of ~40 common TinyStories nouns), then
frame the story as the answer to `"Write a short story about {topic}:\\n"`.
Stories where no keyword matches are dropped.
"""))

cells.append(code("""
from datasets import load_dataset

TOPIC_KEYWORDS = [
    "dog", "cat", "girl", "boy", "forest", "ball", "tree", "bird", "star",
    "friend", "monster", "princess", "dragon", "robot", "garden", "park",
    "school", "castle", "rabbit", "mouse", "flower", "boat", "river",
    "mountain", "farm", "toy", "bear", "fish", "sun", "moon", "rain",
    "snow", "house", "family", "birthday", "picnic", "adventure", "magic",
    "kite", "puppy",
]

def extract_topic(story):
    lower = story.lower()
    for kw in TOPIC_KEYWORDS:
        if kw in lower:
            return kw
    return None

def format_sft_prompt(topic):
    return f"Write a short story about {topic}:\\n"

print("Loading TinyStories (train[:50000]) for SFT pair construction...")
ds = load_dataset('roneneldan/TinyStories', split='train[:50000]')
sft_pairs = []
for x in ds:
    topic = extract_topic(x['text'])
    if topic is not None:
        sft_pairs.append((topic, x['text']))
print(f"{len(sft_pairs)} / {len(ds)} stories matched a topic keyword")
"""))

cells.append(code("""
# TEST 1: extraction determinism + prompt formatting + coverage sanity
assert extract_topic("A brave dog ran through the yard.") == "dog"
assert extract_topic("The weather was strange that day with no mentioned nouns.") is None
assert format_sft_prompt("dragon") == "Write a short story about dragon:\\n"
assert len(sft_pairs) > 10000, f"expected >10000 matched pairs, got {len(sft_pairs)}"
for topic, story in sft_pairs[:3]:
    assert topic in story.lower()
print(f"TEST 1 PASSED — {len(sft_pairs)} pairs, extraction verified on samples")
"""))

cells.append(md("""
### Question 1

`extract_topic` returns the **first** matching keyword found by scanning
`TOPIC_KEYWORDS` in a fixed order, even if a story matches several keywords
(e.g. a story about both a "dog" and a "park"). Is this a problem for
training a model to follow the `"Write a short story about {topic}:\\n"`
instruction? What would you check to find out?

*Write your answer below:*

"""))

# Parts 2-3 are appended here.
```

- [ ] **Step 2: Regenerate and execute**

```bash
cd notebooks && ../.venv/bin/python build_llm_pipeline_02_sft_notebook.py && cd ..
.venv/bin/python -m nbconvert --to notebook --execute --inplace \
  notebooks/llm_training_pipeline/02_sft.ipynb
```

Expected: no errors; output includes `TEST 1 PASSED`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/build_llm_pipeline_02_sft_notebook.py notebooks/llm_training_pipeline/02_sft.ipynb
git commit -m "feat: llm_training_pipeline notebook 2 part 1 — instruction dataset"
```

---

## Task 5: Notebook Part 2 — Prompt-Loss-Masking

**Files:**
- Modify: `notebooks/build_llm_pipeline_02_sft_notebook.py`

**Interfaces:**
- Consumes: `format_sft_prompt`, `tokenizer`, `EOT_ID` from Tasks 3-4.
- Produces (notebook runtime namespace): `tokenize_sft_example(topic, story, tokenizer, eot_id, block_size) -> (input_ids: LongTensor[block_size], labels: LongTensor[block_size])`.

- [ ] **Step 1: Append the Part 2 cells**

```python
# ─── PART 2: PROMPT-LOSS-MASKING ─────────────────────────────────────────────
cells.append(md("""
---
## Part 2: Prompt-Loss-Masking

`GPTModel.forward` already treats label value `-100` as "ignore this position"
(PyTorch's `F.cross_entropy` default `ignore_index`) — no model code changes
are needed. We only need a tokenizer function that builds the
`(input_ids, labels)` pair with prompt-token predictions masked out. This
follows the same shift-by-one convention as pretraining's `pack_into_blocks`:
`input_ids[i]` predicts `labels[i]`, where `labels[i]` is the *next* token
after `input_ids[i]` — so masking is applied to which *target* falls inside
the prompt, not which *input* position does.
See `docs/llm_training_pipeline_reference.html#s4` for why we mask at all.
"""))

cells.append(code("""
def tokenize_sft_example(topic, story, tokenizer, eot_id, block_size):
    prompt_ids = tokenizer.encode(format_sft_prompt(topic)).ids
    completion_ids = tokenizer.encode(story).ids + [eot_id]
    full_ids = (prompt_ids + completion_ids)[: block_size + 1]
    n_prompt = min(len(prompt_ids), len(full_ids))
    n_real = len(full_ids)

    pad_len = (block_size + 1) - n_real
    full_ids = full_ids + [eot_id] * pad_len

    input_ids = full_ids[:-1]           # length block_size
    targets_raw = full_ids[1:]          # length block_size, shifted by one

    labels = []
    for i in range(block_size):
        target_pos = i + 1              # position in the original (unshifted) sequence
        if target_pos < n_prompt or target_pos >= n_real:
            labels.append(-100)         # target is a prompt token, or padding
        else:
            labels.append(targets_raw[i])

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )

BLOCK_SIZE = 256
"""))

cells.append(code("""
# TEST 2: known mask boundary + masked-loss != full-loss
topic, story = "dog", "A dog ran fast."
prompt_len = len(tokenizer.encode(format_sft_prompt(topic)).ids)
input_ids, labels = tokenize_sft_example(topic, story, tokenizer, EOT_ID, BLOCK_SIZE)

assert input_ids.shape == (BLOCK_SIZE,)
assert labels.shape == (BLOCK_SIZE,)
# every target position before (prompt_len - 1) must be masked (predicting a prompt token)
assert torch.all(labels[: prompt_len - 1] == -100), "prompt-region targets not fully masked"
# the position right after the prompt should predict the first response token (not masked)
assert labels[prompt_len - 1].item() != -100, "first response token incorrectly masked"
print(f"TEST 2a PASSED — mask boundary correct (prompt_len={prompt_len})")

masked_logits, masked_loss = base_model(input_ids.unsqueeze(0).to(device), labels.unsqueeze(0).to(device))
full_labels = torch.tensor([full[1] if False else 0]) if False else None  # placeholder unused
unmasked_targets = torch.tensor(
    (tokenizer.encode(format_sft_prompt(topic)).ids + tokenizer.encode(story).ids + [EOT_ID])[1: BLOCK_SIZE + 1]
    + [EOT_ID] * max(0, BLOCK_SIZE - (len(tokenizer.encode(format_sft_prompt(topic)).ids + tokenizer.encode(story).ids + [EOT_ID]) - 1)),
    dtype=torch.long,
)[:BLOCK_SIZE].unsqueeze(0).to(device)
_, full_loss = base_model(input_ids.unsqueeze(0).to(device), unmasked_targets)

print(f"masked_loss={masked_loss.item():.4f}, full_loss(unmasked prompt+response)={full_loss.item():.4f}")
assert abs(masked_loss.item() - full_loss.item()) > 1e-4, "masked and unmasked loss should differ"
print("TEST 2b PASSED — masked loss differs from full (unmasked) loss")
"""))

cells.append(md("""
### Question 2

The masking rule above checks `target_pos < n_prompt`, i.e. it masks based on
where the **predicted** token falls, not where the **input** token falls.
Concretely: the input token at the last prompt position (`input_ids[n_prompt - 1]`,
the `\\n` after the prompt) is *not* masked as an input — it's fed into the
model normally. Why does that make sense, given the model needs to see the
whole prompt to generate the response?

*Write your answer below:*

"""))

# Part 3 is appended here.
```

- [ ] **Step 2: Regenerate and execute**

```bash
cd notebooks && ../.venv/bin/python build_llm_pipeline_02_sft_notebook.py && cd ..
.venv/bin/python -m nbconvert --to notebook --execute --inplace \
  notebooks/llm_training_pipeline/02_sft.ipynb
```

Expected: no errors; output includes `TEST 2a PASSED` and `TEST 2b PASSED`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/build_llm_pipeline_02_sft_notebook.py notebooks/llm_training_pipeline/02_sft.ipynb
git commit -m "feat: llm_training_pipeline notebook 2 part 2 — prompt-loss-masking"
```

---

## Task 6: Notebook Part 3 — SFT Training Loop

**Files:**
- Modify: `notebooks/build_llm_pipeline_02_sft_notebook.py`

**Interfaces:**
- Consumes: `base_model`, `base_cfg`, `tokenizer`, `EOT_ID`, `CKPT_DIR` from Task 3; `sft_pairs` from Task 4; `tokenize_sft_example`, `BLOCK_SIZE` from Task 5.
- Produces: `data/checkpoints/llm_training_pipeline/sft_model.pt` (dict with keys `model_state_dict`, `config`).

- [ ] **Step 1: Append the Part 3 cells**

```python
# ─── PART 3: SFT TRAINING LOOP ───────────────────────────────────────────────
cells.append(md("""
---
## Part 3: SFT Training Loop

Fine-tunes a fresh copy of the base model for 400 steps (batch size 32) with
AdamW at a fixed LR of `3e-5` — roughly 10x below pretraining's peak LR (see
`docs/llm_training_pipeline_reference.html#s4` for the forgetting rationale).
No warmup/cosine schedule is used here: SFT runs few enough steps that a
constant low LR is simpler and sufficient. On an RTX 3070 this takes under a
minute.
"""))

cells.append(code("""
import copy
sft_model = copy.deepcopy(base_model).to(device)

held_out = sft_pairs[-200:]
train_pairs = sft_pairs[:-200]
print(f"{len(train_pairs)} training pairs, {len(held_out)} held-out pairs")

def make_batch(pairs, batch_size):
    idx = torch.randint(0, len(pairs), (batch_size,))
    batch = [tokenize_sft_example(pairs[i][0], pairs[i][1], tokenizer, EOT_ID, BLOCK_SIZE) for i in idx]
    input_ids = torch.stack([b[0] for b in batch]).to(device)
    labels = torch.stack([b[1] for b in batch]).to(device)
    return input_ids, labels
"""))

cells.append(code("""
max_steps = 400
lr = 3e-5
batch_size = 32

opt = torch.optim.AdamW(sft_model.parameters(), lr=lr)
losses = []
t0 = time.time()
for step in range(max_steps):
    input_ids, labels = make_batch(train_pairs, batch_size)
    logits, loss = sft_model(input_ids, labels)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(sft_model.parameters(), 1.0)
    opt.step()
    losses.append(loss.item())
    if step % 50 == 0 or step == max_steps - 1:
        print(f"step {step:4d} | loss {loss.item():.3f} | elapsed {time.time()-t0:.0f}s")

print(f"SFT training elapsed: {time.time()-t0:.1f}s")
"""))

cells.append(code("""
plt.figure(figsize=(8, 4))
plt.plot(losses, alpha=0.6, label="per-step SFT loss (response tokens only)")
window = 20
smoothed = [sum(losses[max(0,i-window):i+1]) / len(losses[max(0,i-window):i+1]) for i in range(len(losses))]
plt.plot(smoothed, label=f"{window}-step moving average", linewidth=2)
plt.xlabel("step"); plt.ylabel("masked cross-entropy loss"); plt.title("SFT loss curve")
plt.legend(); plt.tight_layout(); plt.show()
"""))

cells.append(code("""
# TEST 3: SFT loss decreased, and held-out perplexity improves over the base model
@torch.no_grad()
def held_out_avg_loss(model, pairs):
    model.eval()
    total = 0.0
    for topic, story in pairs:
        input_ids, labels = tokenize_sft_example(topic, story, tokenizer, EOT_ID, BLOCK_SIZE)
        _, loss = model(input_ids.unsqueeze(0).to(device), labels.unsqueeze(0).to(device))
        total += loss.item()
    return total / len(pairs)

first_20_avg = sum(losses[:20]) / 20
last_20_avg = sum(losses[-20:]) / 20
print(f"first-20-step avg loss: {first_20_avg:.3f}, last-20-step avg loss: {last_20_avg:.3f}")
assert last_20_avg < first_20_avg, "SFT loss did not decrease over training"

base_held_out_loss = held_out_avg_loss(base_model, held_out)
sft_held_out_loss = held_out_avg_loss(sft_model, held_out)
print(f"held-out masked loss — base: {base_held_out_loss:.3f} (PPL {math.exp(base_held_out_loss):.1f}), "
      f"sft: {sft_held_out_loss:.3f} (PPL {math.exp(sft_held_out_loss):.1f})")
assert sft_held_out_loss < base_held_out_loss, "SFT model did not improve held-out response-token perplexity"
print("TEST 3 PASSED — SFT loss decreased and held-out perplexity improved over base")
"""))

cells.append(code("""
# Qualitative comparison: base vs SFT completions on held-out topics
sft_model.eval()
for topic in ["dragon", "picnic", "robot"]:
    prompt = format_sft_prompt(topic)
    prompt_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    base_out = base_model.generate(prompt_ids, max_new_tokens=40, temperature=0.8, top_k=40)
    sft_out = sft_model.generate(prompt_ids, max_new_tokens=40, temperature=0.8, top_k=40)
    print(f"=== topic: {topic} ===")
    print("BASE:", tokenizer.decode(base_out[0].tolist()))
    print("SFT :", tokenizer.decode(sft_out[0].tolist()))
    print()
"""))

cells.append(md("""
### Question 3

Compare the BASE and SFT completions above. The base model was never trained
on the `"Write a short story about {topic}:\\n"` template at all — it will
either ignore the instruction and continue it as generic text, or coincidentally
produce something topical because TinyStories already contains that vocabulary.
Look specifically at whether the SFT model's response is more consistently
*about* the stated topic than the base model's. Does the held-out perplexity
number from TEST 3 alone tell you this, or did you need to read the actual
generations to know?

*Write your answer below:*

"""))

cells.append(code("""
ckpt_path = f"{CKPT_DIR}/sft_model.pt"
torch.save({'model_state_dict': sft_model.state_dict(), 'config': base_cfg}, ckpt_path)
print(f"Saved SFT checkpoint to {ckpt_path}")
"""))
```

- [ ] **Step 2: Regenerate and execute**

```bash
cd notebooks && ../.venv/bin/python build_llm_pipeline_02_sft_notebook.py && cd ..
.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 \
  notebooks/llm_training_pipeline/02_sft.ipynb
```

Expected: no errors; output includes `TEST 3 PASSED` and
`Saved SFT checkpoint to ../../data/checkpoints/llm_training_pipeline/sft_model.pt`.

- [ ] **Step 3: Verify the checkpoint loads**

```bash
.venv/bin/python -c "
import torch
ckpt = torch.load('data/checkpoints/llm_training_pipeline/sft_model.pt', weights_only=False)
n_params = sum(v.numel() for v in ckpt['model_state_dict'].values())
assert n_params == 13817856
print('sft_model.pt checkpoint OK, param count', n_params)
"
```

Expected: `sft_model.pt checkpoint OK, param count 13817856`

- [ ] **Step 4: Commit**

```bash
git add notebooks/build_llm_pipeline_02_sft_notebook.py notebooks/llm_training_pipeline/02_sft.ipynb
git commit -m "feat: llm_training_pipeline notebook 2 part 3 — SFT training loop"
```

---

## Task 7: Consolidate into `src/llm_pipeline/data.py`

**Files:**
- Modify: `src/llm_pipeline/data.py`

**Interfaces:**
- Consumes: the validated function bodies from Tasks 4-5 (copied verbatim).
- Produces: `from src.llm_pipeline.data import TOPIC_KEYWORDS, extract_topic, format_sft_prompt, build_sft_pairs, tokenize_sft_example`.

- [ ] **Step 1: Append to `src/llm_pipeline/data.py`**

Add to the end of `src/llm_pipeline/data.py` (after the existing `pack_into_blocks` function):

```python

TOPIC_KEYWORDS = [
    "dog", "cat", "girl", "boy", "forest", "ball", "tree", "bird", "star",
    "friend", "monster", "princess", "dragon", "robot", "garden", "park",
    "school", "castle", "rabbit", "mouse", "flower", "boat", "river",
    "mountain", "farm", "toy", "bear", "fish", "sun", "moon", "rain",
    "snow", "house", "family", "birthday", "picnic", "adventure", "magic",
    "kite", "puppy",
]


def extract_topic(story: str) -> str | None:
    """Returns the first TOPIC_KEYWORDS entry found in `story` (case-insensitive),
    or None if no keyword matches."""
    lower = story.lower()
    for kw in TOPIC_KEYWORDS:
        if kw in lower:
            return kw
    return None


def format_sft_prompt(topic: str) -> str:
    return f"Write a short story about {topic}:\n"


def build_sft_pairs(texts):
    """Returns a list of (topic, story) tuples for stories that matched a
    TOPIC_KEYWORDS entry; stories with no match are dropped."""
    pairs = []
    for t in texts:
        topic = extract_topic(t)
        if topic is not None:
            pairs.append((topic, t))
    return pairs


def tokenize_sft_example(topic: str, story: str, tokenizer, eot_id: int, block_size: int):
    """Builds a prompt-loss-masked (input_ids, labels) pair for SFT. Follows
    the same shift-by-one convention as pack_into_blocks: input_ids[i]
    predicts labels[i], the token that comes after input_ids[i]. labels[i]
    is -100 (ignore_index) whenever that target token falls inside the
    prompt or padding region."""
    prompt_ids = tokenizer.encode(format_sft_prompt(topic)).ids
    completion_ids = tokenizer.encode(story).ids + [eot_id]
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

- [ ] **Step 2: Smoke-test the additions**

```bash
.venv/bin/python -c "
from src.llm_pipeline.data import (
    TOPIC_KEYWORDS, extract_topic, format_sft_prompt, build_sft_pairs, tokenize_sft_example,
)
from tokenizers import ByteLevelBPETokenizer

assert extract_topic('A brave dog ran through the yard.') == 'dog'
assert extract_topic('Nothing topical here whatsoever today.') is None
assert format_sft_prompt('dragon') == 'Write a short story about dragon:\n'

texts = ['A dog played in the park.', 'The weather changed suddenly with no listed nouns.']
pairs = build_sft_pairs(texts)
assert len(pairs) == 1 and pairs[0][0] == 'dog'

tok = ByteLevelBPETokenizer()
tok.train(files=None, vocab_size=300) if False else None
print('src.llm_pipeline.data SFT helpers smoke test OK')
"
```

Expected: `src.llm_pipeline.data SFT helpers smoke test OK` with no assertion errors.
(`tokenize_sft_example` itself is already exercised end-to-end by notebook 2's
TEST 2 against the real trained tokenizer — this step only checks the
pure-Python topic-extraction helpers import and behave correctly in isolation.)

- [ ] **Step 3: Commit**

```bash
git add src/llm_pipeline/data.py
git commit -m "refactor: consolidate notebook 2's SFT dataset utilities into src/llm_pipeline/data.py"
```

---

## Self-Review Checklist

**1. Spec coverage** (against `docs/superpowers/specs/2026-07-02-llm-training-pipeline-learning-materials-design.md`):
- HTML reference Section 4 (SFT — prompt-loss-masking, why SFT unlocks instruction-following, catastrophic forgetting / low-LR rationale): Task 1. ✓
- Concepts Q&A additions for SFT: Task 2. ✓
- Notebook 2 (`02_sft.ipynb`): load `base_model.pt`, build instruction dataset, implement prompt-loss-masking with a synthetic-example test (known mask boundary + masked-loss ≠ full-loss), SFT training loop, qualitative + perplexity comparison of base vs SFT, saves `sft_model.pt`: Tasks 3-6. ✓
- Consolidation into `src/llm_pipeline/data.py`: Task 7. ✓
- SFT data format from spec (`"Write a short story about {topic}:\n" -> story`): used exactly as specified in `format_sft_prompt`. ✓

**2. Placeholder scan:** No "TBD"/"TODO"/"similar to Task N" patterns; every code step contains complete, runnable code; every HTML/Markdown step contains complete final content.

**3. Type/interface consistency:** `tokenize_sft_example(topic, story, tokenizer, eot_id, block_size) -> (LongTensor, LongTensor)` has an identical signature and body in the notebook (Task 5) and in `src/llm_pipeline/data.py` (Task 7). `format_sft_prompt` and `TOPIC_KEYWORDS` are identical across Task 4 (notebook) and Task 7 (`src/llm_pipeline/data.py`). `sft_model.pt`'s checkpoint dict shape (`model_state_dict`, `config`) matches `base_model.pt`'s shape from Part 1, which later plans (Parts 3-6) will rely on when loading either checkpoint interchangeably via `GPTModel(ckpt['config'])` + `load_state_dict(ckpt['model_state_dict'])`.
