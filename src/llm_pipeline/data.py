import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer


def load_tinystories(split: str = "train[:50000]"):
    """Returns a list of story strings from the TinyStories dataset."""
    ds = load_dataset("roneneldan/TinyStories", split=split)
    return [x["text"] for x in ds]


def train_bpe_tokenizer(texts, vocab_size: int, save_txt_path: str,
                         n_texts_for_training: int = 20000):
    """Trains a byte-level BPE tokenizer on a slice of `texts` and returns it,
    along with the id of the `<|endoftext|>` special token."""
    with open(save_txt_path, "w") as f:
        f.write("\n".join(texts[:n_texts_for_training]))

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[save_txt_path], vocab_size=vocab_size, min_frequency=2,
        special_tokens=["<|endoftext|>"],
    )
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    return tokenizer, eot_id


def pack_into_blocks(texts, tokenizer, eot_id: int, block_size: int) -> torch.Tensor:
    """Concatenates all texts (EOT-separated) into one token stream and chops
    it into fixed-length (block_size + 1) blocks. Returns a LongTensor of
    shape (n_blocks, block_size + 1)."""
    all_ids = []
    for t in texts:
        all_ids.extend(tokenizer.encode(t).ids)
        all_ids.append(eot_id)
    n_blocks = len(all_ids) // (block_size + 1)
    return torch.tensor(
        all_ids[: n_blocks * (block_size + 1)], dtype=torch.long
    ).view(n_blocks, block_size + 1)


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


def tokenize_prompt_response(prompt: str, response: str, tokenizer, eot_id: int, block_size: int):
    """Generalizes tokenize_sft_example to arbitrary prompt/response strings (not
    just the SFT topic template). Same prompt-loss-masking convention: a target
    token is masked (-100) iff it falls inside the prompt or padding region."""
    prompt_ids = tokenizer.encode(prompt).ids
    completion_ids = tokenizer.encode(response).ids + [eot_id]
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
