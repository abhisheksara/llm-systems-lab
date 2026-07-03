import torch
import torch.nn as nn
import torch.nn.functional as F

from src.llm_pipeline.model import GPTConfig, GPTModel, Block


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
        """idx: (B, T) padded token ids. lengths: (B,) real (unpadded) sequence
        lengths, or None to use position T-1 for every example."""
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
    """Copies token/position embeddings, transformer blocks, and final LayerNorm
    from an SFT GPTModel's state dict into a freshly-initialized RewardModel,
    leaving reward_head randomly initialized."""
    trunk_keys = [k for k in sft_state_dict if k.startswith(('tok_emb', 'pos_emb', 'blocks', 'ln_f'))]
    own_state = reward_model.state_dict()
    for k in trunk_keys:
        own_state[k].copy_(sft_state_dict[k])


def bradley_terry_loss(reward_chosen, reward_rejected):
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()


def encode_pair_text(prompt, completion, tokenizer, eot_id, block_size):
    """Returns (padded_ids: LongTensor[block_size], real_length: int)."""
    ids = (tokenizer.encode(prompt).ids + tokenizer.encode(completion).ids + [eot_id])[:block_size]
    length = len(ids)
    ids = ids + [eot_id] * (block_size - length)
    return torch.tensor(ids, dtype=torch.long), length


class PPOActorCritic(nn.Module):
    """Wraps a GPTModel, exposing both LM logits and a per-position scalar value
    estimate. Reuses the wrapped model's tok_emb/pos_emb/drop/blocks/ln_f/lm_head
    directly — no changes to GPTModel itself."""
    def __init__(self, gpt: GPTModel):
        super().__init__()
        self.gpt = gpt
        self.value_head = nn.Linear(gpt.config.n_embd, 1, bias=False)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.gpt.tok_emb(idx) + self.gpt.pos_emb(pos)
        x = self.gpt.drop(x)
        for block in self.gpt.blocks:
            x = block(x)
        x = self.gpt.ln_f(x)
        logits = self.gpt.lm_head(x)
        values = self.value_head(x).squeeze(-1)
        return logits, values


@torch.no_grad()
def generate_rollout(actor_critic, ref_model, prompt_ids, max_new_tokens, temperature, top_k, block_size):
    """Samples max_new_tokens autoregressively from actor_critic, recording the
    policy's log-prob, the frozen ref_model's log-prob, and the value estimate at
    each sampled token. Returns (idx, policy_logprobs, ref_logprobs, values)."""
    idx = prompt_ids.clone()
    policy_logprobs, ref_logprobs, values = [], [], []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, vals = actor_critic(idx_cond)
        logits_last = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits_last, top_k)
            logits_last[logits_last < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits_last, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        policy_lp = F.log_softmax(logits_last, dim=-1).gather(1, next_id).squeeze(-1)

        ref_logits, _ = ref_model(idx_cond)
        ref_lp = F.log_softmax(ref_logits[:, -1, :], dim=-1).gather(1, next_id).squeeze(-1)

        idx = torch.cat([idx, next_id], dim=1)
        policy_logprobs.append(policy_lp)
        ref_logprobs.append(ref_lp)
        values.append(vals[:, -1])
    return (
        idx,
        torch.stack(policy_logprobs, dim=1),
        torch.stack(ref_logprobs, dim=1),
        torch.stack(values, dim=1),
    )


def compute_token_rewards(policy_logprobs, ref_logprobs, terminal_reward, kl_beta):
    """Per-token reward = -kl_beta * KL at every step, plus terminal_reward added
    only at the last generated token. Returns (rewards, kl), both (B, T)."""
    kl = policy_logprobs - ref_logprobs
    rewards = -kl_beta * kl
    rewards = rewards.clone()
    rewards[:, -1] = rewards[:, -1] + terminal_reward
    return rewards, kl


def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """rewards, values: (B, T). Returns (advantages, returns), both (B, T)."""
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)
    next_value = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae
        next_value = values[:, t]
    returns = advantages + values
    return advantages, returns


def ppo_clipped_loss(new_logprobs, old_logprobs, advantages, clip_eps=0.2):
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()


def evaluate_actions(policy, idx, prompt_len, gen_len):
    """Re-runs policy over the full generated sequence, extracting log-probs and
    values at the positions/tokens that were sampled during the rollout."""
    logits, values = policy(idx[:, :-1])
    action_logits = logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :]
    action_values = values[:, prompt_len - 1 : prompt_len - 1 + gen_len]
    actions = idx[:, prompt_len : prompt_len + gen_len]
    logprobs = F.log_softmax(action_logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return logprobs, action_values


def sequence_logprob(model, input_ids, labels):
    """Returns (B,): sum of log pi(token) over only the non-masked (response)
    positions in each sequence — log pi(y|x) for the whole completion."""
    logits, _ = model(input_ids)
    logprobs = F.log_softmax(logits, dim=-1)
    mask = labels != -100
    safe_labels = labels.clone()
    safe_labels[~mask] = 0
    token_logprobs = logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs * mask
    return token_logprobs.sum(dim=-1)


def dpo_loss(policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta=0.1):
    pi_logratios = policy_chosen_lp - policy_rejected_lp
    ref_logratios = ref_chosen_lp - ref_rejected_lp
    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()


def compute_group_relative_advantage(rewards):
    """rewards: (B, G) — B prompts, G completions per prompt (one group per row).
    Returns advantages of the same shape: each completion's reward normalized by
    its own group's mean and std. No value function or baseline network needed —
    this is GRPO's replacement for GAE + a learned value function."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + 1e-4)
