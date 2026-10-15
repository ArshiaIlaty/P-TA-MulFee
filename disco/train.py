"""
DISCO-TAB: generator (GPT-2) + reward-shaped PPO with IFRS (Section IV-D/E).

Pipeline per dataset/seed:
  1. SFT  : GReaT-style supervised warm-start on serialized real records.
  2. PPO  : policy = SFT GPT-2; terminal reward R_total(S) = W_y * sum_k lambda_k D_k(S)
            from the hierarchical discriminators (co-trained each round);
            KL penalty to a frozen reference; a mean-pooled value-head baseline.
IFRS weight: W_y = 1 + alpha * max(0, log(N_total / (N_y * |Y|))).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2LMHeadModel


def build_generator(device):
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    return model, tok


# --------------------------- supervised warm-start ---------------------------
def sft(model, tok, texts, device, epochs=3, bs=16, lr=5e-5, max_len=128, log=print):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        perm = np.random.permutation(len(texts))
        tot = 0.0
        for i in range(0, len(texts), bs):
            batch = [texts[j] + tok.eos_token for j in perm[i:i + bs]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len).to(device)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            out = model(**enc, labels=labels)
            opt.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += out.loss.item()
        log(f"  [SFT] epoch {ep+1}/{epochs} loss={tot/max(1,len(texts)//bs):.4f}")
    model.eval()
    return model


# --------------------------- IFRS ---------------------------
def ifrs_weights(class_counts, alpha):
    N = sum(class_counts.values())
    Y = len(class_counts)
    w = {}
    for y, n in class_counts.items():
        w[str(y)] = 1.0 + alpha * max(0.0, math.log(N / (max(1, n) * Y)))
    return w


# --------------------------- value head ---------------------------
class ValueHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))

    def forward(self, pooled):
        return self.v(pooled).squeeze(-1)


# --------------------------- generation ---------------------------
@torch.no_grad()
def generate(model, tok, prompt, n, device, max_new=64, temperature=1.0,
             top_p=0.95, bs=64, repetition_penalty=1.0, no_repeat_ngram_size=0):
    model.eval()
    outs = []
    pids = tok(prompt, return_tensors="pt").input_ids.to(device)
    attn = torch.ones_like(pids)
    gkw = {}
    if no_repeat_ngram_size:
        gkw["no_repeat_ngram_size"] = no_repeat_ngram_size
    for i in range(0, n, bs):
        k = min(bs, n - i)
        batch = pids.repeat(k, 1)
        # Mild repetition control breaks the degenerate clause-repetition loop on
        # high-dimensional schemas with near-duplicate feature names, without the
        # off-distribution garbage a strong token penalty causes on tabular text.
        gen = model.generate(batch, attention_mask=attn.repeat(k, 1),
                             do_sample=True, temperature=temperature,
                             top_p=top_p, max_new_tokens=max_new,
                             repetition_penalty=repetition_penalty,
                             pad_token_id=tok.eos_token_id, **gkw)
        for row in gen:
            txt = tok.decode(row, skip_special_tokens=True)
            outs.append(txt.strip())
    return outs


# --------------------------- PPO ---------------------------
def _resp_logprobs(model, input_ids, attn, prompt_len):
    """Per-token logprobs for response tokens; returns (logp, mask, pooled_hidden)."""
    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    # Memory-efficient per-token log-prob: avoid materializing a full
    # (B, L, |V|) log-softmax tensor (OOM on long records / shared GPU).
    tok_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)    # (B,L-1)
    tok_logp = tok_logits - torch.logsumexp(logits, dim=-1)             # (B,L-1)
    L = input_ids.size(1)
    pos = torch.arange(L - 1, device=input_ids.device).unsqueeze(0)
    resp_mask = (pos >= (prompt_len - 1)).float() * attn[:, 1:].float()
    H = out.hidden_states[-1]                                           # (B,L,d)
    m = (attn.unsqueeze(-1).float())
    pooled = (H * m).sum(1) / m.sum(1).clamp(min=1.0)
    return tok_logp, resp_mask, pooled


class PPO:
    def __init__(self, policy, tok, ref, value, device, lr=1e-5, clip=0.2,
                 beta_kl=0.1, c_v=0.1, kl_target=0.5):
        self.policy, self.tok, self.ref, self.value = policy, tok, ref, value
        self.device, self.clip, self.beta_kl, self.c_v = device, clip, beta_kl, c_v
        self.kl_target = kl_target
        self.opt = torch.optim.AdamW(
            list(policy.parameters()) + list(value.parameters()), lr=lr)
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()

    def step(self, prompt, rewards_fn, batch=32, max_new=64, ppo_epochs=4,
             temperature=1.0, top_p=0.95, on_rollout=None):
        # --- rollout ---
        self.policy.eval()
        texts = generate(self.policy, self.tok, prompt, batch, self.device,
                         max_new=max_new, temperature=temperature, top_p=top_p,
                         bs=batch)
        # Co-train discriminators on this same rollout (reuses generation).
        if on_rollout is not None:
            on_rollout(texts)
        rewards, info = rewards_fn(texts)                      # (B,), dict
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=128).to(self.device)
        ids, attn = enc["input_ids"], enc["attention_mask"]
        prompt_len = self.tok(prompt, return_tensors="pt").input_ids.size(1)

        with torch.no_grad():
            old_logp, mask, _ = _resp_logprobs(self.policy, ids, attn, prompt_len)
            ref_logp, _, _ = _resp_logprobs(self.ref, ids, attn, prompt_len)

        # Center-only advantage: keeps magnitudes bounded (reward in [0, ~W_max]),
        # avoiding the variance amplification that destabilizes small-batch PPO.
        adv = rewards - rewards.mean()

        self.policy.train()
        self.value.train()
        last = {}
        for ep in range(ppo_epochs):
            logp, mask2, pooled = _resp_logprobs(self.policy, ids, attn, prompt_len)
            v = self.value(pooled)
            ratio = torch.exp((logp - old_logp) * mask2)
            a = adv.unsqueeze(1)
            s1 = ratio * a
            s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * a
            pg = -(torch.min(s1, s2) * mask2).sum() / mask2.sum().clamp(min=1)
            kl_tok = (logp - ref_logp)
            kl = ((kl_tok * mask2).sum() / mask2.sum().clamp(min=1))
            vloss = F.mse_loss(v, rewards)
            # Penalize drift away from the reference (positive KL direction).
            loss = pg + self.beta_kl * F.relu(kl) + self.c_v * vloss
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value.parameters()), 1.0)
            self.opt.step()
            last = {"pg": float(pg), "kl": float(kl), "vloss": float(vloss),
                    "loss": float(loss), "ppo_epochs_used": ep + 1}
            # Early stop if the policy has drifted too far this round.
            if float(kl) > self.kl_target:
                break
        self.policy.eval()
        last.update({"reward_mean": float(rewards.mean()),
                     "reward_std": float(rewards.std()) if rewards.numel() > 1 else 0.0})
        last.update(info)
        return last
