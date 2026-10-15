"""
DISCO-TAB: hierarchical multi-objective discriminator ensemble (Section IV-C).

Four discriminators, each a binary real/synthetic classifier at its granularity:
  D_token : own embedding + BiLSTM over GPT-2 token ids (syntactic validity)
  D_sent  : frozen DistilBERT encoder of each "col is val" clause + head
            (semantic plausibility; a lighter stand-in for the paper's BERT-base)
  D_feat  : MLP over concatenated generator value-token embeddings for each
            P_crit pair (dependency consistency)
  D_row   : MLP over mean-pooled generator hidden states (global coherence)

D_feat / D_row consume the GENERATOR's last-layer hidden states (the generator
is never updated here — hidden states are detached). Record scores D_k(S) in
[0,1] are aggregated by mean, exactly as the composite reward expects.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def clauses_of(text):
    return [seg.strip() for seg in text.split(",") if seg.strip()]


# ----------------------------- sub-networks -----------------------------
class TokenDiscriminator(nn.Module):
    def __init__(self, vocab, emb=128, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hid, 1)

    def forward(self, ids, mask):
        x = self.emb(ids)
        h, _ = self.lstm(x)
        tok = torch.sigmoid(self.head(h)).squeeze(-1)        # (B,T)
        m = mask.float()
        return (tok * m).sum(1) / m.sum(1).clamp(min=1.0)    # (B,)


class HeadMLP(nn.Module):
    def __init__(self, din, hid=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hid), nn.ReLU(),
                                 nn.Linear(hid, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


# ----------------------------- ensemble -----------------------------
class HierarchicalDiscriminators:
    def __init__(self, gen_tokenizer, pcrit, device="cuda", d_model=768,
                 sent_model="distilbert-base-uncased", max_len=128, lr=5e-4):
        self.device = device
        self.gen_tok = gen_tokenizer
        self.pcrit = [(a, b) for a, b, *_ in pcrit]
        self.max_len = max_len

        self.d_token = TokenDiscriminator(len(gen_tokenizer)).to(device)
        self.sent_tok = AutoTokenizer.from_pretrained(sent_model)
        self.sent_enc = AutoModel.from_pretrained(sent_model).to(device).eval()
        for p in self.sent_enc.parameters():
            p.requires_grad_(False)
        self.d_sent = HeadMLP(self.sent_enc.config.hidden_size).to(device)
        self.d_feat = HeadMLP(2 * d_model).to(device)
        self.d_row = HeadMLP(d_model).to(device)

        self.opt = torch.optim.AdamW(
            list(self.d_token.parameters()) + list(self.d_sent.parameters())
            + list(self.d_feat.parameters()) + list(self.d_row.parameters()),
            lr=lr)

    def train(self):
        for m in (self.d_token, self.d_sent, self.d_feat, self.d_row):
            m.train()

    def eval(self):
        for m in (self.d_token, self.d_sent, self.d_feat, self.d_row):
            m.eval()

    # ---- generator hidden states (detached) ----
    def _gen_hidden(self, gen_model, texts):
        enc = self.gen_tok(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=self.max_len,
                           return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = gen_model(**enc, output_hidden_states=True)
        H = out.hidden_states[-1].detach()                   # (B,T,d)
        return H, enc["attention_mask"], enc["input_ids"], offsets

    @staticmethod
    def _pool_span(H_b, offs_b, span):
        cs, ce = span
        idx = [t for t, (a, c) in enumerate(offs_b.tolist())
               if c > a and a < ce and c > cs]
        if not idx:
            return torch.zeros(H_b.size(-1), device=H_b.device)
        return H_b[idx].mean(0)

    def _feat_vectors(self, H, offsets, spans_list):
        """-> tensor (B, n_pairs, 2d) of concatenated value embeddings."""
        B = H.size(0)
        out = []
        for b in range(B):
            spans = spans_list[b]
            pairs = []
            for (fa, fb) in self.pcrit:
                ha = self._pool_span(H[b], offsets[b], spans[fa]) if fa in spans \
                    else torch.zeros(H.size(-1), device=H.device)
                hb = self._pool_span(H[b], offsets[b], spans[fb]) if fb in spans \
                    else torch.zeros(H.size(-1), device=H.device)
                pairs.append(torch.cat([ha, hb]))
            out.append(torch.stack(pairs))
        return torch.stack(out)                              # (B,P,2d)

    def _sent_feats(self, flat_clauses):
        enc = self.sent_tok(flat_clauses, return_tensors="pt", padding=True,
                            truncation=True, max_length=32).to(self.device)
        with torch.no_grad():
            h = self.sent_enc(**enc).last_hidden_state       # (n,t,d)
        m = enc["attention_mask"].unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)      # (n,d)

    # ---- scoring (reward): no grad, returns numpy per-record component scores ----
    @torch.no_grad()
    def score(self, gen_model, serializer, texts):
        self.eval()
        H, mask, ids, offsets = self._gen_hidden(gen_model, texts)
        spans_list = [serializer.value_spans(t) for t in texts]

        tok = self.d_token(ids, mask).cpu().numpy()
        mpool = (H * mask.unsqueeze(-1).float()).sum(1) / \
                mask.sum(1, keepdim=True).clamp(min=1).float()
        row = self.d_row(mpool).cpu().numpy()
        fv = self._feat_vectors(H, offsets, spans_list)      # (B,P,2d)
        feat = self.d_feat(fv.reshape(-1, fv.size(-1))).reshape(fv.size(0), -1)
        feat = feat.mean(1).cpu().numpy()

        clause_lists = [clauses_of(t) for t in texts]
        flat, counts = [], []
        for cl in clause_lists:
            cl = cl if cl else [""]
            flat.extend(cl)
            counts.append(len(cl))
        sfeat = self._sent_feats(flat)
        sscore = self.d_sent(sfeat).cpu().numpy()
        sent, k = [], 0
        for c in counts:
            sent.append(float(sscore[k:k + c].mean()))
            k += c
        return {"token": tok, "sent": np.array(sent), "feat": feat, "row": row}

    # ---- one discriminator training step on real(=1) vs synthetic(=0) ----
    def train_step(self, gen_model, serializer, real_texts, syn_texts):
        self.train()
        self.opt.zero_grad()
        loss = torch.zeros((), device=self.device)
        logs = {}

        # D_token + D_row + D_feat share generator hidden states.
        for label, texts in ((1.0, real_texts), (0.0, syn_texts)):
            H, mask, ids, offsets = self._gen_hidden(gen_model, texts)
            spans_list = [serializer.value_spans(t) for t in texts]
            y = torch.full((len(texts),), label, device=self.device)

            tok = self.d_token(ids, mask)
            mpool = (H * mask.unsqueeze(-1).float()).sum(1) / \
                    mask.sum(1, keepdim=True).clamp(min=1).float()
            row = self.d_row(mpool)
            fv = self._feat_vectors(H, offsets, spans_list)
            feat = self.d_feat(fv.reshape(-1, fv.size(-1))).reshape(fv.size(0), -1).mean(1)

            lt = F.binary_cross_entropy(tok.clamp(1e-6, 1 - 1e-6), y)
            lr_ = F.binary_cross_entropy(row.clamp(1e-6, 1 - 1e-6), y)
            lf = F.binary_cross_entropy(feat.clamp(1e-6, 1 - 1e-6), y)
            loss = loss + lt + lr_ + lf
            logs[f"token_{int(label)}"] = float(lt)
            logs[f"row_{int(label)}"] = float(lr_)
            logs[f"feat_{int(label)}"] = float(lf)

        # D_sent per-clause.
        for label, texts in ((1.0, real_texts), (0.0, syn_texts)):
            flat = []
            for t in texts:
                flat.extend(clauses_of(t) or [""])
            sfeat = self._sent_feats(flat)
            s = self.d_sent(sfeat)
            y = torch.full((len(flat),), label, device=self.device)
            ls = F.binary_cross_entropy(s.clamp(1e-6, 1 - 1e-6), y)
            loss = loss + ls
            logs[f"sent_{int(label)}"] = float(ls)

        loss.backward()
        self.opt.step()
        logs["total"] = float(loss)
        return logs

    def save(self, path):
        torch.save({"token": self.d_token.state_dict(),
                    "sent": self.d_sent.state_dict(),
                    "feat": self.d_feat.state_dict(),
                    "row": self.d_row.state_dict(),
                    "pcrit": self.pcrit}, path)

    def load(self, path):
        sd = torch.load(path, map_location=self.device)
        self.d_token.load_state_dict(sd["token"])
        self.d_sent.load_state_dict(sd["sent"])
        self.d_feat.load_state_dict(sd["feat"])
        self.d_row.load_state_dict(sd["row"])


if __name__ == "__main__":
    from data import load_dataset
    from serialize import Serializer
    from constraints import discover_constraints
    from transformers import AutoTokenizer, GPT2LMHeadModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = load_dataset("heart_failure", seed=0)
    ser = Serializer(b.num_cols, b.cat_cols, b.target).fit(b.df_train)
    pcrit, _ = discover_constraints(b.df_train, b.num_cols, b.cat_cols)
    gtok = AutoTokenizer.from_pretrained("gpt2")
    gtok.pad_token = gtok.eos_token
    gmodel = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
    disc = HierarchicalDiscriminators(gtok, pcrit, device=dev)

    real = ser.serialize_df(b.df_train.iloc[:8])
    syn = [t.replace("is", "is", 1) for t in real]  # trivial "synthetic"
    print("scores (untrained):", {k: round(float(v.mean()), 3)
                                   for k, v in disc.score(gmodel, ser, real).items()})
    for i in range(5):
        logs = disc.train_step(gmodel, ser, real, syn[:8])
    print("after 5 steps, total loss:", round(logs["total"], 4))
    print("OK discriminators")
