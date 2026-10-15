"""
DISCO-TAB: end-to-end run for one (dataset, seed).

  SFT warm-start -> constraint discovery -> discriminator pretrain ->
  reward-shaped PPO (co-training discriminators) -> generate -> evaluate.

Writes synthetic/<dataset>__disco__seed<seed>.csv and
results/<dataset>__disco__seed<seed>.json. Also supports ablation configs.
"""
import os, sys, json, time, copy, argparse, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))

from data import load_dataset
from serialize import Serializer
from constraints import discover_constraints
from discriminators import HierarchicalDiscriminators, clauses_of
from train import build_generator, sft, ifrs_weights, ValueHead, PPO, generate
import metrics as M

DEFAULTS = dict(
    lambda_token=0.25, lambda_sent=0.25, lambda_feat=0.25, lambda_row=0.25,
    alpha=0.2, delta_thresh=0.3, top_k=8,
    sft_epochs=40, sft_auto=True, sft_lr=5e-5,
    ppo_steps=40, ppo_batch=16, ppo_lr=5e-6, clip=0.1, beta_kl=0.5, c_v=0.1,
    kl_target=0.5, ppo_epochs=2, disc_lr=2e-4, disc_pretrain=40, disc_pool=512,
    disc_every=2, max_train_sft=3000, gen_cap=3000, temperature=0.9, top_p=0.95,
    use_ifrs=True, use_token=True, use_sent=True, use_feat=True, use_row=True,
)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def seconds():
    return time.time()


def run_disco(name, seed=0, cfg=None, tag="disco", log=print):
    cfg = {**DEFAULTS, **(cfg or {})}
    set_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = seconds()
    b = load_dataset(name, seed=seed)
    log(f"[{name} seed{seed}] {b}")

    ser = Serializer(b.num_cols, b.cat_cols, b.target).fit(b.df_train)
    train_df = b.df_train
    if len(train_df) > cfg["max_train_sft"]:
        train_df = train_df.sample(cfg["max_train_sft"], random_state=seed)
    real_texts = ser.serialize_df(train_df)

    model, tok = build_generator(dev)
    # token budget for generation = a bit beyond the longest real record
    sample = real_texts[:300]
    tok_lens = [len(tok(t).input_ids) for t in sample]
    # Need enough budget to reach the (last) target clause even on the longest,
    # highest-dimensional records; headroom for the LM rendering longer values.
    max_new = int(max(tok_lens) * 1.5) + 12
    max_new = min(max_new, 400)
    # Adaptive SFT epochs: tiny datasets need many passes, large ones few.
    if cfg.get("sft_auto", True):
        n = len(train_df)
        epochs = 50 if n <= 500 else (30 if n <= 2000 else 10)
    else:
        epochs = cfg["sft_epochs"]
    log(f"[{name}] SFT on {len(real_texts)} records, epochs={epochs}, max_new={max_new}")
    sft(model, tok, real_texts, dev, epochs=epochs, lr=cfg["sft_lr"], log=log)

    pcrit, _ = discover_constraints(b.df_train, b.num_cols, b.cat_cols,
                                    cfg["delta_thresh"], cfg["top_k"])
    log(f"[{name}] P_crit: {[(a,c) for a,c,*_ in pcrit]}")
    disc = HierarchicalDiscriminators(tok, pcrit, device=dev, lr=cfg["disc_lr"])

    ref = copy.deepcopy(model).to(dev)
    value = ValueHead(model.config.n_embd).to(dev)
    ppo = PPO(model, tok, ref, value, dev, lr=cfg["ppo_lr"], clip=cfg["clip"],
              beta_kl=cfg["beta_kl"], c_v=cfg["c_v"], kl_target=cfg["kl_target"])

    ifrs_w = ifrs_weights(b.class_counts, cfg["alpha"]) if cfg["use_ifrs"] else {}
    lam = {"token": cfg["lambda_token"] * cfg["use_token"],
           "sent": cfg["lambda_sent"] * cfg["use_sent"],
           "feat": cfg["lambda_feat"] * cfg["use_feat"],
           "row": cfg["lambda_row"] * cfg["use_row"]}
    lam_sum = sum(lam.values()) or 1.0
    prompt = f"{ser.order[0]} is"

    def reward_fn(texts):
        sc = disc.score(model, ser, texts)
        raw = sum(lam[k] * sc[k] for k in lam) / lam_sum
        W = np.ones(len(texts))
        if cfg["use_ifrs"]:
            for i, t in enumerate(texts):
                d = ser.parse_text(t)
                lab = str(d.get(b.target))
                W[i] = ifrs_w.get(lab, 1.0)
        total = raw * W
        return total.tolist(), {f"D_{k}": float(np.mean(sc[k])) for k in sc}

    # --- discriminator pretrain (real vs SFT samples) ---
    def sample_real(nbatch):
        idx = np.random.randint(0, len(real_texts), size=nbatch)
        return [real_texts[i] for i in idx]

    def sample_syn(pool, nbatch):
        idx = np.random.randint(0, len(pool), size=nbatch)
        return [pool[i] for i in idx]

    if cfg["ppo_steps"] > 0 and cfg["disc_pretrain"] > 0:
        log(f"[{name}] building synthetic pool ({cfg['disc_pool']}) + pretraining "
            f"discriminators ({cfg['disc_pretrain']} steps)")
        pool = generate(model, tok, prompt, cfg["disc_pool"], dev, max_new=max_new,
                        temperature=cfg["temperature"], top_p=cfg["top_p"], bs=64)
        for _ in range(cfg["disc_pretrain"]):
            disc.train_step(model, ser, sample_real(cfg["ppo_batch"]),
                            sample_syn(pool, cfg["ppo_batch"]))

    # --- PPO loop; discriminators co-trained on the rollout every disc_every steps ---
    step_ctr = {"i": 0}

    def on_rollout(texts):
        if step_ctr["i"] % cfg["disc_every"] == 0:
            disc.train_step(model, ser, sample_real(len(texts)), texts)

    log(f"[{name}] PPO {cfg['ppo_steps']} steps (batch {cfg['ppo_batch']})")
    hist = []
    for step in range(cfg["ppo_steps"]):
        step_ctr["i"] = step
        logs = ppo.step(prompt, reward_fn, batch=cfg["ppo_batch"], max_new=max_new,
                        ppo_epochs=cfg["ppo_epochs"], temperature=cfg["temperature"],
                        top_p=cfg["top_p"], on_rollout=on_rollout)
        hist.append(logs)
        if step % 20 == 0 or step == cfg["ppo_steps"] - 1:
            log(f"   step {step:4d} R={logs['reward_mean']:.3f} kl={logs['kl']:.3f} "
                f"pg={logs['pg']:.3f} | Dtok={logs.get('D_token',0):.2f} "
                f"Dsent={logs.get('D_sent',0):.2f} Dfeat={logs.get('D_feat',0):.2f} "
                f"Drow={logs.get('D_row',0):.2f}")

    # --- final generation + parse ---
    n_gen = min(len(b.df_train), cfg["gen_cap"])
    log(f"[{name}] generating {n_gen} synthetic records")
    raw_texts = generate(model, tok, prompt, int(n_gen * 1.4) + 20, dev,
                         max_new=max_new, temperature=cfg["temperature"],
                         top_p=cfg["top_p"], bs=64)
    df_syn = ser.parse_batch(raw_texts)
    df_syn = df_syn.head(n_gen)
    valid_rate = len(df_syn) / max(1, len(raw_texts))
    log(f"[{name}] parsed {len(df_syn)} valid rows (valid rate {valid_rate:.2f})")

    os.makedirs(os.path.join(HERE, "synthetic"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    syn_path = os.path.join(HERE, "synthetic", f"{name}__{tag}__seed{seed}.csv")
    df_syn.to_csv(syn_path, index=False)

    res = {"dataset": name, "seed": seed, "tag": tag, "n_syn": len(df_syn),
           "valid_rate": valid_rate, "pcrit": [(a, c) for a, c, *_ in pcrit],
           "max_new": max_new, "runtime_s": round(seconds() - t0, 1),
           "config": {k: cfg[k] for k in cfg},
           "reward_curve": [h["reward_mean"] for h in hist]}
    if len(df_syn) >= 10:
        res.update(M.evaluate_all(b, df_syn, seed=seed))
        res["trtr_f1_macro"] = M.trtr_f1(b, seed=seed)["f1_macro"]
    out_path = os.path.join(HERE, "results", f"{name}__{tag}__seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=float)
    log(f"[{name} seed{seed}] DONE in {res['runtime_s']}s  "
        f"TSTR_F1_macro={res.get('tstr_f1_macro')}  "
        f"JS={res.get('fid_JS')}  DCR={res.get('dcr_median')}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="heart_failure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="disco")
    ap.add_argument("--ppo_steps", type=int, default=None)
    ap.add_argument("--ppo_batch", type=int, default=None)
    ap.add_argument("--sft_epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = {}
    # Ablation / override config passed by run_all.py through the environment.
    env_cfg = os.environ.get("DISCO_CFG")
    if env_cfg:
        cfg.update(json.loads(env_cfg))
    for k in ("ppo_steps", "ppo_batch", "sft_epochs"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.sft_epochs is not None:
        cfg["sft_auto"] = False
    run_disco(args.dataset, args.seed, cfg, tag=args.tag)
