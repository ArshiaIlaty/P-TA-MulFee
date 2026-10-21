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
from calibrate import calibrate_numerics
import metrics as M
import pandas as pd

DEFAULTS = dict(
    lambda_token=0.25, lambda_sent=0.25, lambda_feat=0.25, lambda_row=0.25,
    alpha=0.2, delta_thresh=0.3, top_k=8,
    sft_epochs=40, sft_auto=True, sft_lr=5e-5,
    ppo_steps=40, ppo_batch=16, ppo_lr=5e-6, clip=0.1, beta_kl=0.5, c_v=0.1,
    kl_target=0.5, ppo_epochs=2, disc_lr=2e-4, disc_pretrain=40, disc_pool=512,
    disc_every=2, max_train_sft=3000, gen_cap=3000, temperature=0.9, top_p=0.95,
    use_ifrs=True, use_token=True, use_sent=True, use_feat=True, use_row=True,
    # --- revision improvements ---
    # (1) calibration shipped ON (safe fidelity win at ~neutral utility).
    # (2)+(3) reward terms kept in code but OFF by default: at w=0.3 they helped
    # fidelity + the weakest datasets but regressed utility on strong datasets
    # (obesity/WHAS500); re-enable with lower weights after tuning.
    use_calibration=True,                 # (1) post-hoc quantile/copula calibration
    use_distmatch=False, w_dist=0.3,      # (2) per-row distribution+correlation (Mahalanobis) reward
    use_utility=False, w_util=0.3,        # (3) per-row utility (frozen real-classifier) reward
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

    # --- precompute reward-shaping targets (improvements 2 & 3) ---
    # (2) real numeric mean + (regularized) covariance for a per-row Mahalanobis
    #     "distribution+correlation" fit reward; (3) a frozen classifier trained
    #     on real train for a per-row downstream-utility (label-consistency) reward.
    num = b.num_cols
    mu = cov_inv = None
    if cfg["use_distmatch"] and num:
        Xr = b.df_train[num].apply(pd.to_numeric, errors="coerce")
        mu = Xr.mean()
        Xr = Xr.fillna(mu).values.astype(float)
        mu = mu.values.astype(float)
        if len(num) > 1:
            cov = np.cov(Xr.T) + np.eye(len(num)) * 1e-3
        else:
            cov = np.array([[Xr.var() + 1e-6]])
        cov_inv = np.linalg.pinv(np.atleast_2d(cov))
    util_clf = util_cls_idx = None
    if cfg["use_utility"]:
        from sklearn.ensemble import RandomForestClassifier
        Xtr, ytr = b.pp.xy(b.df_train)
        if len(set(ytr)) > 1:
            util_clf = RandomForestClassifier(n_estimators=100, random_state=seed,
                                              class_weight="balanced", n_jobs=-1).fit(Xtr, ytr)
            util_cls_idx = {c: i for i, c in enumerate(list(util_clf.classes_))}

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
        n = len(texts)
        dicts = [ser.parse_text(t) for t in texts]
        W = np.ones(n)
        if cfg["use_ifrs"]:
            for i, d in enumerate(dicts):
                W[i] = ifrs_w.get(str(d.get(b.target)), 1.0)
        total = raw * W
        info = {f"D_{k}": float(np.mean(sc[k])) for k in sc}

        # (2) per-row Mahalanobis fit to real (mean, covariance): rewards rows that
        #     respect the real marginals AND correlation structure.
        if cov_inv is not None:
            dist_r = np.zeros(n)
            for i, d in enumerate(dicts):
                x = np.array([float(d[c]) if (c in d and d.get(c) is not None
                              and not pd.isna(d.get(c))) else np.nan for c in num], float)
                mk = np.isnan(x); x[mk] = mu[mk]
                diff = x - mu
                dist_r[i] = np.exp(-float(diff @ cov_inv @ diff) / (2.0 * len(num)))
            total = total + cfg["w_dist"] * dist_r
            info["D_dist"] = float(dist_r.mean())

        # (3) per-row utility: frozen real-trained classifier's probability of the
        #     row's own claimed label -> rewards label-consistent (useful) rows.
        if util_clf is not None:
            dfb = pd.DataFrame(dicts)
            for c in ser.order:
                if c not in dfb.columns:
                    dfb[c] = np.nan
            try:
                proba = util_clf.predict_proba(b.pp.transform_X(dfb))
                yb = b.pp.transform_y(dfb)
                util_r = np.array([proba[i, util_cls_idx[y]]
                                   if (y >= 0 and y in util_cls_idx) else 0.0
                                   for i, y in enumerate(yb)])
                total = total + cfg["w_util"] * util_r
                info["D_util"] = float(util_r.mean())
            except Exception:
                pass
        return total.tolist(), info

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
                f"Drow={logs.get('D_row',0):.2f} Ddist={logs.get('D_dist',0):.2f} "
                f"Dutil={logs.get('D_util',0):.2f}")

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
    # (1) post-hoc class-conditional quantile/copula calibration of numerics
    if cfg["use_calibration"] and len(df_syn) >= 10:
        df_syn = calibrate_numerics(df_syn, b.df_train, b.num_cols, target=b.target)
        log(f"[{name}] applied numeric calibration ({len(b.num_cols)} cols)")

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
