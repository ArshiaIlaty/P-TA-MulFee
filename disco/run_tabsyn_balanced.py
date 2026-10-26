"""
Class-conditional TabSyn for the datasets where vanilla (unconditional) TabSyn
collapses the minority class to F1=0. TabSyn models the joint latent with no
label conditioning, so on imbalanced data the argmax target decode returns only
the majority class. Fix: generate PER CLASS. For each target class c we train a
separate TabSyn on that class's rows (oversampling tiny classes to a floor so
the VAE/diffusion can train), sample a count proportional to the real class
frequency, force target=c, and concatenate. This makes TabSyn a fair,
non-collapsing baseline. Writes synthetic/baselines/<ds>__tabsynbal__seed0.csv
(a NEW tag, so the original tabsyn runs are preserved for comparison).
"""
import os, sys, json, shutil, argparse, subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data_raw")
OUTDIR = os.path.join(HERE, "synthetic", "baselines")
TABSYN = "/home/ec2-user/disco_llm_models/tabsyn"
MINROWS = 80  # oversample a class subset up to this many rows before training

COLLAPSED = ["heart_failure", "breast_cancer", "liver_disorders", "parkinsons",
             "surv_framingham", "surv_grace", "surv_melanoma", "surv_trace",
             "surv_whas500", "tab_blood_transfusion", "tab_qsar_biodeg"]


def _prep_class(ds, ci, rows):
    """Write per-class data + Info under a derived dataname; return dataname."""
    dn = f"{ds}__bal{ci}"
    ddir = os.path.join(TABSYN, "data", dn)
    os.makedirs(ddir, exist_ok=True)
    rows.to_csv(os.path.join(ddir, f"{dn}.csv"), index=False)
    info = json.load(open(os.path.join(TABSYN, "data", "Info", f"{ds}.json")))
    info["name"] = dn
    info["data_path"] = f"data/{dn}/{dn}.csv"
    json.dump(info, open(os.path.join(TABSYN, "data", "Info", f"{dn}.json"), "w"))
    return dn


def _run_pipeline(dn, py, env, timeout):
    def sh(args, tmo):
        subprocess.run([py] + args, cwd=TABSYN, env=env, check=True, timeout=tmo)
    sh(["process_dataset.py", "--dataname", dn], 300)
    sh(["main.py", "--dataname", dn, "--method", "vae", "--mode", "train"], timeout)
    sh(["main.py", "--dataname", dn, "--method", "tabsyn", "--mode", "train"], timeout)
    sh(["main.py", "--dataname", dn, "--method", "tabsyn", "--mode", "sample"], 900)
    return pd.read_csv(os.path.join(TABSYN, "synthetic", dn, "tabsyn.csv"))


def _cleanup(dn):
    for d in (os.path.join(TABSYN, "tabsyn", "ckpt", dn),
              os.path.join(TABSYN, "tabsyn", "vae", "ckpt", dn),
              os.path.join(TABSYN, "synthetic", dn),
              os.path.join(TABSYN, "data", dn)):
        shutil.rmtree(d, ignore_errors=True)
    f = os.path.join(TABSYN, "data", "Info", f"{dn}.json")
    if os.path.exists(f):
        os.remove(f)


def run_one(ds, seed=0, timeout=3600):
    py = sys.executable
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")}
    full = pd.read_csv(os.path.join(RAW, f"{ds}.csv"))
    full["target"] = full["target"].astype(str)
    tcol = "target"
    counts = full[tcol].value_counts()
    parts = []
    for ci, (c, n_c) in enumerate(counts.items()):
        rows = full[full[tcol] == c].copy()
        if len(rows) < MINROWS:   # oversample tiny classes so the diffusion can train
            rows = rows.sample(MINROWS, replace=True, random_state=seed).reset_index(drop=True)
        dn = _prep_class(ds, ci, rows)
        try:
            syn_c = _run_pipeline(dn, py, env, timeout)
        except Exception as e:
            print(f"[WARN] {ds} class {c}: pipeline failed ({e}); using real-subset resample", flush=True)
            syn_c = full[full[tcol] == c].sample(int(n_c), replace=True, random_state=seed)
        finally:
            _cleanup(dn)
        syn_c = syn_c.copy()
        syn_c[tcol] = c                       # force the class label
        syn_c = syn_c.sample(min(int(n_c), len(syn_c)), replace=(len(syn_c) < n_c),
                             random_state=seed)
        parts.append(syn_c)
    syn = pd.concat(parts, ignore_index=True)
    for col in full.columns:
        if col not in syn.columns:
            syn[col] = np.nan
    syn = syn[full.columns]
    out = os.path.join(OUTDIR, f"{ds}__tabsynbal__seed{seed}.csv")
    syn.to_csv(out, index=False)
    print(f"[OK] {ds} tabsynbal -> {syn.shape}, classes={sorted(syn[tcol].unique())}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=COLLAPSED)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--one", default=None)
    args = ap.parse_args()
    if args.one:
        run_one(args.one, args.seed)
        return
    for ds in args.datasets:
        out = os.path.join(OUTDIR, f"{ds}__tabsynbal__seed{args.seed}.csv")
        if os.path.exists(out):
            print(f"skip (exists): {ds}", flush=True); continue
        print(f">>> {ds} tabsynbal", flush=True)
        subprocess.run([sys.executable, __file__, "--one", ds, "--seed", str(args.seed)])


if __name__ == "__main__":
    main()
