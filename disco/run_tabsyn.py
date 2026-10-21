"""
TabSyn (Zhang et al., ICLR 2024 Oral) baseline: a VAE that encodes mixed-type
tabular data into a continuous latent space + a score-based diffusion model over
the latents. We drive the official repo's pipeline per dataset
(process -> VAE train -> diffusion train -> sample), then align the sampled CSV
to our schema and write synthetic/baselines/<ds>__tabsyn__seed0.csv so
evaluate_baselines/evaluate_ext score it like every other generator.

Runs in `discobase` (+ category_encoders/ml_collections/icecream/prdc). The
per-dataset data/Info/<ds>.json + data/<ds>/<ds>.csv are already prepared.
Resumable (skips a dataset whose output CSV exists); one subprocess per stage
with a timeout.
"""
import os, sys, argparse, subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data_raw")
OUTDIR = os.path.join(HERE, "synthetic", "baselines")
TABSYN = "/home/ec2-user/disco_llm_models/tabsyn"
os.makedirs(OUTDIR, exist_ok=True)

DEF_DATASETS = ["heart_failure", "obesity", "diabetes", "surv_whas500", "surv_pbc3",
                "tab_maternal_risk", "breast_cancer", "liver_disorders", "parkinsons",
                "german_credit", "bank_marketing", "census_income", "surv_melanoma",
                "surv_pbc", "surv_retinopathy", "surv_gbsg2", "surv_colon", "surv_grace",
                "surv_trace", "surv_framingham", "tab_blood_transfusion",
                "tab_qsar_biodeg", "tab_website_phishing"]


def align(syn, real_cols, target_classes):
    syn = syn.copy()
    for c in real_cols:
        if c not in syn.columns:
            syn[c] = np.nan
    syn = syn[real_cols]
    mode = pd.Series(list(target_classes)).mode()[0]
    syn["target"] = syn["target"].astype(str)
    syn.loc[~syn["target"].isin(target_classes), "target"] = mode
    return syn


def run_one(ds, seed=0, timeout=3600):
    py = sys.executable
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")}

    def sh(args, tmo):
        subprocess.run([py] + args, cwd=TABSYN, env=env, check=True, timeout=tmo)

    sh(["process_dataset.py", "--dataname", ds], 300)
    sh(["main.py", "--dataname", ds, "--method", "vae", "--mode", "train"], timeout)
    sh(["main.py", "--dataname", ds, "--method", "tabsyn", "--mode", "train"], timeout)
    sh(["main.py", "--dataname", ds, "--method", "tabsyn", "--mode", "sample"], 900)

    syn = pd.read_csv(os.path.join(TABSYN, "synthetic", ds, "tabsyn.csv"))
    full = pd.read_csv(os.path.join(RAW, f"{ds}.csv"))
    full["target"] = full["target"].astype(str)
    syn = align(syn, list(full.columns), set(full["target"]))
    out = os.path.join(OUTDIR, f"{ds}__tabsyn__seed{seed}.csv")
    syn.to_csv(out, index=False)
    print(f"[OK] {ds} tabsyn seed{seed} -> {syn.shape}", flush=True)
    # Free this dataset's VAE/diffusion checkpoints + latents once the synthetic
    # CSV is saved, so finished runs don't accumulate (GBs across 23 datasets).
    import shutil
    for d in (os.path.join(TABSYN, "tabsyn", "ckpt", ds),
              os.path.join(TABSYN, "tabsyn", "vae", "ckpt", ds),
              os.path.join(TABSYN, "synthetic", ds)):
        shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEF_DATASETS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--one", default=None, help="internal: single DATASET")
    args = ap.parse_args()
    if args.one:
        run_one(args.one, args.seed)
        return
    for ds in args.datasets:
        out = os.path.join(OUTDIR, f"{ds}__tabsyn__seed{args.seed}.csv")
        if os.path.exists(out):
            print(f"skip (exists): {ds}", flush=True)
            continue
        print(f">>> {ds} tabsyn", flush=True)
        r = subprocess.run([sys.executable, __file__, "--one", ds, "--seed", str(args.seed)])
        if r.returncode != 0 or not os.path.exists(out):
            print(f"[FAIL] {ds} tabsyn rc={r.returncode}", flush=True)


if __name__ == "__main__":
    main()
