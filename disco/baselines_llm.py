"""
LLM-based tabular SDG baselines (the architecture family we were missing):
  - GReaT          (be_great)      label "great"
  - REaLTabFormer  (realtabformer) label "realtabformer"

DISCO-TAB is itself GPT-2-based, so these are the "unconstrained LLM" comparison.

Run in the `discollm` conda env:
  conda run -n discollm python baselines_llm.py --datasets ... --models ... --seeds 0
Batch mode isolates each (dataset,model,seed) in a subprocess with a timeout and
a GPU->CPU fallback. Writes synthetic/baselines/<dataset>__<label>__seed<seed>.csv
(same schema as the other baselines), resumable.
"""
import os, sys, argparse, subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data_raw")
OUTDIR = os.path.join(HERE, "synthetic", "baselines")
os.makedirs(OUTDIR, exist_ok=True)

DEF_DATASETS = ["heart_failure", "obesity", "diabetes", "german_credit",
                "surv_gbsg2", "surv_pbc3", "tab_credit_g", "tab_qsar_biodeg"]
DEF_MODELS = ["great", "realtabformer"]
TRAIN_CAP = 6000


def split(ds, seed):
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(os.path.join(RAW, f"{ds}.csv"))
    df["target"] = df["target"].astype(str)
    try:
        tr, te = train_test_split(df, test_size=0.2, random_state=seed,
                                  stratify=df["target"])
    except ValueError:
        tr, te = train_test_split(df, test_size=0.2, random_state=seed)
    return tr.reset_index(drop=True), df


def align(syn, real_cols, target_classes):
    syn = syn.copy()
    for c in real_cols:
        if c not in syn.columns:
            syn[c] = np.nan
    syn = syn[real_cols]
    # coerce target to the real class set
    mode = pd.Series(list(target_classes)).mode()[0]
    syn["target"] = syn["target"].astype(str)
    syn.loc[~syn["target"].isin(target_classes), "target"] = mode
    return syn


def gen_great(train, n, epochs, device):
    from be_great import GReaT
    m = GReaT(llm="distilgpt2", epochs=epochs, batch_size=16,
              save_steps=10**9, logging_steps=10**9, experiment_dir="/tmp/great_tmp")
    m.fit(train)
    return m.sample(n_samples=n, device=device)


def gen_realtabformer(train, n, epochs, device):
    # torch>=2.6 defaults torch.load(weights_only=True), which rejects
    # REaLTabFormer's own checkpoint pickles. We trust our checkpoints -> force False.
    import torch
    _orig_load = torch.load
    torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
    from realtabformer import REaLTabFormer
    m = REaLTabFormer(model_type="tabular", epochs=epochs, batch_size=16,
                      checkpoints_dir="/tmp/rtf_tmp", samples_save_dir="/tmp/rtf_s")
    m.fit(train, device=device, num_bootstrap=10)
    return m.sample(n_samples=n, device=device)


def run_one(ds, model, seed):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, full = split(ds, seed)
    if len(tr) > TRAIN_CAP:
        tr = tr.sample(TRAIN_CAP, random_state=0).reset_index(drop=True)
    n = len(tr)
    epochs = 50 if n <= 2000 else 10
    fn = {"great": gen_great, "realtabformer": gen_realtabformer}[model]
    syn = fn(tr, n, epochs, device)
    syn = align(syn, list(full.columns), set(full["target"].astype(str)))
    out = os.path.join(OUTDIR, f"{ds}__{model}__seed{seed}.csv")
    syn.to_csv(out, index=False)
    print(f"[OK] {ds} {model} seed{seed} -> {syn.shape}", flush=True)
    # Free the /tmp (RAM-backed tmpfs!) checkpoint/sample dirs so finished runs
    # don't keep occupying memory (these accumulated to 12 GB before).
    import shutil
    for d in ("/tmp/great_tmp", "/tmp/rtf_tmp", "/tmp/rtf_s"):
        shutil.rmtree(d, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEF_DATASETS)
    ap.add_argument("--models", nargs="+", default=DEF_MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--one", nargs=3, default=None,
                    help="internal: DATASET MODEL SEED (single-run worker)")
    args = ap.parse_args()

    if args.one:
        ds, model, seed = args.one[0], args.one[1], int(args.one[2])
        run_one(ds, model, seed)
        return

    for ds in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                out = os.path.join(OUTDIR, f"{ds}__{model}__seed{seed}.csv")
                if os.path.exists(out):
                    print(f"skip (exists): {ds} {model} seed{seed}", flush=True)
                    continue
                base = [sys.executable, __file__, "--one", ds, model, str(seed)]
                ok = False
                for dev_env, tmo in (({}, 1200), ({"CUDA_VISIBLE_DEVICES": ""}, 2400)):
                    env = {**os.environ, **dev_env, "TOKENIZERS_PARALLELISM": "false"}
                    tag = "cpu" if dev_env else "gpu"
                    print(f">>> {ds} {model} seed{seed} [{tag}]", flush=True)
                    try:
                        r = subprocess.run(base, env=env, timeout=tmo)
                        if r.returncode == 0 and os.path.exists(out):
                            ok = True
                            break
                        print(f"    rc={r.returncode} on {tag}", flush=True)
                    except subprocess.TimeoutExpired:
                        print(f"    TIMEOUT on {tag}", flush=True)
                if not ok:
                    print(f"[FAIL] {ds} {model} seed{seed}", flush=True)


if __name__ == "__main__":
    main()
