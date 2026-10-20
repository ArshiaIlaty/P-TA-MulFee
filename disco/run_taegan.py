"""
TAEGAN (Li et al., "Revisit GANs for Tabular Data Generation", ACML 2025)
tabular GAN baseline, label "taegan".

TAEGAN's released code deliberately omits data preprocessing (`prepare`) and the
inverse transform (`recover`) -- the authors direct users to reconstruct them
from the open-source CTAB-GAN+/CTGAN pipeline. We do exactly that with CTGAN's
DataTransformer (mode-specific VGM normalization for continuous columns +
one-hot for categorical), which yields precisely the (width, "normal"/"onehot")
span-info TAEGAN's networks expect: a continuous column -> (1,"normal") mode
value in (-1,1) [matches TAEGAN's tanh activation] + (k,"onehot") mode select
[matches TAEGAN's hard gumbel-softmax], a categorical -> (n,"onehot"). After
sampling we invert the same transformer back to a CSV. The GAN itself
(networks.py / taegan.py / dataset.py) is used unmodified.

Run in `discobase` (ctgan 0.12 + torch cu13). Mirrors baselines_llm.py:
  conda run -n discobase python run_taegan.py --datasets ... --seeds 0
Writes synthetic/baselines/<ds>__taegan__seed<seed>.csv (same schema as the
other baselines), resumable, one isolated subprocess per (dataset,seed).
"""
import os, sys, json, argparse, tempfile, shutil, subprocess
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data_raw")
OUTDIR = os.path.join(HERE, "synthetic", "baselines")
TAEGAN_REPO = "/home/ec2-user/disco_llm_models/taegan"
os.makedirs(OUTDIR, exist_ok=True)

DEF_DATASETS = ["heart_failure", "obesity", "diabetes", "surv_whas500",
                "surv_pbc3", "tab_maternal_risk"]
TRAIN_CAP = 6000

sys.path.insert(0, HERE)
from data import infer_types, TARGET


def split(ds, seed):
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(os.path.join(RAW, f"{ds}.csv"))
    df[TARGET] = df[TARGET].astype(str)
    try:
        tr, te = train_test_split(df, test_size=0.2, random_state=seed,
                                  stratify=df[TARGET])
    except ValueError:
        tr, te = train_test_split(df, test_size=0.2, random_state=seed)
    return tr.reset_index(drop=True), df


def align(syn, real_cols, target_classes):
    syn = syn.copy()
    for c in real_cols:
        if c not in syn.columns:
            syn[c] = np.nan
    syn = syn[real_cols]
    mode = pd.Series(list(target_classes)).mode()[0]
    syn[TARGET] = syn[TARGET].astype(str)
    syn.loc[~syn[TARGET].isin(target_classes), TARGET] = mode
    return syn


def build_span_info(transformer):
    """CTGAN output_info_list -> TAEGAN span-info: tanh->normal, softmax->onehot."""
    span = []
    for col_info in transformer.output_info_list:
        for s in col_info:
            kind = "normal" if s.activation_fn == "tanh" else "onehot"
            span.append([int(s.dim), kind])
    return span


def run_one(ds, model, seed):
    import torch
    from ctgan.data_transformer import DataTransformer
    sys.path.insert(0, TAEGAN_REPO)
    from taegan import TAEGAN

    tr, full = split(ds, seed)
    if len(tr) > TRAIN_CAP:
        tr = tr.sample(TRAIN_CAP, random_state=0).reset_index(drop=True)
    n = len(tr)
    num_cols, cat_cols = infer_types(tr, TARGET)
    discrete = list(cat_cols) + [TARGET]
    train_df = tr.copy()
    for c in discrete:
        train_df[c] = train_df[c].astype(str)

    # --- reconstructed `prepare` (CTGAN transformer == CTAB-GAN+ style) ---
    transformer = DataTransformer()
    transformer.fit(train_df, discrete_columns=discrete)
    mat = np.asarray(transformer.transform(train_df), dtype=np.float32)
    span = build_span_info(transformer)
    assert sum(w for w, _ in span) == mat.shape[1], \
        f"span width {sum(w for w,_ in span)} != data_dim {mat.shape[1]}"

    cache = tempfile.mkdtemp(prefix=f"taegan_{ds}_s{seed}_")
    try:
        torch.save(torch.from_numpy(mat), os.path.join(cache, "train-data.pt"))
        with open(os.path.join(cache, "span-info.json"), "w") as f:
            json.dump(span, f)
        bs = int(min(500, max(50, n)))
        warmup = 30 if n <= 3000 else 15
        epochs = 150 if n <= 3000 else 60
        m = TAEGAN(cache)
        m.train(batch_size=bs, epochs=epochs, warmup_epochs=warmup)
        samp = m.generate(n=n, batch_size=int(min(500, n)))        # (n, data_dim)
        # --- reconstructed `recover` (inverse transform -> CSV) ---
        syn = transformer.inverse_transform(samp.numpy())
        syn = align(syn, list(full.columns), set(full[TARGET].astype(str)))
        out = os.path.join(OUTDIR, f"{ds}__{model}__seed{seed}.csv")
        syn.to_csv(out, index=False)
        print(f"[OK] {ds} {model} seed{seed} -> {syn.shape}", flush=True)
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEF_DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--one", nargs=2, default=None,
                    help="internal: DATASET SEED (single-run worker)")
    args = ap.parse_args()

    if args.one:
        run_one(args.one[0], "taegan", int(args.one[1]))
        return

    for ds in args.datasets:
        for seed in args.seeds:
            out = os.path.join(OUTDIR, f"{ds}__taegan__seed{seed}.csv")
            if os.path.exists(out):
                print(f"skip (exists): {ds} taegan seed{seed}", flush=True)
                continue
            base = [sys.executable, __file__, "--one", ds, str(seed)]
            ok = False
            for dev_env, tmo in (({}, 2400), ({"CUDA_VISIBLE_DEVICES": ""}, 5400)):
                env = {**os.environ, **dev_env, "TOKENIZERS_PARALLELISM": "false"}
                tag = "cpu" if dev_env else "gpu"
                print(f">>> {ds} taegan seed{seed} [{tag}]", flush=True)
                try:
                    r = subprocess.run(base, env=env, timeout=tmo)
                    if r.returncode == 0 and os.path.exists(out):
                        ok = True
                        break
                    print(f"    rc={r.returncode} on {tag}", flush=True)
                except subprocess.TimeoutExpired:
                    print(f"    TIMEOUT on {tag}", flush=True)
            if not ok:
                print(f"[FAIL] {ds} taegan seed{seed}", flush=True)


if __name__ == "__main__":
    main()
