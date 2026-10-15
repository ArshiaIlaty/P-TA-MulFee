"""
DISCO-TAB: TabDDPM (diffusion) baseline via synthcity's "ddpm" plugin.

Trained ONLY on the real train split (same stratified 80/20 split as data.py /
baselines.py) and emits train-split row count, writing
synthetic/baselines/<dataset>__tabddpm__seed<seed>.csv so evaluate_baselines.py
auto-scores it under the same protocol as the other baselines.

Run with the isolated `discodiff` env:
    conda run -n discodiff python baselines_diffusion.py --datasets ... --seeds 0
"""
import os
import argparse
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data_raw")
OUTDIR = os.path.join(HERE, "synthetic", "baselines")
os.makedirs(OUTDIR, exist_ok=True)

ALL_DATASETS = [
    "heart_failure", "breast_cancer", "liver_disorders", "parkinsons",
    "obesity", "diabetes", "german_credit", "bank_marketing", "census_income",
]
LARGE = {"diabetes", "bank_marketing", "census_income"}
SUBSAMPLE_CAP = 20000
TARGET = "target"


def _train_split(dataset, seed):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(os.path.join(RAW, f"{dataset}.csv"))
    df[TARGET] = df[TARGET].astype(str)
    try:
        tr, _ = train_test_split(df, test_size=0.2, random_state=seed,
                                 stratify=df[TARGET])
    except ValueError:
        tr, _ = train_test_split(df, test_size=0.2, random_state=seed)
    return tr.reset_index(drop=True)


def _generate(dataset, seed, device):
    import pandas as pd
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader

    tr = _train_split(dataset, seed)
    n = len(tr)
    fit_df = tr
    n_iter = 2000
    if dataset in LARGE:
        if len(tr) > SUBSAMPLE_CAP:
            fit_df = tr.sample(SUBSAMPLE_CAP, random_state=seed).reset_index(drop=True)
        n_iter = 600
        # Diffusion sampling of tens of thousands of rows (x1000 denoising steps)
        # is impractically slow; cap to a scale comparable to DISCO-TAB's gen_cap.
        n = min(n, 5000)

    import torch
    dev = torch.device(device)
    loader = GenericDataLoader(fit_df, target_column=TARGET)
    plugin = Plugins().get(
        "ddpm", n_iter=n_iter, batch_size=1024, num_timesteps=1000,
        is_classification=True, device=dev,
    )
    plugin.fit(loader)
    syn = plugin.generate(count=n).dataframe()
    # Align columns to the real schema.
    syn = syn[[c for c in tr.columns if c in syn.columns]]
    for c in tr.columns:
        if c not in syn.columns:
            syn[c] = tr[c].iloc[0]
    return syn[list(tr.columns)]


def run(datasets, seeds):
    import pandas as pd
    for ds in datasets:
        for seed in seeds:
            out = os.path.join(OUTDIR, f"{ds}__tabddpm__seed{seed}.csv")
            if os.path.exists(out):
                print(f"[skip] {ds} seed{seed} exists", flush=True)
                continue
            for device in ("cuda", "cpu"):
                try:
                    syn = _generate(ds, seed, device)
                    syn.to_csv(out, index=False)
                    print(f"[ok] {ds}__tabddpm__seed{seed}: {syn.shape} ({device})", flush=True)
                    break
                except Exception as e:
                    msg = str(e)[:160]
                    print(f"[fail {device}] {ds} seed{seed}: {msg}", flush=True)
                    if device == "cpu":
                        print(f"[GIVEUP] {ds} seed{seed}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = ap.parse_args()
    run(args.datasets, args.seeds)
