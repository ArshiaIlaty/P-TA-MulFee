"""
DISCO-TAB: reproducible baseline synthetic-data generators.

Generators (GaussianCopula / CTGAN / TVAE from SDV) are trained ONLY on the
real train split (same stratified 80/20 split as data.py) and emit the same
number of rows as the train split, so downstream TSTR is directly comparable
to DISCO-TAB.

Two layers:
  * Base-env-importable helpers (`list_baseline_csvs`, `load_baseline`) that only
    read already-generated CSVs -- NO sdv import, safe to import from base.
  * A CLI generation path (`__main__`) that lazily imports sdv and must be run
    with the isolated `discobase` conda env:
        conda run -n discobase python baselines.py --datasets heart_failure \
            --methods gaussian_copula ctgan tvae --seeds 0
"""
import os
import glob
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data_raw")
OUTDIR = os.path.join(HERE, "synthetic", "baselines")
os.makedirs(OUTDIR, exist_ok=True)

ALL_DATASETS = [
    "heart_failure", "breast_cancer", "liver_disorders", "parkinsons",
    "obesity", "diabetes", "german_credit", "bank_marketing", "census_income",
]
ALL_METHODS = ["gaussian_copula", "ctgan", "tvae"]
LARGE = {"diabetes", "bank_marketing", "census_income"}
SUBSAMPLE_CAP = 20000
TARGET = "target"


def _csv_path(dataset, method, seed):
    return os.path.join(OUTDIR, f"{dataset}__{method}__seed{seed}.csv")


# ---- base-env-importable helpers (no sdv) --------------------------------
def list_baseline_csvs(dataset, seed=0):
    """Return {method: path} for every baseline CSV already written."""
    out = {}
    for p in glob.glob(os.path.join(OUTDIR, f"{dataset}__*__seed{seed}.csv")):
        method = os.path.basename(p).split("__")[1]
        out[method] = p
    return out


def load_baseline(dataset, method, seed=0):
    import pandas as pd
    return pd.read_csv(_csv_path(dataset, method, seed))


# ---- generation path (requires sdv / discobase env) ----------------------
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


def _metadata(df):
    from sdv.metadata import SingleTableMetadata
    md = SingleTableMetadata()
    md.detect_from_dataframe(df)
    # Force the label to categorical so synthesizers model it as a class.
    try:
        md.update_column(column_name=TARGET, sdtype="categorical")
    except Exception:
        pass
    return md


def _make_synth(method, metadata, seed, epochs):
    from sdv.single_table import (GaussianCopulaSynthesizer,
                                  CTGANSynthesizer, TVAESynthesizer)
    if method == "gaussian_copula":
        return GaussianCopulaSynthesizer(metadata)
    if method == "ctgan":
        return CTGANSynthesizer(metadata, epochs=epochs, verbose=False, cuda=True)
    if method == "tvae":
        return TVAESynthesizer(metadata, epochs=epochs, cuda=True)
    raise ValueError(method)


def generate(dataset, method, seed=0, epochs=100):
    import numpy as np
    np.random.seed(seed)
    tr = _train_split(dataset, seed)
    n_rows = len(tr)
    fit_df = tr
    if dataset in LARGE and method in ("ctgan", "tvae") and len(tr) > SUBSAMPLE_CAP:
        fit_df = tr.sample(n=SUBSAMPLE_CAP, random_state=seed).reset_index(drop=True)
    md = _metadata(fit_df)
    synth = _make_synth(method, md, seed, epochs)
    synth.fit(fit_df)
    sample = synth.sample(num_rows=n_rows)
    sample = sample[tr.columns]  # identical column order incl. target
    out = _csv_path(dataset, method, seed)
    sample.to_csv(out, index=False)
    return out, sample.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    for ds in args.datasets:
        for seed in args.seeds:
            for m in args.methods:
                try:
                    path, shape = generate(ds, m, seed, args.epochs)
                    print(f"[OK] {ds} / {m} / seed{seed} -> {os.path.basename(path)} {shape}",
                          flush=True)
                except Exception as e:
                    print(f"[FAIL] {ds} / {m} / seed{seed}: {type(e).__name__}: {e}",
                          flush=True)


if __name__ == "__main__":
    main()
