#!/usr/bin/env python
"""True independent-marginals baseline.

Resamples every column independently from the real training split, so each
feature's marginal distribution is reproduced exactly while all inter-feature
dependencies are destroyed. This is the textbook lower-bound baseline and a
sanity anchor for the metric pipeline: it should be near-best on marginal
metrics (KS, JSD, SDMetrics-Shapes) and near-worst on dependency-sensitive
metrics (correlation preservation, C2ST, downstream utility).

Replaces the synthcity ``marginal_distributions`` plugin, which does not
reproduce skewed empirical marginals. Writes
synthetic/baselines/<dataset>__marginal__seed<seed>.csv so evaluate_baselines.py
scores it unchanged. Samples from the exact split the scorer uses
(data.load_dataset), so generation and scoring are consistent.
"""
import os
import argparse
import numpy as np
import pandas as pd
from data import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "synthetic", "baselines")
MAX_ROWS = 6000  # cap large datasets, matching the other baselines' generation


def gen(dataset, seed):
    b = load_dataset(dataset, seed=seed)
    train = b.df_train
    n = min(len(train), MAX_ROWS)
    rng = np.random.RandomState(seed)
    syn = pd.DataFrame({c: rng.choice(train[c].values, size=n, replace=True)
                        for c in train.columns})
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{dataset}__marginal__seed{seed}.csv")
    syn.to_csv(path, index=False)
    return path, syn.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    for ds in args.datasets:
        for s in args.seeds:
            try:
                path, shape = gen(ds, s)
                print(f"[OK] {ds:22s} marginal seed{s} shape={shape}", flush=True)
            except Exception as e:
                print(f"[FAIL] {ds} seed{s}: {e}", flush=True)


if __name__ == "__main__":
    main()
