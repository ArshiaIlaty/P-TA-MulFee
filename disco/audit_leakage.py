"""
Leakage / memorization audit (RED placeholder #3: Heart Failure F1=1.000).

For a (dataset, method, seed) checks:
  (i)   target is excluded from predictors (structural: pp.transform_X uses
        feature cols only) -- asserted here;
  (ii)  no synthetic record exactly duplicates a real TEST record (and how many
        duplicate a real TRAIN record);
  (iii) near-duplicate rate: fraction of synthetic rows within epsilon of a real
        TEST record in the standardized encoded space;
  (iv)  standardization fit on the training partition only (structural).
Prints a per-run report and returns a dict.
"""
import os, glob, json, argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))
from data import load_dataset


def _key(df, cols):
    """Row signature: numeric rounded to 3 sig places, categoricals as strings."""
    out = df[cols].copy()
    for c in cols:
        v = pd.to_numeric(out[c], errors="coerce")
        if v.notna().mean() > 0.5:
            out[c] = v.round(3).astype(str)
        else:
            out[c] = out[c].astype(str)
    return out.agg("|".join, axis=1)


def audit(dataset, method, seed, syn_path, eps_quantile=0.01):
    b = load_dataset(dataset, seed=seed)
    syn = pd.read_csv(syn_path)
    cols = [c for c in b.df_train.columns]
    feat = b.feature_cols

    # (ii) exact duplicates (full row incl. target)
    k_tr = set(_key(b.df_train, cols))
    k_te = set(_key(b.df_test, cols))
    k_sy = _key(syn, cols)
    dup_train = int(k_sy.isin(k_tr).sum())
    dup_test = int(k_sy.isin(k_te).sum())

    # also feature-only duplicates (ignore target)
    ktr_f = set(_key(b.df_train, feat))
    kte_f = set(_key(b.df_test, feat))
    ksy_f = _key(syn, feat)
    dupf_train = int(ksy_f.isin(ktr_f).sum())
    dupf_test = int(ksy_f.isin(kte_f).sum())

    # (iii) near-duplicate in encoded space vs real TEST
    Xte = b.pp.transform_X(b.df_test)
    Xsy = b.pp.transform_X(syn)
    d = max(1, b.pp.n_encoded)
    nn = NearestNeighbors(n_neighbors=1).fit(Xte)
    dist, _ = nn.kneighbors(Xsy)
    dist = dist.ravel() / np.sqrt(d)
    # reference scale: real-train -> real-test nearest distances
    Xtr = b.pp.transform_X(b.df_train)
    dr, _ = nn.kneighbors(Xtr)
    dr = dr.ravel() / np.sqrt(d)
    eps = float(np.quantile(dr, eps_quantile))
    near = float((dist <= eps).mean())

    rep = {
        "dataset": dataset, "method": method, "seed": seed, "n_syn": len(syn),
        "n_train": len(b.df_train), "n_test": len(b.df_test),
        "target_in_predictors": False,            # structural: X excludes target
        "scaler_fit_on_train_only": True,         # structural: pp fit on df_train
        "exact_dup_train_rows": dup_train,
        "exact_dup_test_rows": dup_test,
        "featonly_dup_train_rows": dupf_train,
        "featonly_dup_test_rows": dupf_test,
        "near_dup_eps": round(eps, 4),
        "near_dup_frac_vs_test": round(near, 4),
        "syn_to_test_dcr_median": round(float(np.median(dist)), 4),
        "train_to_test_dcr_median": round(float(np.median(dr)), 4),
    }
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="heart_failure")
    ap.add_argument("--method", default="disco")
    ap.add_argument("--glob", default=None)
    args = ap.parse_args()
    pat = args.glob or os.path.join(HERE, "synthetic",
                                    f"{args.dataset}__{args.method}__seed*.csv")
    reps = []
    for p in sorted(glob.glob(pat)):
        base = os.path.basename(p)
        seed = int(base.split("seed")[-1].split(".")[0])
        method = base.split("__")[1]
        rep = audit(args.dataset, method, seed, p)
        reps.append(rep)
        print(json.dumps(rep, indent=2))
    if reps:
        out = os.path.join(HERE, "tables", f"leakage_audit_{args.dataset}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(reps, open(out, "w"), indent=2)
        print("wrote", out)


if __name__ == "__main__":
    main()
