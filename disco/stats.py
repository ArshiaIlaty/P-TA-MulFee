"""
DISCO-TAB: aggregate per-seed results and run the statistical tests the
reviewers asked for (RED placeholders #2, #6).

Reads every results/*.json (DISCO-TAB + baselines), aggregates mean +/- SD over
seeds per (dataset, method, metric), and runs:
  - Friedman test across methods over datasets (per-dataset mean F1 as blocks)
    + Nemenyi post-hoc;
  - Wilcoxon signed-rank, DISCO-TAB vs each baseline, paired across datasets,
    with Holm-Bonferroni correction and a rank-biserial effect size.
Writes tables/ CSV + LaTeX snippets and prints a summary.
"""
import os, re, json, glob, warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TAB = os.path.join(HERE, "tables")
os.makedirs(TAB, exist_ok=True)

METRICS = ["tstr_f1_macro", "tstr_f1_weighted", "fid_JS", "fid_KL",
           "fid_Hellinger", "fid_KS", "fid_Pearson", "fid_JS_num", "fid_JS_cat",
           "dcr_median", "dcr_realreal_median", "faith_FAITH", "faith_Factuality",
           "faith_Alignment", "faith_Integrity", "faith_Tracking", "valid_rate",
           "trtr_f1_macro"]


MAIN_METHODS = {"disco", "ctgan", "tvae", "gaussian_copula"}


def load_results(methods=None):
    rows = []
    for p in glob.glob(os.path.join(RES, "*.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if methods is not None and d.get("tag") not in methods:
            continue
        rec = {"dataset": d.get("dataset"), "method": d.get("tag"),
               "seed": d.get("seed")}
        for m in METRICS:
            rec[m] = d.get(m, np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def aggregate(df, metric="tstr_f1_macro"):
    g = df.groupby(["dataset", "method"])[metric]
    agg = g.agg(["mean", "std", "count"]).reset_index()
    return agg


def pivot_mean(df, metric="tstr_f1_macro"):
    return df.pivot_table(index="dataset", columns="method", values=metric,
                          aggfunc="mean")


def friedman_nemenyi(df, metric="tstr_f1_macro", disco="disco"):
    piv = pivot_mean(df, metric).dropna(axis=0, how="any")
    if piv.shape[0] < 3 or piv.shape[1] < 3:
        return {"note": f"insufficient data for Friedman ({piv.shape})",
                "pivot": piv}
    arrays = [piv[c].values for c in piv.columns]
    stat, p = stats.friedmanchisquare(*arrays)
    out = {"friedman_stat": float(stat), "friedman_p": float(p),
           "methods": list(piv.columns), "n_datasets": int(piv.shape[0])}
    try:
        import scikit_posthocs as sp
        nem = sp.posthoc_nemenyi_friedman(piv.values)
        nem.index = piv.columns
        nem.columns = piv.columns
        out["nemenyi"] = nem
        if disco in piv.columns:
            out["nemenyi_vs_disco"] = nem[disco].to_dict()
    except Exception as e:
        out["nemenyi_error"] = str(e)
    # average ranks (higher F1 = better rank 1)
    ranks = piv.rank(axis=1, ascending=False).mean(axis=0)
    out["avg_ranks"] = ranks.to_dict()
    return out


def wilcoxon_vs_disco(df, metric="tstr_f1_macro", disco="disco", higher_better=True):
    piv = pivot_mean(df, metric)
    if disco not in piv.columns:
        return pd.DataFrame()
    rows = []
    for m in piv.columns:
        if m == disco:
            continue
        pair = piv[[disco, m]].dropna()
        if len(pair) < 3:
            rows.append({"method": m, "n": len(pair), "p": np.nan,
                         "note": "n<3"})
            continue
        a, b = pair[disco].values, pair[m].values
        try:
            w, p = stats.wilcoxon(a, b)
        except ValueError:
            w, p = np.nan, 1.0
        diff = (a - b) if higher_better else (b - a)
        rows.append({"method": m, "n": len(pair),
                     "disco_mean": float(a.mean()), "base_mean": float(b.mean()),
                     "mean_diff": float(diff.mean()), "wins": int((diff > 0).sum()),
                     "W": float(w) if w == w else np.nan, "p": float(p)})
    out = pd.DataFrame(rows)
    if len(out):
        # Holm-Bonferroni
        order = out["p"].fillna(1.0).argsort().values
        mcnt = len(order)
        holm = np.empty(mcnt)
        prev = 0.0
        for rank, idx in enumerate(order):
            adj = min(1.0, (mcnt - rank) * out["p"].fillna(1.0).values[idx])
            prev = max(prev, adj)
            holm[idx] = prev
        out["p_holm"] = holm
        out["sig_holm_0.05"] = out["p_holm"] < 0.05
    return out


def main():
    df = load_results(methods=MAIN_METHODS)
    if df.empty:
        print("no results found in", RES)
        return
    df.to_csv(os.path.join(TAB, "all_results_long.csv"), index=False)
    print(f"loaded {len(df)} result rows; methods={sorted(df['method'].unique())}; "
          f"datasets={sorted(df['dataset'].unique())}\n")

    # main F1 table (mean)
    piv = pivot_mean(df, "tstr_f1_macro")
    piv.to_csv(os.path.join(TAB, "tstr_f1_macro_mean.csv"))
    print("=== TSTR F1-macro (mean over seeds) ===")
    print(piv.round(3).to_string(), "\n")

    # mean +/- std for the headline metric
    agg = aggregate(df, "tstr_f1_macro")
    agg.to_csv(os.path.join(TAB, "tstr_f1_macro_agg.csv"), index=False)

    fn = friedman_nemenyi(df, "tstr_f1_macro")
    print("=== Friedman (TSTR F1-macro across methods/datasets) ===")
    for k in ("friedman_stat", "friedman_p", "n_datasets", "avg_ranks"):
        if k in fn:
            print(f"  {k}: {fn[k]}")
    print()
    wil = wilcoxon_vs_disco(df, "tstr_f1_macro")
    if len(wil):
        wil.to_csv(os.path.join(TAB, "wilcoxon_vs_disco.csv"), index=False)
        print("=== Wilcoxon DISCO-TAB vs baselines (paired across datasets) ===")
        print(wil.round(4).to_string(index=False))
    return df, piv, fn, wil


if __name__ == "__main__":
    main()
