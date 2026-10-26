#!/usr/bin/env python
"""Extended downstream-utility table (Table IV), dataset-averaged over the
23-dataset benchmark. Per-classifier minority-class recall (RF/LogReg/XGBoost)
plus minority precision and F1 (averaged over the three classifiers), AUROC,
AUPRC, and the synthetic/real minority-class ratio. Top row is the
train-on-real, test-on-real (TRTR) reference. Ranking is GLOBAL (across all
generators); bold+underline/bold/underline = best/second/third. The class-ratio
column is reported without a winner mark (ideal = 1: a ratio above 1 means the
generator over-represents the minority class, e.g. DISCO-TAB via inverse-
frequency reward shaping; below 1 means it is dropped).

Laid out as a two-column table* so the per-classifier recall columns fit beside
the added precision / F1 / class-ratio columns.
"""
import os
import numpy as np
import pandas as pd
from data import load_dataset
from eval_downstream import make_clfs, eval_one

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "tables", "comprehensive", "downstream_ext.csv")
CLASSDIST = os.path.join(HERE, "tables", "comprehensive", "synthetic_class_distribution.csv")
OUT = "/home/ec2-user/P-TA-MulFee/2026_JBHI___DISCO_Tab/downstream_ext_table.tex"

FULL = ["disco", "taegan", "adsgan", "tabddpm", "rtvae", "nflow", "ctgan",
        "pategan", "marginal", "tvae"]
LLM = ["great", "realtabformer", "tagal", "tabsyn"]
RANK = FULL + LLM
MLAB = {"disco": "DISCO-TAB", "ctgan": "CTGAN", "tvae": "TVAE", "tabddpm": "TabDDPM",
        "adsgan": "ADS-GAN", "pategan": "PATE-GAN", "rtvae": "RTVAE", "nflow": "NFlow",
        "marginal": "Marginals", "great": "GReaT", "realtabformer": "RTF",
        "taegan": "TAEGAN", "tagal": "TAGAL", "tabsyn": "TabSyn"}

# number of ranked columns (recall, precision, f1, auroc, auprc); the trailing
# class-ratio column is reported but not ranked.
NRANK = 5


def dsmean(d, model, clf, metric):
    s = d[(d.model == model) & (d.clf == clf)]
    pm = s.groupby("dataset")[metric].mean()
    return pm.mean() if len(pm) else np.nan


def clfmean(d, model, metric):
    vals = [dsmean(d, model, c, metric) for c in ("rf", "logreg", "xgb")]
    vals = [v for v in vals if not np.isnan(v)]
    return np.mean(vals) if vals else np.nan


def synratio(cd, model):
    s = cd[(cd.method == model) & (cd.real_minority_frac > 0)]
    if not len(s):
        return np.nan
    return (s.syn_minority_frac / s.real_minority_frac).mean()


def trtr_ref(datasets, seeds=(0, 1, 2)):
    rows = []
    for ds in datasets:
        for s in seeds:
            try:
                b = load_dataset(ds, seed=s)
                Xtr, ytr = b.pp.xy(b.df_train); keep = ytr >= 0; Xtr, ytr = Xtr[keep], ytr[keep]
                Xte, yte = b.pp.xy(b.df_test)
                minority = int(pd.Series(ytr).value_counts().idxmin())
            except Exception:
                continue
            for cname, clf in make_clfs(s).items():
                rows.append({"dataset": ds, "seed": s, "clf": cname,
                             **eval_one(Xtr, ytr, Xte, yte, clf, minority)})
    t = pd.DataFrame(rows)

    def dsm(clf, metric):
        v = t[t.clf == clf]
        return v.groupby("dataset")[metric].mean().mean() if len(v) else np.nan
    cm = lambda metric: np.nanmean([dsm(c, metric) for c in ("rf", "logreg", "xgb")])
    # real data preserves the real class balance by definition -> ratio 1.00
    return [cm("minority_recall"), cm("minority_precision"), cm("minority_f1"),
            cm("auroc"), cm("auprc"), 1.00]


def main():
    d = pd.read_csv(SRC).drop_duplicates(["dataset", "model", "seed", "clf"])
    d = d[d.dataset != "tab_credit_g"]
    cd = pd.read_csv(CLASSDIST)
    cd = cd[cd.dataset != "tab_credit_g"]

    def row_vals(m):
        return [clfmean(d, m, "minority_recall"),
                clfmean(d, m, "minority_precision"),
                clfmean(d, m, "minority_f1"),
                clfmean(d, m, "auroc"), clfmean(d, m, "auprc"),
                synratio(cd, m)]

    vals = {m: row_vals(m) for m in RANK}
    # Rank only methods with near-complete coverage, so a partially-scored method
    # (e.g. mid-recompute) can't be bolded on an inflated, incomplete mean.
    cov = {m: int(d[d.model == m]["dataset"].nunique()) for m in RANK}
    maxcov = max(cov.values()) if cov else 0
    eligible = {m for m in RANK if cov.get(m, 0) >= max(1, int(0.8 * maxcov))}
    orders = [sorted((m for m in RANK if m in eligible and not np.isnan(vals[m][j])),
                     key=lambda m: -vals[m][j]) for j in range(NRANK)]

    def fmt(m, vv):
        cells = []
        for j, v in enumerate(vv):
            s = "--" if np.isnan(v) else (f"{v:.2f}" if j == NRANK else f"{v:.3f}")
            if j < NRANK:                       # ranked columns get best/2nd/3rd marks
                o = orders[j]
                if len(o) > 0 and m == o[0]:
                    s = f"\\textbf{{\\underline{{{s}}}}}"
                elif len(o) > 1 and m == o[1]:
                    s = f"\\textbf{{{s}}}"
                elif len(o) > 2 and m == o[2]:
                    s = f"\\underline{{{s}}}"
            cells.append(s)
        return " & ".join(cells)

    L = [
        r"\begin{table}[t]\centering\footnotesize",
        r"\setlength{\tabcolsep}{4.5pt}\renewcommand{\arraystretch}{1.1}",
        r"\caption{\newexp{Extended downstream utility (dataset-averaged, three seeds); "
        r"minority recall, precision, $F_1$, AUROC and AUPRC are each averaged over three "
        r"classifiers (random forest, logistic regression, XGBoost). The last column is the "
        r"synthetic/real minority-class ratio (1 = faithful; $>$1 = minority over-represented, "
        r"as DISCO-TAB does via inverse-frequency reward shaping; $<$1 = dropped). Per column "
        r"across all generators, \textbf{\underline{bold+underline}}/\textbf{bold}/"
        r"\underline{underline} = best/second/third (class ratio not ranked); the top row is the "
        r"train-on-real (TRTR) reference; the lower block is partial-coverage.}}",
        r"\label{tab:downstream_ext}",
        r"\begin{tabular}{@{}l ccc cc c@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Minority class $\uparrow$} & "
        r"\multicolumn{2}{c}{Mean $\uparrow$} & Class \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
        r"Generator & Recall & Prec. & $F_1$ & AUROC & AUPRC & ratio \\",
        r"\midrule",
    ]
    datasets = sorted(d.dataset.unique())
    tref = trtr_ref(datasets)
    tcells = " & ".join("--" if np.isnan(v) else (f"{v:.2f}" if j == NRANK else f"{v:.3f}")
                        for j, v in enumerate(tref))
    L.append(r"\emph{Real data (TRTR)} & " + tcells + r" \\")
    L.append(r"\midrule")
    for m in FULL:
        L.append(f"{MLAB[m]} & {fmt(m, vals[m])} \\\\")
    L.append(r"\midrule")
    for m in LLM:
        L.append(f"{MLAB[m]} & {fmt(m, vals[m])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(OUT, "w").write("\n".join(L) + "\n")
    print("wrote", OUT)
    print("AUROC order (global):", orders[5][:3], "| AUPRC:", orders[6][:3],
          "| F1:", orders[4][:3], "| Prec:", orders[3][:3])


if __name__ == "__main__":
    main()
