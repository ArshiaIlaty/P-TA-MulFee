#!/usr/bin/env python
"""Hyperparameter / serialization sensitivity sweep (DISCO-TAB, Task 2).

Nine configurations (IFRS weight alpha, sampling top-k, constraint-discovery
threshold delta, and random vs. fixed column serialization) are each run on three
representative datasets (Heart Failure, Obesity, Diabetes), seed 0. Macro-F1,
fidelity (JS / correlation) and the DCR ratio come from results/<ds>__sw_*__seed0.json;
minority-class recall (averaged over RF/LogReg/XGBoost) comes from
downstream_ext.csv. Values are means over the three datasets.

The default row (alpha=0.2, top_k=8, delta=0.3, fixed order) is the production
config and anchors every block.
"""
import os, json, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, "tables", "comprehensive", "downstream_ext.csv")
OUT = "/home/ec2-user/P-TA-MulFee/2026_JBHI___DISCO_Tab/tab_sweep.tex"
DS = ["heart_failure", "obesity", "diabetes"]

# (tag, label) grouped by the knob being varied; None entries are block headers.
BLOCKS = [
    ("sw_default", r"Default ($\alpha{=}0.2$, $k{=}8$, $\delta{=}0.3$)"),
    (None, r"\emph{IFRS weight $\alpha$}"),
    ("sw_alpha0",  r"\quad $\alpha=0$ (IFRS off)"),
    ("sw_alpha01", r"\quad $\alpha=0.1$"),
    ("sw_alpha04", r"\quad $\alpha=0.4$"),
    (None, r"\emph{Sampling top-$k$}"),
    ("sw_k4",  r"\quad $k=4$"),
    ("sw_k16", r"\quad $k=16$"),
    (None, r"\emph{Constraint threshold $\delta$}"),
    ("sw_d02", r"\quad $\delta=0.2$"),
    ("sw_d04", r"\quad $\delta=0.4$"),
    (None, r"\emph{Serialization}"),
    ("sw_randord", r"\quad Random column order"),
]


def jmean(tag, key, ratio=False):
    vals = []
    for ds in DS:
        p = os.path.join(HERE, "results", f"{ds}__{tag}__seed0.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        if ratio:
            rr = d.get("dcr_realreal_median") or np.nan
            vals.append(d.get("dcr_median", np.nan) / rr if rr else np.nan)
        else:
            vals.append(d.get(key, np.nan))
    return np.nanmean(vals) if vals else np.nan


def main():
    ext = pd.read_csv(EXT)
    ext = ext[ext.model.astype(str).str.startswith("sw_")]
    rec = ext.groupby("model")["minority_recall"].mean()

    def row(tag):
        return (jmean(tag, "tstr_f1_macro"), rec.get(tag, np.nan),
                jmean(tag, "fid_JS"), jmean(tag, "fid_Pearson"),
                jmean(tag, None, ratio=True))

    L = [
        r"\begin{table}[t]\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.1}",
        r"\caption{\newexp{Sensitivity sweep (three representative datasets, seed 0; "
        r"means). Raising the IFRS weight $\alpha$ trades macro-$F_1$ for minority "
        r"recall; results are robust to top-$k$ and $\delta$; a random column order "
        r"degrades every metric. Min.\ recall is averaged over three classifiers.}}",
        r"\label{tab:sweep}",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Configuration & F1$_{\text{mac}}\!\uparrow$ & Min.rec$\uparrow$ & "
        r"JS$\downarrow$ & Corr.$\uparrow$ & DCR$\approx\!1$ \\",
        r"\midrule",
    ]
    for tag, lab in BLOCKS:
        if tag is None:
            L.append(r"\multicolumn{6}{@{}l}{" + lab + r"} \\")
            continue
        f1, mr, js, pe, dcr = row(tag)
        cells = [f"{f1:.3f}", f"{mr:.3f}", f"{js:.3f}", f"{pe:.3f}", f"{dcr:.2f}"]
        cells = ["--" if "nan" in c else c for c in cells]
        line = f"{lab} & " + " & ".join(cells) + r" \\"
        if tag == "sw_default":
            line += "\n\\midrule"
        L.append(line)
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(OUT, "w").write("\n".join(L) + "\n")
    print("wrote", OUT)
    for tag, lab in BLOCKS:
        if tag:
            print(f"  {tag:12s} F1={row(tag)[0]:.3f} minrec={row(tag)[1]:.3f}")


if __name__ == "__main__":
    main()
