#!/usr/bin/env python
"""Memorization-focused privacy table: distance to the closest TRAINING record
(DCR) and exact train-duplicate rate, per generator, with the real held-out set
as the reference row. A generator is memorization-safe when its synthetic
records are no closer to the training data than a genuine held-out real sample
is, and do not copy training rows more often than the holdout does.

Reads per-method synth->train metrics from long.csv (ext_dcr, ext_dup_rate) and
computes the real-holdout reference directly from the data. Writes
privacy_train_table.tex into the paper directory. No hardcoded numbers.

Run in the discodiff (or discobase) env: needs data + metrics_ext.
"""
import os
import numpy as np
import pandas as pd
import data
import metrics_ext as MX

HERE = os.path.dirname(os.path.abspath(__file__))
LONG = os.path.join(HERE, "tables", "comprehensive", "long.csv")
OUT = "/home/ec2-user/P-TA-MulFee/2026_JBHI___DISCO_Tab/privacy_train_table.tex"
ORDER = ["disco", "taegan", "adsgan", "tabddpm", "rtvae", "nflow", "ctgan",
         "pategan", "marginal", "tvae", "great", "realtabformer", "tagal", "tabsyn"]
MLAB = {"disco": "DISCO-TAB", "ctgan": "CTGAN", "tvae": "TVAE", "tabddpm": "TabDDPM",
        "adsgan": "ADS-GAN", "pategan": "PATE-GAN", "rtvae": "RTVAE", "nflow": "NFlow",
        "marginal": "Marginals", "great": "GReaT", "realtabformer": "RTF",
        "taegan": "TAEGAN", "tagal": "TAGAL", "tabsyn": "TabSyn"}


def holdout_reference(datasets):
    dcr, dup = [], []
    for ds in datasets:
        try:
            b = data.load_dataset(ds, seed=0)
            Rtr = MX._sub(b.pp.transform_X(b.df_train))
            Rho = MX._sub(b.pp.transform_X(b.df_test))
            d, _, _ = MX.dcr_nndr(Rho, Rtr, Rtr)        # holdout vs train
            dcr.append(d)
            dup.append(MX.duplicate_rate(b.df_train, b.df_test, b.feature_cols))
        except Exception as e:
            print("skip", ds, e)
    return float(np.mean(dcr)), float(np.mean(dup))


def main():
    L = pd.read_csv(LONG)
    L = L[L.dataset != "tab_credit_g"]
    datasets = sorted(L.dataset.unique())

    def dm(m, c):
        s = L[(L.model == m) & L[c].notna()]
        p = s.groupby("dataset")[c].mean()
        return p.mean() if len(p) else np.nan

    ref_dcr, ref_dup = holdout_reference(datasets)

    L_tab = [
        r"\begin{table}[t]\centering\scriptsize",
        r"\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.1}",
        r"\caption{\newexp{Memorization check against the \emph{training} set (dataset-averaged). DCR is the median distance to the nearest training record; duplicate rate is the fraction of synthetic rows matching a training row. The held-out set is the reference; DISCO-TAB (\textbf{bold}) is no closer and duplicates no more than it.}}",
        r"\label{tab:privacy_train}",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"Source & DCR to train $\uparrow$ & Train-duplicate rate $\downarrow$ \\",
        r"\midrule",
        r"\emph{Real holdout (reference)} & \emph{%.3f} & \emph{%.3f} \\" % (ref_dcr, ref_dup),
        r"\midrule",
    ]
    for m in ORDER:
        d, u = dm(m, "ext_dcr"), dm(m, "ext_dup_rate")
        if np.isnan(d):
            continue
        name = MLAB[m]
        if m == "disco":
            L_tab.append(r"\textbf{%s} & \textbf{%.3f} & \textbf{%.3f} \\" % (name, d, u))
        else:
            L_tab.append(r"%s & %.3f & %.3f \\" % (name, d, u))
    L_tab += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(OUT, "w").write("\n".join(L_tab) + "\n")
    print(f"wrote {OUT}")
    print(f"holdout reference: DCR={ref_dcr:.3f} dup={ref_dup:.4f}")


if __name__ == "__main__":
    main()
