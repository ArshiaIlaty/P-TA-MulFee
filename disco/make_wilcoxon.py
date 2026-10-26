#!/usr/bin/env python
"""Holm-corrected pairwise Wilcoxon signed-rank tests, DISCO-TAB vs. every other
generator on downstream utility, written to the paper as tab_wilcoxon.tex.

Pairing unit: one observation per dataset = the per-dataset mean TSTR macro-F1
(averaged over the three seeds). Each test pairs DISCO-TAB against one baseline
over the datasets both cover. Raw two-sided Wilcoxon p-values are corrected for
the family of comparisons with the Holm-Bonferroni step-down procedure.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

LONG = "tables/comprehensive/long.csv"
OUT = "/home/ec2-user/P-TA-MulFee/2026_JBHI___DISCO_Tab/tab_wilcoxon.tex"
ORDER = ["taegan", "realtabformer", "adsgan", "tabddpm", "tagal", "tvae", "rtvae",
         "nflow", "tabsyn", "great", "ctgan", "pategan", "marginal"]
MLAB = {"ctgan": "CTGAN", "tvae": "TVAE", "tabddpm": "TabDDPM", "adsgan": "ADS-GAN",
        "pategan": "PATE-GAN", "rtvae": "RTVAE", "nflow": "NFlow", "marginal": "Marginals",
        "great": "GReaT", "realtabformer": "RTF", "taegan": "TAEGAN", "tagal": "TAGAL",
        "tabsyn": "TabSyn"}


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def main():
    L = pd.read_csv(LONG)
    L = L[(L.dataset != "tab_credit_g") & L.tstr_f1_macro.notna()]
    piv = L.groupby(["dataset", "model"]).tstr_f1_macro.mean().unstack("model")
    disco = piv["disco"]

    recs = []
    for m in ORDER:
        if m not in piv:
            continue
        d = pd.concat([disco, piv[m]], axis=1).dropna()
        d.columns = ["disco", "other"]
        if len(d) < 6:
            continue
        delta = d["disco"] - d["other"]
        try:
            p = wilcoxon(d["disco"], d["other"]).pvalue
        except ValueError:          # all-zero differences
            p = 1.0
        recs.append([m, len(d), float(delta.median()), float(p)])

    praw = [r[3] for r in recs]
    padj = holm(praw)

    # Friedman over the full-coverage generators (complete 23-dataset blocks)
    full = [m for m in ["disco", "taegan", "adsgan", "tabddpm", "rtvae", "nflow",
                        "ctgan", "pategan", "marginal", "tvae"] if m in piv]
    fb = piv[full].dropna(axis=0, how="any")
    chi2, pf = friedmanchisquare(*[fb[m].values for m in full])

    L2 = [
        r"\begin{table}[t]\centering\footnotesize",
        r"\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.1}",
        r"\caption{\newexp{Pairwise Wilcoxon signed-rank tests of DISCO-TAB against "
        r"each generator on downstream utility. The pairing unit is one observation "
        r"per dataset (the per-dataset mean TSTR macro-$F_1$ over three seeds); each "
        r"test runs over the $n$ datasets both methods cover. $\Delta$ is the median "
        r"per-dataset margin (DISCO-TAB $-$ baseline); $p$ is the two-sided Wilcoxon "
        r"$p$-value and $p_{\text{Holm}}$ its Holm-Bonferroni correction across the "
        r"family. A Friedman test over the ten full-coverage generators "
        rf"($\chi^2={chi2:.1f}$, $p={pf:.1e}$) rejects equal mean ranks; the "
        r"critical-difference diagram (released with the code) shows the same "
        r"groupings. $\dagger$: DISCO-TAB significantly higher ($p_{\text{Holm}}<0.05$).}}",
        r"\label{tab:wilcoxon}",
        r"\begin{tabular}{@{}l r r r r@{}}",
        r"\toprule",
        r"Baseline & $n$ & median $\Delta F_1$ & $p$ & $p_{\text{Holm}}$ \\",
        r"\midrule",
    ]
    for (m, n, dlt, p), pa in zip(recs, padj):
        sig = r"$^{\dagger}$" if (pa < 0.05 and dlt > 0) else ""
        pr = "<0.001" if p < 1e-3 else f"{p:.3f}"
        pah = "<0.001" if pa < 1e-3 else f"{pa:.3f}"
        L2.append(f"{MLAB[m]}{sig} & {n} & ${dlt:+.3f}$ & {pr} & {pah} \\\\")
    L2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(OUT, "w").write("\n".join(L2) + "\n")
    print("wrote", OUT)
    print(f"Friedman (full-coverage): chi2={chi2:.2f} p={pf:.2e}")
    for (m, n, dlt, p), pa in zip(recs, padj):
        print(f"  {MLAB[m]:10s} n={n:2d} d={dlt:+.3f} p={p:.4f} holm={pa:.4f}")


if __name__ == "__main__":
    main()
