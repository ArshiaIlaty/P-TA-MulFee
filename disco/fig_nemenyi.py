#!/usr/bin/env python
"""Nemenyi critical-difference diagrams, one per metric.

Complete-block design: for each metric we include every generator scored on the
metric and restrict to the datasets that *all* shown methods cover (the
Friedman/Nemenyi procedure requires complete blocks). Partial-coverage LLM/
latent baselines therefore shrink the shared-dataset intersection but are NOT
dropped from the comparison -- all methods appear. Reads
tables/comprehensive/long.csv and writes one PDF per metric.

Metric directions:
  high     higher is better (utility, correlation, PRDC, SDMetrics quality)
  low      lower is better  (divergences, distances, duplicate/association err)
  ideal0.5 closeness to 0.5 is better (C2ST, membership-inference AUROC)

Non-monotonic proximity metrics (raw DCR / NNDR / NNAA, where neither higher nor
lower is strictly "better" -- near-1 ratio is the goal) are intentionally NOT
ranked here; they are analysed in the privacy figures/tables instead.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, rankdata

LONG = "tables/comprehensive/long.csv"
PAPER = "/home/ec2-user/P-TA-MulFee/2026_JBHI___DISCO_Tab"
OUTDIR = "tables/comprehensive/nemenyi"       # internal: all per-metric CDs
DROP = {"tab_credit_g"}

DISPLAY = {
    "disco": "DISCO-TAB", "taegan": "TAEGAN", "adsgan": "ADS-GAN",
    "tabddpm": "TabDDPM", "rtvae": "RTVAE", "nflow": "NFlow",
    "ctgan": "CTGAN", "pategan": "PATE-GAN", "marginal": "Marginals",
    "tvae": "TVAE", "great": "GReaT", "realtabformer": "RTF",
    "tagal": "TAGAL", "tabsyn": "TabSyn",
}
# Studentized-range q_alpha / sqrt(2) for Nemenyi at alpha=0.05.
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
       8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
       14: 3.354, 15: 3.391}

# (column, label, direction, filename stem)
METRICS = [
    ("tstr_f1_macro",  "Downstream utility (TSTR macro-F1)",      "high", "utility_f1macro"),
    ("tstr_f1_weighted", "Downstream utility (TSTR weighted-F1)", "high", "utility_f1weighted"),
    ("tstr_accuracy",  "Downstream accuracy (TSTR)",              "high", "utility_acc"),
    ("fid_JS",         "Jensen-Shannon divergence",               "low",  "fid_js"),
    ("fid_KS",         "Kolmogorov-Smirnov statistic",            "low",  "fid_ks"),
    ("fid_Hellinger",  "Hellinger distance",                      "low",  "fid_hellinger"),
    ("fid_KL",         "KL divergence",                           "low",  "fid_kl"),
    ("fid_Pearson",    "Correlation preservation",                "high", "fid_pearson"),
    ("ext_wasserstein", "Wasserstein distance",                   "low",  "fid_wasserstein"),
    ("ext_mmd",        "Maximum mean discrepancy (MMD)",          "low",  "fid_mmd"),
    ("ext_assoc_err",  "Association error",                       "low",  "fid_assocerr"),
    ("ext_sdm_shapes", "SDMetrics column shapes",                 "high", "sdm_shapes"),
    ("ext_sdm_pairs",  "SDMetrics pair trends",                   "high", "sdm_pairs"),
    ("ext_sdm_quality", "SDMetrics overall quality",              "high", "sdm_quality"),
    ("ext_c2st",       "Detectability (C2ST AUROC)",              "ideal0.5", "detect_c2st"),
    ("ext_precision",  "PRDC precision",                          "high", "prdc_precision"),
    ("ext_recall",     "PRDC recall",                             "high", "prdc_recall"),
    ("ext_density",    "PRDC density",                            "high", "prdc_density"),
    ("ext_coverage",   "PRDC coverage",                           "high", "prdc_coverage"),
    ("ext_mia_auc",    "Membership-inference AUROC",              "ideal0.5", "priv_mia"),
    ("ext_dup_rate",   "Train-duplicate rate",                    "low",  "priv_duprate"),
]


def goodness(vals, direction):
    """Transform metric values so that a HIGHER transformed value = better."""
    if direction == "high":
        return vals
    if direction == "low":
        return -vals
    if direction == "ideal0.5":
        return -np.abs(vals - 0.5)
    raise ValueError(direction)


def cd_diagram(piv, title, out):
    methods = list(piv.columns)
    N, k = piv.shape
    if N < 3 or k < 3:
        print(f"skip {out}: only N={N} datasets, k={k} methods (too few)")
        return
    avg_rank = piv.values.mean(axis=0)   # piv already holds per-dataset ranks
    CD = Q05.get(k, 3.4) * np.sqrt(k * (k + 1) / (6.0 * N))

    order = np.argsort(avg_rank)
    names = [DISPLAY.get(methods[i], methods[i]) for i in order]
    rr = avg_rank[order]
    sr = rr

    cliques = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and sr[j + 1] - sr[i] <= CD:
            j += 1
        if j > i:
            cliques.append((i, j))
        i += 1
    maximal = [c for c in cliques
               if not any(c != d and d[0] <= c[0] and c[1] <= d[1] for d in cliques)]

    lo, hi = 1, k
    clq_gap = 0.030
    band = len(maximal) * clq_gap
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")
    axis_y = 0.78
    ax.plot([lo, hi], [axis_y, axis_y], "k-", lw=1.2)
    for x in range(lo, hi + 1):
        ax.plot([x, x], [axis_y, axis_y + 0.025], "k-", lw=1.2)
        ax.text(x, axis_y + 0.05, str(x), ha="center", va="bottom", fontsize=9)
    ax.text((lo + hi) / 2, axis_y + 0.165,
            f"{title}\nmean rank over {N} common datasets, {k} methods (lower is better)",
            ha="center", fontsize=9)
    ax.plot([lo, lo + CD], [axis_y + 0.255, axis_y + 0.255], "k-", lw=2.2)
    ax.plot([lo, lo], [axis_y + 0.235, axis_y + 0.275], "k-", lw=1.3)
    ax.plot([lo + CD, lo + CD], [axis_y + 0.235, axis_y + 0.275], "k-", lw=1.3)
    ax.text(lo + CD / 2, axis_y + 0.285, f"CD = {CD:.2f}", ha="center",
            va="bottom", fontsize=9)
    for ci, (a, b) in enumerate(maximal):
        yb = axis_y - 0.02 - ci * clq_gap
        ax.plot([sr[a] - 0.05, sr[b] + 0.05], [yb, yb], "k-", lw=3.2,
                solid_capstyle="round")
    row_top = axis_y - 0.04 - band
    half = (k + 1) // 2
    for idx, (nm, r) in enumerate(zip(names, rr)):
        left = idx < half
        row_y = row_top - (idx if left else (k - 1 - idx)) * 0.062
        xtext = lo - 0.45 if left else hi + 0.45
        ax.plot([r, r], [axis_y, row_y], "k-", lw=0.8)
        ax.plot([r, xtext], [row_y, row_y], "k-", lw=0.8)
        ax.text(xtext + (-0.05 if left else 0.05), row_y,
                f"{nm} ({r:.2f})", ha="right" if left else "left",
                va="center", fontsize=9,
                fontweight="bold" if nm == "DISCO-TAB" else "normal")
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    L = pd.read_csv(LONG)
    L = L[~L.dataset.isin(DROP)]
    summary = []
    for col, label, direction, stem in METRICS:
        if col not in L.columns:
            continue
        sub = L[L[col].notna()]
        piv = sub.groupby(["dataset", "model"])[col].mean().unstack("model")
        methods = [m for m in DISPLAY if m in piv.columns]
        piv = piv[methods].dropna(axis=0, how="any")   # complete blocks (all methods)
        N, k = piv.shape
        if N < 3 or k < 3:
            print(f"skip {col}: N={N} k={k}")
            continue
        g = goodness(piv.values, direction)
        ranks = np.vstack([rankdata(-row, method="average") for row in g])
        rank_piv = pd.DataFrame(ranks, columns=methods)
        try:
            stat, p = friedmanchisquare(*[g[:, i] for i in range(k)])
        except Exception:
            stat, p = float("nan"), float("nan")
        out = os.path.join(OUTDIR, f"fig_nemenyi_{stem}.pdf")
        cd_diagram(rank_piv, label, out)
        summary.append((col, N, k, stat, p, out))
        # the headline utility CD also lives in the paper dir (back-compat name)
        if col == "tstr_f1_macro":
            cd_diagram(rank_piv, label, os.path.join(PAPER, "fig_nemenyi_cd.pdf"))

    print(f"\nwrote {len(summary)} CD diagrams to {OUTDIR}/")
    print(f"{'metric':18s}{'N':>4}{'k':>4}{'chi2':>9}{'p':>11}")
    for col, N, k, stat, p, out in summary:
        print(f"{col:18s}{N:>4}{k:>4}{stat:>9.2f}{p:>11.2e}")


if __name__ == "__main__":
    main()
