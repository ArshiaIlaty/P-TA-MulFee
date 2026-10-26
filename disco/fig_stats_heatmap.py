#!/usr/bin/env python
"""Pairwise 'tournament' heatmap of downstream utility across all generators.

Cell (row i, col j) = median over the datasets both cover of the per-dataset
margin (F1_i - F1_j). Diverging palette centered at 0: blue = row beats column,
orange = row loses (Wong-safe poles, neutral-white midpoint). A dot marks pairs
that are significant under a two-sided Wilcoxon signed-rank test with
Holm-Bonferroni correction across all unique pairs. Methods are ordered by mean
rank (best first); DISCO-TAB is highlighted on the axes.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import wilcoxon, rankdata

LONG = "tables/comprehensive/long.csv"
OUT = "tables/comprehensive/figures/fig_stats_heatmap.pdf"
PAPER = "/home/ec2-user/P-TA-MulFee/2026_JBHI___DISCO_Tab/fig_stats_heatmap.pdf"
MLAB = {"disco": "DISCO-TAB", "taegan": "TAEGAN", "adsgan": "ADS-GAN",
        "tabddpm": "TabDDPM", "rtvae": "RTVAE", "nflow": "NFlow", "ctgan": "CTGAN",
        "pategan": "PATE-GAN", "marginal": "Marginals", "tvae": "TVAE",
        "great": "GReaT", "realtabformer": "RTF", "tagal": "TAGAL", "tabsyn": "TabSyn"}
CMAP = LinearSegmentedColormap.from_list("ou_bl", ["#D55E00", "#f7f7f7", "#0072B2"])


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx]); adj[idx] = min(run, 1.0)
    return adj


def main():
    L = pd.read_csv(LONG)
    L = L[(L.dataset != "tab_credit_g") & L.tstr_f1_macro.notna()]
    piv = L.groupby(["dataset", "model"]).tstr_f1_macro.mean().unstack("model")
    methods = [m for m in MLAB if m in piv.columns]

    # order by mean rank over the common (complete-block) datasets
    blk = piv[methods].dropna(axis=0, how="any")
    ranks = np.vstack([rankdata(-row) for row in blk.values])
    order = np.argsort(ranks.mean(axis=0))
    methods = [methods[i] for i in order]
    k = len(methods)

    M = np.full((k, k), np.nan)
    pairs, pidx = [], []
    for a in range(k):
        for b in range(k):
            if a == b:
                continue
            d = pd.concat([piv[methods[a]], piv[methods[b]]], axis=1).dropna()
            if len(d) < 6:
                continue
            M[a, b] = (d.iloc[:, 0] - d.iloc[:, 1]).median()
            if a < b:
                try:
                    pv = wilcoxon(d.iloc[:, 0], d.iloc[:, 1]).pvalue
                except ValueError:
                    pv = 1.0
                pairs.append((a, b)); pidx.append(pv)
    padj = holm(pidx)
    sig = {pr: pa for pr, pa in zip(pairs, padj)}

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    vmax = np.nanmax(np.abs(M))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(M, cmap=CMAP, norm=norm)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    labs = [MLAB[m] for m in methods]
    ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labs, fontsize=8)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        if t.get_text() == "DISCO-TAB":
            t.set_fontweight("bold"); t.set_color("#0072B2")
    for a in range(k):
        for b in range(k):
            if a == b:
                ax.add_patch(plt.Rectangle((b - .5, a - .5), 1, 1, color="#dddddd")); continue
            if np.isnan(M[a, b]):
                continue
            pr = (a, b) if a < b else (b, a)
            s = sig.get(pr, 1.0) < 0.05
            val = f"{M[a, b]:+.2f}"
            ax.text(b, a, (val + "•") if s else val, ha="center", va="center",
                    fontsize=6.0, color="#111" if abs(M[a, b]) < 0.6 * vmax else "white")
    ax.set_title("Pairwise downstream-utility margin (row $-$ column), median per-dataset $\\Delta F_1$\n"
                 "blue: row wins · orange: row loses · • Holm-corrected $p<0.05$",
                 fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"median $\Delta F_1$ (row $-$ column)", fontsize=8)
    ax.set_xticks(np.arange(-.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-.5, k, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="both", length=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight"); fig.savefig(OUT[:-4] + ".png", dpi=150, bbox_inches="tight")
    if os.path.isdir(os.path.dirname(PAPER)):
        fig.savefig(PAPER, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT, f"({k} methods)")


if __name__ == "__main__":
    main()
