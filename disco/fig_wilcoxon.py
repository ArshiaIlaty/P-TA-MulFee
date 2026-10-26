#!/usr/bin/env python
"""Wilcoxon signed-rank visualization: DISCO-TAB vs every other generator on
downstream utility (TSTR macro-F1).

Pairing unit = one observation per dataset (per-dataset mean over seeds), paired
over each pair's common datasets. Bars show the median per-dataset margin
(DISCO - other) with 25-75 percentile whiskers; a two-sided Wilcoxon p-value is
Holm-Bonferroni corrected across the family. Significance is encoded redundantly
(not color alone): a filled dagger marker + bold label where DISCO is
significantly higher at Holm p<0.05. Diverging palette: blue = DISCO higher,
orange = DISCO lower (Wong colorblind-safe poles).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

LONG = "tables/comprehensive/long.csv"
OUT = "tables/comprehensive/figures/fig_wilcoxon_holm.pdf"
PAPER = "/home/ec2-user/P-TA-MulFee/2026_JBHI___DISCO_Tab/fig_wilcoxon_holm.pdf"
POS, NEG, INK, MUT = "#0072B2", "#D55E00", "#222222", "#888888"
MLAB = {"ctgan": "CTGAN", "tvae": "TVAE", "tabddpm": "TabDDPM", "adsgan": "ADS-GAN",
        "pategan": "PATE-GAN", "rtvae": "RTVAE", "nflow": "NFlow", "marginal": "Marginals",
        "great": "GReaT", "realtabformer": "RTF", "taegan": "TAEGAN", "tagal": "TAGAL",
        "tabsyn": "TabSyn"}


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
    disco = piv["disco"]
    rows = []
    for m in MLAB:
        if m not in piv:
            continue
        d = pd.concat([disco, piv[m]], axis=1).dropna()
        d.columns = ["disco", "other"]
        if len(d) < 6:
            continue
        diff = d["disco"] - d["other"]
        try:
            p = wilcoxon(d["disco"], d["other"]).pvalue
        except ValueError:
            p = 1.0
        rows.append([MLAB[m], float(diff.median()), float(np.percentile(diff, 25)),
                     float(np.percentile(diff, 75)), p, len(d)])
    padj = holm([r[4] for r in rows])
    for r, pa in zip(rows, padj):
        r.append(pa)
    rows.sort(key=lambda r: r[1])                         # by median margin

    names = [r[0] for r in rows]
    med = np.array([r[1] for r in rows])
    q1 = np.array([r[2] for r in rows]); q3 = np.array([r[3] for r in rows])
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    colors = [POS if v >= 0 else NEG for v in med]
    ax.barh(y, med, color=colors, edgecolor="white", linewidth=0.6, height=0.66, zorder=3)
    # 25-75 percentile whiskers (per-dataset spread)
    ax.hlines(y, q1, q3, color=INK, lw=1.0, zorder=4)
    ax.vlines(np.concatenate([q1, q3]), np.concatenate([y, y]) - 0.12,
              np.concatenate([y, y]) + 0.12, color=INK, lw=1.0, zorder=4)
    ax.axvline(0, color="#333", lw=1.0, zorder=2)

    for i, r in enumerate(rows):
        _, m_, _, _, p, n, pa = r
        sig = pa < 0.05 and m_ > 0
        tag = ("<0.001" if pa < 1e-3 else f"{pa:.3f}")
        lbl = (f"† " if sig else "") + f"{m_:+.3f} (p$_{{H}}$={tag})"
        xoff = 0.006 if m_ >= 0 else -0.006
        ax.text(max(q3[i], m_) + 0.006 if m_ >= 0 else min(q1[i], m_) - 0.006, i, lbl,
                va="center", ha="left" if m_ >= 0 else "right",
                fontsize=7, fontweight="bold" if sig else "normal", color=INK, zorder=5)

    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9, color=INK)
    ax.set_xlabel(r"median per-dataset $\Delta F_1$  (DISCO-TAB $-$ baseline)", fontsize=9, color=INK)
    ax.set_title("DISCO-TAB vs. each generator — Wilcoxon signed-rank\n"
                 r"whiskers = 25–75th pct; $\dagger$ = Holm-corrected $p<0.05$ (DISCO higher)",
                 fontsize=9.5, color=INK)
    span = max(q3.max(), med.max()) - min(q1.min(), med.min())
    ax.set_xlim(min(q1.min(), med.min()) - 0.10 * span - 0.02,
                max(q3.max(), med.max()) + 0.22 * span + 0.02)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_axisbelow(True); ax.xaxis.grid(True, color="#eee", lw=0.8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight"); fig.savefig(OUT[:-4] + ".png", dpi=150, bbox_inches="tight")
    if os.path.isdir(os.path.dirname(PAPER)):
        fig.savefig(PAPER, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
