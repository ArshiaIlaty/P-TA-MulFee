#!/usr/bin/env python
"""Combined privacy figure (two panels, full width) replacing the separate
per-dataset DCR figure and the utility-privacy scatter.
(a) utility vs mean DCR ratio across generators; (b) per-dataset DCR ratio for
DISCO-TAB vs the diffusion baseline TabDDPM. Reads long.csv; writes the PDF into
the paper directory.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LONG = os.path.join(HERE, "tables", "comprehensive", "long.csv")
OUT = "/home/ec2-user/P-TA-MulFee/2026_JBHI___DISCO_Tab/fig_privacy_combined.pdf"
ACCENT, ORANGE, NEUTRAL, EDGE = "#0072B2", "#D55E00", "#BDBDBD", "#444"
MLAB = {"disco": "DISCO-TAB", "ctgan": "CTGAN", "tvae": "TVAE", "tabddpm": "TabDDPM",
        "adsgan": "ADS-GAN", "pategan": "PATE-GAN", "rtvae": "RTVAE", "nflow": "NFlow",
        "marginal": "Marginals", "great": "GReaT", "realtabformer": "RTF",
        "taegan": "TAEGAN", "tagal": "TAGAL", "tabsyn": "TabSyn"}
DROP = {"arf", "bayesian_network", "gaussian_copula"}
DLAB = lambda d: d.replace("surv_", "").replace("tab_", "").replace("_", " ").title()


def main():
    L = pd.read_csv(LONG)
    L = L[L.dataset != "tab_credit_g"]
    L["ratio"] = L["dcr_median"] / L["dcr_realreal_median"]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 3.0),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})

    # ---- panel (a): utility vs mean DCR ratio ----
    u = L.groupby("model").tstr_f1_macro.mean()
    r = L.groupby("model").ratio.mean()
    axa.axvline(1.0, color="#888", linestyle=(0, (3, 2)), linewidth=0.9)
    for m in u.index:
        if m in DROP or m not in MLAB or np.isnan(r.get(m, np.nan)):
            continue
        x, y = r[m], u[m]
        if m == "disco":
            axa.scatter(x, y, s=150, marker="*", color=ACCENT, edgecolor=EDGE, lw=0.6, zorder=5)
            axa.annotate("DISCO-TAB", (x, y), fontsize=8, fontweight="bold", color=ACCENT,
                         xytext=(5, 4), textcoords="offset points")
        else:
            axa.scatter(x, y, s=30, color=NEUTRAL, edgecolor=EDGE, lw=0.4, zorder=3)
            axa.annotate(MLAB[m], (x, y), fontsize=6, color="#666",
                         xytext=(3, -6), textcoords="offset points")
    axa.set_xlabel("Mean DCR ratio (1 = calibrated)", fontsize=8.5)
    axa.set_ylabel("Mean TSTR F1-macro ($\\uparrow$)", fontsize=8.5)
    axa.set_title("(a) Utility--privacy trade-off", fontsize=9)

    # ---- panel (b): per-dataset DCR ratio, DISCO vs TabDDPM ----
    piv = L.pivot_table(index="dataset", columns="model", values="ratio", aggfunc="mean")
    dss = [d for d in piv.index if not np.isnan(piv.loc[d].get("disco", np.nan))]
    dss = sorted(dss, key=lambda d: piv.loc[d, "disco"])
    y = np.arange(len(dss))
    disco = [piv.loc[d, "disco"] for d in dss]
    tabd = [piv.loc[d].get("tabddpm", np.nan) for d in dss]
    XMAX = 4.0
    axb.axvline(1.0, color="#888", linestyle=(0, (3, 2)), linewidth=0.9)
    axb.scatter(np.clip(tabd, 0, XMAX), y, s=20, marker="D", color=ORANGE,
                edgecolor=EDGE, lw=0.3, label="TabDDPM", zorder=3)
    axb.scatter(np.clip(disco, 0, XMAX), y, s=22, marker="o", color=ACCENT,
                edgecolor=EDGE, lw=0.3, label="DISCO-TAB", zorder=4)
    for i, t in enumerate(tabd):
        if not np.isnan(t) and t > XMAX:
            axb.annotate(f"{t:.1f}$\\times$", (XMAX, i), fontsize=5.3, color=ORANGE,
                         va="center", ha="right", xytext=(-1, 0), textcoords="offset points")
    axb.set_yticks(y)
    axb.set_yticklabels([DLAB(d) for d in dss], fontsize=5.3)
    axb.set_xlim(0, XMAX + 0.15)
    axb.set_xlabel("DCR ratio (synthetic / real--real)", fontsize=8.5)
    axb.set_title("(b) Per-dataset disclosure risk", fontsize=9)
    axb.legend(fontsize=6.5, loc="lower right", frameon=False)

    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
