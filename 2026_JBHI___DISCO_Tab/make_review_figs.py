#!/usr/bin/env python3
"""Results-section figure: empirical disclosure-risk (DCR) ratio across the whole
benchmark, computed from the committed per-run JSONs in ../disco/results.

fig_privacy.pdf -- for every dataset, the median distance to the closest real
record divided by the within-dataset real-real baseline, for DISCO-TAB and
TabDDPM. A ratio near 1 (dashed line) means synthetic records sit about as far
from real records as real records sit from each other. DISCO-TAB stays near 1
everywhere; TabDDPM inflates sharply on several small/imbalanced datasets,
the ``privacy via noise'' artifact.
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "disco", "results")
plt.rcParams.update({
    "font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
    "axes.edgecolor": "#5a5a5a", "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": "#5a5a5a",
    "ytick.color": "#5a5a5a", "figure.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
BLUE, ORANGE, GRID, INK, MUTED = "#0072B2", "#D55E00", "#E6E6E6", "#222222", "#8a8a8a"

DS = {"heart_failure": "Heart Failure", "breast_cancer": "Breast Cancer",
      "liver_disorders": "Liver Disorders", "parkinsons": "Parkinson's",
      "obesity": "Obesity", "diabetes": "Diabetes", "german_credit": "German Credit",
      "bank_marketing": "Bank Marketing", "census_income": "Census Income",
      "surv_melanoma": "Melanoma", "surv_pbc": "PBC", "surv_pbc3": "PBC3",
      "surv_retinopathy": "Retinopathy", "surv_whas500": "WHAS500",
      "surv_gbsg2": "GBSG2", "surv_colon": "Colon", "surv_grace": "GRACE",
      "surv_trace": "TRACE", "surv_framingham": "Framingham",
      "tab_blood_transfusion": "Blood Transf.", "tab_qsar_biodeg": "QSAR Biodeg",
      "tab_maternal_risk": "Maternal Risk", "tab_website_phishing": "Website Phish."}


def mean(ds, tag, key):
    xs = []
    for f in sorted(glob.glob(os.path.join(RES, f"{ds}__{tag}__seed*.json"))):
        try:
            v = json.load(open(f)).get(key)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                xs.append(float(v))
        except Exception:
            pass
    return float(np.mean(xs)) if xs else None


rows = []
for ds in DS:
    base = mean(ds, "disco", "dcr_realreal_median")
    d = mean(ds, "disco", "dcr_median")
    t = mean(ds, "tabddpm", "dcr_median")
    if base and d and t:
        rows.append((DS[ds], d / base, t / base))
rows.sort(key=lambda r: r[2])                       # by TabDDPM ratio
labels = [r[0] for r in rows]
disco = np.array([r[1] for r in rows])
tabddpm = np.array([r[2] for r in rows])
y = np.arange(len(rows))
XMAX = 3.0

fig, ax = plt.subplots(figsize=(3.5, 5.0))
ax.xaxis.grid(True, color=GRID, linewidth=0.6)
ax.axvline(1.0, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.9)
ax.text(1.0, len(rows) - 0.2, "real-real\nbaseline", ha="center", va="bottom",
        fontsize=5.6, color=MUTED)
ax.scatter(np.clip(tabddpm, 0, XMAX), y, s=22, marker="D", color=ORANGE,
           label="TabDDPM", zorder=4)
ax.scatter(np.clip(disco, 0, XMAX), y, s=22, marker="o", color=BLUE,
           label="DISCO-TAB", zorder=5)
for i, v in enumerate(tabddpm):                      # annotate off-scale inflation
    if v > XMAX:
        ax.annotate(f"{v:.1f}$\\times$", (XMAX, i), xytext=(XMAX - 0.02, i),
                    ha="right", va="center", fontsize=5.6, color=ORANGE,
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.6))
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=6.0)
ax.set_xlim(0.3, XMAX + 0.05)
ax.set_ylim(-0.6, len(rows) + 0.4)
ax.set_xlabel("DCR ratio to real-real baseline", fontsize=8)
ax.legend(fontsize=7, frameon=False, loc="lower right")
ax.tick_params(length=0)
fig.tight_layout(pad=0.25)
fig.savefig("fig_privacy.pdf")
plt.close(fig)
print("Wrote fig_privacy.pdf")
