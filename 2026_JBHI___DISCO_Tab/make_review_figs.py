#!/usr/bin/env python3
"""Generate the two summary figures requested by Reviewer 1 (JBHI-00758-2026).

Values are taken verbatim from Table I (relative utility gain, Delta F1) and
Table II (per-dataset Jensen-Shannon divergence for DISCO-TAB). No values are
fabricated: these are the point estimates already reported in the manuscript.
Once repeated-run data are available, replace `yerr=None` with the per-dataset
standard deviations to obtain error bars.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

# Colorblind-safe (Okabe-Ito) accents; print-friendly.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#999999"

datasets = ["Heart\nFailure", "Breast\nCancer", "Liver\nDisorders", "Parkinson's",
            "Obesity", "Diabetes", "German\nCredit", "Bank\nMkt.", "Census\nInc."]

# --- Figure 1: Relative utility gain (Delta F1) over strongest baseline ---
delta = [0.123, 0.004, 0.382, 0.080, 0.027, -0.104, 0.275, -0.014, -0.006]
# std_delta = [...]  # <-- insert per-dataset SD from repeated runs, then set yerr

fig, ax = plt.subplots(figsize=(3.4, 2.1))
colors = [BLUE if v >= 0 else ORANGE for v in delta]
bars = ax.bar(range(len(datasets)), delta, color=colors, width=0.68,
              edgecolor="black", linewidth=0.4)  # yerr=std_delta, capsize=2
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets, fontsize=6.2)
ax.set_ylabel(r"$\Delta$F1 vs. best baseline")
ax.axvline(5.5, color=GRAY, linestyle=":", linewidth=0.7)
ax.text(1.0, 0.47, "Medical", ha="center", fontsize=6.5, color=GRAY)
ax.text(7.0, 0.47, "Non-medical", ha="center", fontsize=6.5, color=GRAY)
for i, v in enumerate(delta):
    ax.text(i, v + (0.012 if v >= 0 else -0.028), f"{v:+.3f}",
            ha="center", fontsize=5.3)
ax.set_ylim(-0.18, 0.52)
fig.tight_layout(pad=0.3)
fig.savefig("fig_utility_margin.pdf", bbox_inches="tight")
plt.close(fig)

# --- Figure 2: Per-dataset statistical fidelity (JSD, lower is better) ---
jsd = [0.003, 0.011, 0.017, 0.005, 0.008, 0.026, 0.001, 0.044, 0.074]
# std_jsd = [...]  # <-- insert per-dataset SD from repeated runs, then set yerr

fig, ax = plt.subplots(figsize=(3.4, 2.1))
bars = ax.bar(range(len(datasets)), jsd, color=BLUE, width=0.68,
              edgecolor="black", linewidth=0.4)  # yerr=std_jsd, capsize=2
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets, fontsize=6.2)
ax.set_ylabel(r"Jensen--Shannon div. (JSD) $\downarrow$")
ax.axvline(5.5, color=GRAY, linestyle=":", linewidth=0.7)
for i, v in enumerate(jsd):
    ax.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=5.3)
ax.set_ylim(0, 0.085)
fig.tight_layout(pad=0.3)
fig.savefig("fig_jsd_fidelity.pdf", bbox_inches="tight")
plt.close(fig)

print("Wrote fig_utility_margin.pdf and fig_jsd_fidelity.pdf")
