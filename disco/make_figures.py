"""
Paper-ready result figures for DISCO-TAB (vector PDF + PNG, 300 dpi).
Reads tables/comprehensive/*.csv. Colorblind-safe Okabe-Ito palette; DISCO
highlighted; direct value labels (covers the low-contrast-accent case).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "tables", "comprehensive")
FIG = os.path.join(C, "figures")
os.makedirs(FIG, exist_ok=True)

ACCENT = "#0072B2"   # DISCO-TAB
NEG = "#D55E00"      # vermillion (DISCO loses)
SER2 = "#E69F00"     # second series (GReaT)
NEUTRAL = "#BDBDBD"
EDGE = "#444444"
INK = "#222222"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "figure.dpi": 300,
                     "axes.linewidth": 0.6, "axes.edgecolor": "#666",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight"})

MLAB = {"disco": "DISCO-TAB", "ctgan": "CTGAN", "tvae": "TVAE",
        "gaussian_copula": "GaussCopula", "tabddpm": "TabDDPM", "adsgan": "ADS-GAN",
        "pategan": "PATE-GAN", "rtvae": "RTVAE", "nflow": "NFlow", "arf": "ARF",
        "bayesian_network": "BayesNet", "marginal": "Marginals",
        "great": "GReaT", "realtabformer": "RealTabFormer",
        "taegan": "TAEGAN", "tagal": "TAGAL"}
DLAB = lambda d: d.replace("surv_", "").replace("tab_", "").replace("_", " ").title()
# ARF + Bayesian network dropped from the analysis (authors' revision choice).
DROP = {"arf", "bayesian_network"}
CLASSIC = ["ctgan", "tvae", "gaussian_copula", "adsgan", "pategan", "rtvae",
           "nflow", "marginal"]
MEDICAL = {"heart_failure", "breast_cancer", "liver_disorders", "parkinsons",
           "obesity", "diabetes", "surv_melanoma", "surv_pbc", "surv_pbc3",
           "surv_retinopathy", "surv_whas500", "surv_gbsg2", "surv_colon",
           "surv_grace", "surv_trace", "surv_framingham", "tab_maternal_risk"}
ORDER = ["heart_failure", "breast_cancer", "liver_disorders", "parkinsons", "obesity",
         "diabetes", "german_credit", "bank_marketing", "census_income",
         "surv_melanoma", "surv_pbc", "surv_pbc3", "surv_retinopathy", "surv_whas500",
         "surv_gbsg2", "surv_colon", "surv_grace", "surv_trace", "surv_framingham",
         "tab_blood_transfusion", "tab_qsar_biodeg",
         "tab_maternal_risk", "tab_website_phishing"]


def mean_df(metric):
    return pd.read_csv(os.path.join(C, f"{metric}__datasets_x_models.csv"), index_col=0)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)


# ---- Fig 1: utility leaderboard (mean TSTR F1 across datasets, DISCO highlighted) ----
def fig_leaderboard():
    d = mean_df("tstr_f1_macro")
    stats = {m: (d[m].mean(), d[m].sem() * 1.96, d[m].notna().sum())
             for m in d.columns if d[m].notna().sum() >= 8 and m not in DROP}
    s = pd.DataFrame(stats, index=["mean", "ci", "n"]).T.sort_values("mean")
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    colors = [ACCENT if m == "disco" else NEUTRAL for m in s.index]
    ax.barh(range(len(s)), s["mean"], xerr=s["ci"], color=colors, edgecolor=EDGE,
            linewidth=0.5, height=0.72, error_kw=dict(ecolor="#777", lw=0.8, capsize=2))
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([MLAB[m] + ("  (ours)" if m == "disco" else "") for m in s.index],
                       fontsize=8)
    for i, (m, r) in enumerate(s.iterrows()):
        ax.text(r["mean"] + r["ci"] + 0.008, i, f"{r['mean']:.3f}", va="center",
                fontsize=7, color=INK)
    ax.set_xlabel(f"Mean TSTR F1-macro across {len(d.index)} datasets (↑)", fontsize=8.5)
    ax.set_xlim(0, min(1.0, s["mean"].max() + 0.12))
    ax.set_title("Downstream utility: DISCO-TAB vs. all generators",
                 fontsize=9.5, color=INK)
    save(fig, "fig1_utility_leaderboard")
    return s


# ---- Fig 2: per-dataset margin DISCO - best established baseline (diverging) ----
def fig_margin():
    d = mean_df("tstr_f1_macro")
    base = [m for m in CLASSIC if m in d.columns]
    rows = []
    for ds in [x for x in ORDER if x in d.index]:
        if pd.isna(d.loc[ds, "disco"]):
            continue
        bb = d.loc[ds, base].dropna()
        if bb.empty:
            continue
        rows.append((ds, d.loc[ds, "disco"] - bb.max()))
    # medical first, then others; within group by value
    rows.sort(key=lambda r: (r[0] not in MEDICAL, r[1]))
    labels = [DLAB(r[0]) for r in rows]
    vals = [r[1] for r in rows]
    med = [r[0] in MEDICAL for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    colors = [ACCENT if v >= 0 else NEG for v in vals]
    ax.barh(range(len(rows)), vals, color=colors, edgecolor=EDGE, linewidth=0.5,
            height=0.74)
    ax.axvline(0, color="#333", lw=0.8)
    # Headroom on both sides so value labels sit in the margin, clear of the
    # y-tick dataset names (the big negative bars otherwise collide at the spine).
    lo, hi = min(vals), max(vals)
    ax.set_xlim(lo - 0.065, hi + 0.035)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{l}{'  (clinical)' if m else ''}" for l, m in zip(labels, med)],
                       fontsize=7)
    for i, v in enumerate(vals):
        ax.text(v + (0.004 if v >= 0 else -0.004), i, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=6.3, color=INK)
    ax.set_xlabel("ΔF1 = DISCO-TAB − best established baseline", fontsize=8.5)
    ax.set_title(f"Per-dataset utility margin vs. best-of-{len(base)} established baselines\n(blue = DISCO-TAB wins)",
                 fontsize=9.5, color=INK)
    wins = sum(v > 0 for v in vals)
    ax.text(0.98, 0.02, f"DISCO-TAB best on {wins}/{len(vals)} datasets",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=ACCENT)
    save(fig, "fig2_utility_margin")


# ---- Fig 3: DISCO vs GReaT (unconstrained LLM) on shared datasets ----
def fig_llm():
    d = mean_df("tstr_f1_macro")
    if "great" not in d.columns:
        return
    shared = [ds for ds in ORDER if ds in d.index and not pd.isna(d.loc[ds, "disco"])
              and not pd.isna(d.loc[ds, "great"])]
    if not shared:
        return
    disco = [d.loc[ds, "disco"] for ds in shared]
    great = [d.loc[ds, "great"] for ds in shared]
    x = np.arange(len(shared)); w = 0.38
    fig, ax = plt.subplots(figsize=(1.1 + 0.9 * len(shared), 3.2))
    ax.bar(x - w / 2, disco, w, label="DISCO-TAB", color=ACCENT, edgecolor=EDGE, lw=0.5)
    ax.bar(x + w / 2, great, w, label="GReaT (vanilla LLM)", color=SER2, edgecolor=EDGE, lw=0.5)
    for i, (a, b) in enumerate(zip(disco, great)):
        ax.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=6.5, color=INK)
        ax.text(i + w / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([DLAB(s) for s in shared], fontsize=7.5)
    ax.set_ylabel("TSTR F1-macro (↑)", fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("DISCO-TAB vs. unconstrained LLM generation (GReaT)",
                 fontsize=9.5, color=INK)
    save(fig, "fig3_vs_llm")


# ---- Fig 5: head-to-head utility win-rate (DISCO vs each baseline over datasets) ----
def fig_winrate():
    d = mean_df("tstr_f1_macro")
    if "disco" not in d.columns:
        return
    rows = []
    for m in d.columns:
        if m == "disco" or m in DROP:
            continue
        both = d[["disco", m]].dropna()
        if len(both) < 12:   # keep the 24-dataset figure to fully-covered baselines
            continue
        n = len(both)
        wins = int((both["disco"] > both[m] + 1e-9).sum())
        rate = wins / n
        rows.append((m, rate, wins, n))
    rows.sort(key=lambda r: r[1])
    labels = [MLAB[m] for m, *_ in rows]
    rates = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    colors = [ACCENT if r >= 0.5 else NEG for r in rates]
    ax.barh(range(len(rows)), [r * 100 for r in rates], color=colors, edgecolor=EDGE,
            linewidth=0.5, height=0.72)
    ax.axvline(50, color="#333", lw=0.9, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    for i, (m, rate, wins, n) in enumerate(rows):
        ax.text(rate * 100 + 1.2, i, f"{rate*100:.0f}%  ({wins}/{n})", va="center",
                fontsize=7, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of datasets where DISCO-TAB wins on TSTR F1-macro", fontsize=8.5)
    ax.set_title("Head-to-head downstream-utility win rate (dashed = 50%)",
                 fontsize=9.5, color=INK)
    save(fig, "fig5_winrate")


# ---- Fig 4: utility vs fidelity trade-off (scatter) ----
def fig_tradeoff():
    u = mean_df("tstr_f1_macro"); f = mean_df("fid_JS")
    models = [m for m in u.columns if m in f.columns and u[m].notna().sum() >= 8
              and m not in DROP]
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    for m in models:
        x, y = f[m].mean(), u[m].mean()
        if m == "disco":
            ax.scatter(x, y, s=150, marker="*", color=ACCENT, edgecolor=EDGE,
                       lw=0.6, zorder=5)
            ax.annotate("DISCO-TAB", (x, y), fontsize=8.5, fontweight="bold",
                        color=ACCENT, xytext=(6, 4), textcoords="offset points")
        else:
            ax.scatter(x, y, s=34, color=NEUTRAL, edgecolor=EDGE, lw=0.4, zorder=3)
            ax.annotate(MLAB[m], (x, y), fontsize=6.3, color="#666",
                        xytext=(4, -6), textcoords="offset points")
    ax.set_xlabel("Mean JS divergence (fidelity, ↓ better)", fontsize=8.5)
    ax.set_ylabel("Mean TSTR F1-macro (utility, ↑ better)", fontsize=8.5)
    ax.set_title("Utility–fidelity trade-off (top-left = useful)",
                 fontsize=9.5, color=INK)
    save(fig, "fig4_utility_fidelity")


if __name__ == "__main__":
    s = fig_leaderboard()
    fig_margin()
    fig_llm()
    fig_tradeoff()
    fig_winrate()
    print("wrote figures to", FIG)
    print(os.listdir(FIG))
