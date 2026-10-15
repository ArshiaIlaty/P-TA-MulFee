"""
Regenerate the paper's two review figures from REAL aggregated results:
  fig_utility_margin.pdf  -- per-dataset Delta F1 (DISCO-TAB - best baseline), +/- SD
  fig_jsd_fidelity.pdf    -- per-dataset DISCO-TAB JSD, +/- SD
Driven by results/*.json (via assemble_tables.load_long). Error bars appear once
>1 seed is available. Writes PDFs to tables/ and, if present, the paper dir.
"""
import os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from assemble_tables import load_long, DS_ORDER, DS_LABEL, _methods

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "2026_JBHI___DISCO_Tab")
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#999999"

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300})


def per_dataset_delta(df, dmeth, base):
    """Delta F1 per dataset (mean over seeds) + SD of DISCO's own F1."""
    out = {}
    for ds in DS_ORDER:
        sub = df[df["dataset"] == ds]
        dvals = sub[sub["method"] == dmeth]["tstr_f1_macro"].dropna()
        if dvals.empty:
            continue
        bbest = -np.inf
        for m in base:
            bv = sub[sub["method"] == m]["tstr_f1_macro"].dropna().mean()
            if np.isfinite(bv):
                bbest = max(bbest, bv)
        if not np.isfinite(bbest):
            continue
        out[ds] = (float(dvals.mean() - bbest), float(dvals.std() if len(dvals) > 1 else 0.0))
    return out


def per_dataset_jsd(df, dmeth):
    out = {}
    for ds in DS_ORDER:
        v = df[(df["dataset"] == ds) & (df["method"] == dmeth)]["fid_JS"].dropna()
        if not v.empty:
            out[ds] = (float(v.mean()), float(v.std() if len(v) > 1 else 0.0))
    return out


def labels(keys):
    return [DS_LABEL[k].replace(" ", "\n") for k in keys]


def main(copy_to_paper=False):
    df = load_long()
    base, dmeth = _methods(df)
    if not dmeth:
        print("no DISCO results yet"); return

    # Figure 1: Delta F1
    delta = per_dataset_delta(df, dmeth, base)
    keys = [k for k in DS_ORDER if k in delta]
    vals = [delta[k][0] for k in keys]; errs = [delta[k][1] for k in keys]
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    colors = [BLUE if v >= 0 else ORANGE for v in vals]
    ax.bar(range(len(keys)), vals, color=colors, width=0.68, edgecolor="black",
           linewidth=0.4, yerr=errs if any(errs) else None, capsize=2)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels(keys), fontsize=6.2)
    ax.set_ylabel(r"$\Delta$F1 vs. best baseline")
    for i, v in enumerate(vals):
        ax.text(i, v + (0.012 if v >= 0 else -0.028), f"{v:+.3f}", ha="center", fontsize=5.3)
    fig.tight_layout(pad=0.3)
    f1 = os.path.join(HERE, "tables", "fig_utility_margin.pdf")
    fig.savefig(f1, bbox_inches="tight"); plt.close(fig)

    # Figure 2: JSD
    jsd = per_dataset_jsd(df, dmeth)
    keys = [k for k in DS_ORDER if k in jsd]
    vals = [jsd[k][0] for k in keys]; errs = [jsd[k][1] for k in keys]
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    ax.bar(range(len(keys)), vals, color=BLUE, width=0.68, edgecolor="black",
           linewidth=0.4, yerr=errs if any(errs) else None, capsize=2)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels(keys), fontsize=6.2)
    ax.set_ylabel(r"Jensen--Shannon div. (JSD) $\downarrow$")
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v:.3f}", ha="center", fontsize=5.3)
    fig.tight_layout(pad=0.3)
    f2 = os.path.join(HERE, "tables", "fig_jsd_fidelity.pdf")
    fig.savefig(f2, bbox_inches="tight"); plt.close(fig)

    print("wrote", f1, "and", f2)
    if copy_to_paper and os.path.isdir(PAPER):
        for f in (f1, f2):
            shutil.copy(f, os.path.join(PAPER, os.path.basename(f)))
        print("copied figures into paper dir")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy-to-paper", action="store_true")
    args = ap.parse_args()
    main(copy_to_paper=args.copy_to_paper)
