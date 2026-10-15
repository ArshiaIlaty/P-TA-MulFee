"""
Generate PAPER_UPDATES.md: every regenerated number + LaTeX-ready tables +
provenance/caveats, mapped to the manuscript's RED placeholders. Reads the
assembled results (run assemble_tables.py + stats.py first, or let the driver
do it). Re-runnable: regenerates with the latest results each time.
"""
import os, json, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "tables")
ORDER = ["heart_failure", "breast_cancer", "liver_disorders", "parkinsons",
         "obesity", "diabetes", "german_credit", "bank_marketing", "census_income"]
LAB = {d: d.replace("_", " ").title() for d in ORDER}
LAB["parkinsons"] = "Parkinson's"
MAIN = ["gaussian_copula", "ctgan", "tvae", "tabddpm", "disco"]
MLAB = {"gaussian_copula": "GaussCopula", "ctgan": "CTGAN", "tvae": "TVAE",
        "tabddpm": "TabDDPM", "disco": "DISCO-TAB"}


def load():
    rows = []
    for p in glob.glob(os.path.join(HERE, "results", "*.json")):
        d = json.load(open(p))
        rows.append(d)
    return pd.DataFrame([{**{k: d.get(k) for k in (
        "dataset", "tag", "seed", "n_syn", "valid_rate", "tstr_f1_macro",
        "tstr_f1_weighted", "fid_JS", "fid_KL", "fid_Hellinger", "fid_KS",
        "fid_Pearson", "fid_JS_num", "fid_JS_cat", "dcr_median",
        "dcr_realreal_median", "faith_FAITH", "faith_Factuality",
        "faith_Alignment", "faith_Integrity", "faith_Tracking")}} for d in rows])


def fmt(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def md_table(df, metric, methods):
    piv = df[df.tag.isin(methods)].pivot_table(index="dataset", columns="tag",
                                                values=metric, aggfunc="mean")
    piv = piv.reindex([d for d in ORDER if d in piv.index])
    cols = [m for m in methods if m in piv.columns]
    bases = [m for m in cols if m != "disco"]
    head = "| Dataset | " + " | ".join(MLAB[m] for m in cols)
    if "disco" in cols and bases:
        head += " | Δ(best base) |"
    else:
        head += " |"
    sep = "|" + "---|" * (len(cols) + 1 + (1 if ("disco" in cols and bases) else 0))
    lines = [head, sep]
    for ds in piv.index:
        row = [LAB[ds]] + [fmt(piv.loc[ds, m]) for m in cols]
        if "disco" in cols and bases:
            bvals = [piv.loc[ds, m] for m in bases if not pd.isna(piv.loc[ds, m])]
            dv = piv.loc[ds, "disco"]
            delta = (dv - max(bvals)) if (bvals and not pd.isna(dv)) else np.nan
            row.append(("**%+.3f**" % delta) if not pd.isna(delta) else "--")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    df = load()
    df = df[df.n_syn.fillna(0) >= 10]
    present = [m for m in MAIN if m in set(df.tag)]
    out = ["# DISCO-TAB — Regenerated Results (real, reproducible)\n",
           f"_Generated from {len(df)} scored runs. Single Tesla T4; GPT-2 backbone. "
           "All methods scored under one protocol (`disco/metrics.py`)._\n",
           "> These are freshly computed numbers from the rebuilt pipeline. They "
           "replace the lost-server values. Baselines reproduced here: "
           f"{', '.join(MLAB[m] for m in present if m!='disco')}. "
           "Paper baselines not regenerated (no surviving code/weights): "
           "TAEGAN, TabSyn, RealTabFormer, MostlyAI, TAGAL.\n"]

    out.append("## Table I — Downstream utility, TSTR macro-F1 (mean over seeds)\n")
    out.append(md_table(df, "tstr_f1_macro", present) + "\n")
    out.append("## Table I(b) — TSTR weighted-F1 (mean)\n")
    out.append(md_table(df, "tstr_f1_weighted", present) + "\n")

    out.append("## Table II — Statistical fidelity (DISCO-TAB, mean; lower JS/KL/Hell/KS better, Pearson higher better)\n")
    d = df[df.tag == "disco"].groupby("dataset")[["fid_JS", "fid_KL", "fid_Hellinger",
         "fid_KS", "fid_Pearson"]].mean().reindex([x for x in ORDER if x in set(df.dataset)])
    out.append("| Dataset | JS | KL | Hellinger | KS | Pearson |\n|---|---|---|---|---|---|")
    for ds in d.index:
        out.append(f"| {LAB[ds]} | " + " | ".join(fmt(d.loc[ds, c]) for c in d.columns) + " |")
    out.append("")

    out.append("## Table III — Empirical disclosure risk (DCR, DISCO-TAB) vs real–real baseline\n")
    dd = df[df.tag == "disco"].groupby("dataset")[["tstr_f1_macro", "dcr_median",
          "dcr_realreal_median"]].mean().reindex([x for x in ORDER if x in set(df.dataset)])
    out.append("| Dataset | Utility (F1) | DCR (median) | real–real DCR |\n|---|---|---|---|")
    for ds in dd.index:
        out.append(f"| {LAB[ds]} | " + " | ".join(fmt(dd.loc[ds, c]) for c in dd.columns) + " |")
    out.append("")

    out.append("## FAITH decomposition (DISCO-TAB, mean)\n")
    fa = df[df.tag == "disco"].groupby("dataset")[["faith_FAITH", "faith_Factuality",
         "faith_Alignment", "faith_Integrity", "faith_Tracking"]].mean().reindex(
         [x for x in ORDER if x in set(df.dataset)])
    out.append("| Dataset | FAITH | Factuality | Alignment | Integrity | Tracking |\n|---|---|---|---|---|---|")
    for ds in fa.index:
        out.append(f"| {LAB[ds]} | " + " | ".join(fmt(fa.loc[ds, c]) for c in fa.columns) + " |")
    out.append("")

    out.append("## Feature-type JSD (mean per method; lower better)\n")
    ft = df.groupby("tag")[["fid_JS_num", "fid_JS_cat"]].mean()
    out.append("| Method | Numerical JSD | Categorical JSD |\n|---|---|---|")
    for m in present:
        if m in ft.index:
            out.append(f"| {MLAB[m]} | {fmt(ft.loc[m,'fid_JS_num'])} | {fmt(ft.loc[m,'fid_JS_cat'])} |")
    out.append("")

    # ablation
    abl = os.path.join(TAB, "ablation.csv")
    if os.path.exists(abl):
        out.append("## Ablation (relative TSTR F1 vs full system)\n")
        a = pd.read_csv(abl)
        out.append("| Config | Rel. perf. | SD | mean F1 | datasets |\n|---|---|---|---|---|")
        for _, r in a.iterrows():
            out.append(f"| {r['config']} | {r['rel_perf_%']}% | {r['rel_sd_%']} | "
                       f"{r['mean_f1']} | {r.get('n','')} |")
        out.append("")

    # stats
    wil = os.path.join(TAB, "wilcoxon_vs_disco.csv")
    if os.path.exists(wil):
        out.append("## Significance (DISCO-TAB vs baselines; Wilcoxon signed-rank, Holm-corrected)\n")
        w = pd.read_csv(wil)
        out.append("| Baseline | n | DISCO mean | base mean | Δ | wins | p | p(Holm) | sig@.05 |\n|---|---|---|---|---|---|---|---|---|")
        for _, r in w.iterrows():
            out.append(f"| {MLAB.get(r['method'], r['method'])} | {int(r['n'])} | "
                       f"{fmt(r.get('disco_mean'))} | {fmt(r.get('base_mean'))} | "
                       f"{fmt(r.get('mean_diff'))} | {int(r.get('wins',0))} | "
                       f"{fmt(r.get('p'),4)} | {fmt(r.get('p_holm'),4)} | {r.get('sig_holm_0.05')} |")
        out.append("")

    with open(os.path.join(HERE, "PAPER_UPDATES_RESULTS.md"), "w") as f:
        f.write("\n".join(out))
    print("wrote PAPER_UPDATES_RESULTS.md (", len(df), "runs )")


if __name__ == "__main__":
    main()
