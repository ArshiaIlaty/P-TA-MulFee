"""
Paper-revision summary tables (compact, 14-page budget). Reads
tables/comprehensive/long.csv and writes tables/comprehensive/paper/:

  Table 1  paper_table1_aggregate.tex  -- models as columns; rows grouped by
           evaluation dimension (rotated dimension label) with its metrics;
           each cell = mean across datasets (normal font) with the 95% CI in a
           smaller font beneath; bold = best model per metric row.
  Table 2  paper_table2_vs_best.tex    -- per metric: DISCO-TAB's score vs the
           single best competing method (named) and its score.
  paper_tables_preview.pdf             -- both compiled for review.

Choices for the revision (per authors): FAITH dropped; ARF and Bayesian network
dropped; SDMetrics added. The aggregate uses the 10 models with full 24-dataset
coverage so the averages are comparable; the LLM-era subset baselines (GReaT,
RealTabFormer, TAEGAN, TAGAL) are reported separately (subset table).
"""
import os, subprocess
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "tables", "comprehensive")
OUT = os.path.join(C, "paper")
os.makedirs(OUT, exist_ok=True)

MODELS = ["disco", "ctgan", "tvae", "gaussian_copula", "tabddpm", "adsgan",
          "pategan", "rtvae", "nflow", "marginal"]
MLAB = {"disco": "DISCO-TAB", "ctgan": "CTGAN", "tvae": "TVAE",
        "gaussian_copula": "GaussCop", "tabddpm": "TabDDPM", "adsgan": "ADS-GAN",
        "pategan": "PATE-GAN", "rtvae": "RTVAE", "nflow": "NFlow",
        "marginal": "Marginals"}

# dimension -> [(metric, direction, short label)]; direction: up|down|half(=0.5 ideal)
DIMENSIONS = [
    ("Downstream\\\\utility (TSTR)", [
        ("tstr_f1_macro", "up", "F1-macro $\\uparrow$"),
        ("tstr_f1_weighted", "up", "F1-weighted $\\uparrow$"),
        ("tstr_accuracy", "up", "Accuracy $\\uparrow$"),
    ]),
    ("Statistical\\\\fidelity", [
        ("fid_JS", "down", "JS div. $\\downarrow$"),
        ("fid_KS", "down", "KS stat. $\\downarrow$"),
        ("fid_Hellinger", "down", "Hellinger $\\downarrow$"),
        ("fid_Pearson", "up", "Corr. pres. $\\uparrow$"),
    ]),
    ("Privacy /\\\\disclosure", [
        ("dcr_median", "up", "DCR $\\uparrow$"),
        ("ext_nndr", "up", "NNDR $\\uparrow$"),
        ("ext_mia_auc", "half", "MIA AUROC $\\to$.5"),
        ("ext_dup_rate", "down", "Dup. rate $\\downarrow$"),
    ]),
    ("Diversity /\\\\coverage (PRDC)", [
        ("ext_precision", "up", "Precision $\\uparrow$"),
        ("ext_recall", "up", "Recall $\\uparrow$"),
        ("ext_density", "up", "Density $\\uparrow$"),
        ("ext_coverage", "up", "Coverage $\\uparrow$"),
    ]),
    ("Detection", [
        ("ext_c2st", "down", "C2ST $\\downarrow$"),
        ("ext_mmd", "down", "MMD $\\downarrow$"),
        ("ext_wasserstein", "down", "Wasserstein $\\downarrow$"),
    ]),
    ("SDMetrics", [
        ("ext_sdm_quality", "up", "Quality $\\uparrow$"),
        ("ext_sdm_shapes", "up", "Col. shapes $\\uparrow$"),
        ("ext_sdm_pairs", "up", "Pair trends $\\uparrow$"),
    ]),
]


def agg(df, metric):
    """mean over seeds per (model,dataset), then mean + 95% t-CI across datasets."""
    if metric not in df.columns:
        return {}
    g = df.groupby(["model", "dataset"])[metric].mean().reset_index()
    out = {}
    for m, sub in g.groupby("model"):
        v = sub[metric].dropna().values
        if len(v) == 0:
            continue
        mean = float(np.mean(v))
        ci = (float(stats.t.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v)))
              if len(v) > 1 else np.nan)
        out[m] = (mean, ci, len(v))
    return out


def best_idx(stats_by_model, direction):
    cand = {m: v[0] for m, v in stats_by_model.items() if m in MODELS}
    if not cand:
        return None
    if direction == "up":
        return max(cand, key=cand.get)
    if direction == "down":
        return min(cand, key=cand.get)
    return min(cand, key=lambda m: abs(cand[m] - 0.5))   # half


def fmt(mean, ci):
    s = f"{mean:.3f}"
    if ci is None or np.isnan(ci):
        return s
    return f"\\shortstack{{{s}\\\\{{\\scriptsize$\\pm${ci:.3f}}}}}"


def table1(df):
    lines = [
        r"\begin{table*}[t]\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.1}",
        r"\caption{Benchmark summary: mean over all datasets (95\% CI beneath, smaller "
        r"font) for each metric and generator. \textbf{Bold} = best per row. "
        r"Arrows give the preferred direction; MIA AUROC is best near 0.5.}",
        r"\label{tab:aggregate}",
        r"\begin{tabular}{@{}cl" + "c" * len(MODELS) + r"@{}}",
        r"\toprule",
        r"& \textbf{Metric} & " + " & ".join(r"\textbf{%s}" % MLAB[m] for m in MODELS) + r" \\",
    ]
    for dim, metrics in DIMENSIONS:
        lines.append(r"\midrule")
        n = len(metrics)
        for j, (metric, direction, lab) in enumerate(metrics):
            sbm = agg(df, metric)
            bi = best_idx(sbm, direction)
            if j == 0:
                cell0 = r"\multirow{%d}{*}{\rotatebox[origin=c]{90}{\shortstack{%s}}}" % (n, dim)
            else:
                cell0 = ""
            row = [cell0, lab]
            for m in MODELS:
                if m in sbm:
                    mean, ci, _ = sbm[m]
                    c = fmt(mean, ci)
                    if m == bi:
                        c = r"\textbf{%s}" % c
                else:
                    c = "--"
                row.append(c)
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def table2(df):
    lines = [
        r"\begin{table}[t]\centering\small",
        r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.15}",
        r"\caption{DISCO-TAB vs.\ the single best competing generator per metric "
        r"(mean over all datasets $\pm$95\% CI). \textbf{Bold} marks the winner.}",
        r"\label{tab:vsbest}",
        r"\begin{tabular}{@{}clccl@{}}",
        r"\toprule",
        r"& \textbf{Metric} & \textbf{DISCO-TAB} & \textbf{Best competitor} & \textbf{(method)} \\",
    ]
    for dim, metrics in DIMENSIONS:
        lines.append(r"\midrule")
        n = len(metrics)
        for j, (metric, direction, lab) in enumerate(metrics):
            sbm = agg(df, metric)
            if "disco" not in sbm:
                continue
            dmean, dci, _ = sbm["disco"]
            others = {m: v for m, v in sbm.items() if m in MODELS and m != "disco"}
            bi = best_idx(others, direction)
            cell0 = (r"\multirow{%d}{*}{\rotatebox[origin=c]{90}{\shortstack{%s}}}" % (n, dim)
                     if j == 0 else "")
            if bi is None:
                lines.append(" & ".join([cell0, lab, fmt(dmean, dci), "--", "--"]) + r" \\")
                continue
            bmean, bci, _ = others[bi]
            # does DISCO win?
            if direction == "up":
                disco_wins = dmean >= bmean
            elif direction == "down":
                disco_wins = dmean <= bmean
            else:
                disco_wins = abs(dmean - 0.5) <= abs(bmean - 0.5)
            d_cell = fmt(dmean, dci)
            b_cell = fmt(bmean, bci)
            name = MLAB[bi]
            if disco_wins:
                d_cell = r"\textbf{%s}" % d_cell
            else:
                b_cell = r"\textbf{%s}" % b_cell
                name = r"\textbf{%s}" % name
            lines.append(" & ".join([cell0, lab, d_cell, b_cell, name]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    df = pd.read_csv(os.path.join(C, "long.csv"))
    t1 = table1(df)
    t2 = table2(df)
    open(os.path.join(OUT, "paper_table1_aggregate.tex"), "w").write(t1 + "\n")
    open(os.path.join(OUT, "paper_table2_vs_best.tex"), "w").write(t2 + "\n")
    doc = (r"""\documentclass[10pt]{article}
\usepackage[landscape,margin=1cm]{geometry}
\usepackage{booktabs,multirow,graphicx,amssymb,amsmath}
\renewcommand\theadfont{\scriptsize}
\begin{document}
""" + t1 + "\n\n\\clearpage\n\\begin{landscape}\n" + "" + t2 +
           "\n\\end{document}\n")
    # table2 is portrait-friendly; keep simple single-doc preview
    doc = (r"""\documentclass[10pt]{article}
\usepackage[landscape,margin=1cm]{geometry}
\usepackage{booktabs,multirow,graphicx,amssymb,amsmath}
\begin{document}
""" + t1 + "\n\n\\clearpage\n" + t2 + "\n\\end{document}\n")
    tex = os.path.join(OUT, "paper_tables_preview.tex")
    open(tex, "w").write(doc)
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        os.path.basename(tex)], cwd=OUT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote:", os.listdir(OUT))


if __name__ == "__main__":
    main()
