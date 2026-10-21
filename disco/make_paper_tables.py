"""
Paper-revision summary tables sized for the JBHI IEEE two-column template
(ieeecolor.cls). Reads tables/comprehensive/long.csv; writes
tables/comprehensive/paper/:

  paper_table1_aggregate.tex  -- table* (spans both columns). Models across the
      top in two groups (Established generators | LLM-era), a #datasets coverage
      row, then rows grouped by evaluation dimension (single-line rotated label)
      with its metrics. Each cell: mean (table font) over a smaller-font CI.
      Bold = best per row among the fully-covered established generators.
  paper_table2_vs_best.tex    -- DISCO-TAB vs the best established competitor per
      metric (named + score).
  table_test.tex / .pdf       -- compiled with the real ieeecolor.cls for a fit
      check (no makecell; uses built-in \\shortstack).

Revision choices: FAITH dropped; ARF + Bayesian network dropped; SDMetrics added.
Established models have full 24-dataset coverage; the LLM-era models cover a
subset (coverage row makes this explicit) and are excluded from the "best"
bolding and from Table 2's competitor pool.
"""
import os, shutil, subprocess
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "tables", "comprehensive")
OUT = os.path.join(C, "paper")
PAPER = "/home/ec2-user/P-TA-MulFee/2026_JBHI___DISCO_Tab"
os.makedirs(OUT, exist_ok=True)

# Column grouping by model family (DISCO-TAB = ours, shown first). FULLCOV =
# the models with full dataset coverage; only these are eligible for the
# "best per row" bolding (subset models TAEGAN/GReaT/RTF/TAGAL are not).
FAMILIES = [
    ("Ours", ["disco"]),
    ("GAN", ["ctgan", "adsgan", "pategan", "taegan"]),
    ("VAE", ["tvae", "rtvae"]),
    ("Diff.", ["tabddpm"]),  # tabsyn added here once its runs land
    ("Stat./Flow", ["gaussian_copula", "nflow", "marginal"]),
    ("LLM", ["great", "realtabformer", "tagal"]),
]
MODELS = [m for _, ms in FAMILIES for m in ms]
FULLCOV = ["disco", "ctgan", "adsgan", "pategan", "tvae", "rtvae", "tabddpm",
           "gaussian_copula", "nflow", "marginal"]
MLAB = {"disco": "DISCO-TAB", "ctgan": "CTGAN", "tvae": "TVAE",
        "gaussian_copula": "GaussCop", "tabddpm": "TabDDPM", "adsgan": "ADS-GAN",
        "pategan": "PATE-GAN", "rtvae": "RTVAE", "nflow": "NFlow",
        "marginal": "Marg.", "great": "GReaT", "realtabformer": "RTF",
        "taegan": "TAEGAN", "tagal": "TAGAL", "tabsyn": "TabSyn"}

# dimension (rotated label; \\ gives a line break) -> [(metric, direction, label)]
DIMENSIONS = [
    ("Utility\\\\(TSTR)", [
        ("tstr_f1_macro", "up", "F1-macro $\\uparrow$"),
        ("tstr_f1_weighted", "up", "F1-weight. $\\uparrow$"),
        ("tstr_accuracy", "up", "Accuracy $\\uparrow$"),
    ]),
    ("Fidelity", [
        ("fid_JS", "down", "JS div. $\\downarrow$"),
        ("fid_KS", "down", "KS stat. $\\downarrow$"),
        ("fid_Hellinger", "down", "Helling. $\\downarrow$"),
        ("fid_Pearson", "up", "Corr. $\\uparrow$"),
    ]),
    ("Privacy", [
        ("dcr_median", "up", "DCR $\\uparrow$"),
        ("ext_nndr", "up", "NNDR $\\uparrow$"),
        ("ext_mia_auc", "half", "MIA $\\to$.5"),
        ("ext_dup_rate", "down", "Dup. $\\downarrow$"),
    ]),
    ("Diversity", [
        ("ext_precision", "up", "Precis. $\\uparrow$"),
        ("ext_recall", "up", "Recall $\\uparrow$"),
        ("ext_density", "up", "Density $\\uparrow$"),
        ("ext_coverage", "up", "Coverage $\\uparrow$"),
    ]),
    ("Detection", [
        ("ext_c2st", "down", "C2ST $\\downarrow$"),
        ("ext_mmd", "down", "MMD $\\downarrow$"),
        ("ext_wasserstein", "down", "Wass. $\\downarrow$"),
    ]),
    ("SDMetrics", [
        ("ext_sdm_quality", "up", "Quality $\\uparrow$"),
        ("ext_sdm_shapes", "up", "Shapes $\\uparrow$"),
        ("ext_sdm_pairs", "up", "Pairs $\\uparrow$"),
    ]),
]


def agg(df, metric):
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


def best_among(sbm, direction, pool):
    cand = {m: v[0] for m, v in sbm.items() if m in pool}
    if not cand:
        return None
    if direction == "up":
        return max(cand, key=cand.get)
    if direction == "down":
        return min(cand, key=cand.get)
    return min(cand, key=lambda m: abs(cand[m] - 0.5))


def fmt(mean, ci, bold=False):
    # value and CI on the SAME line; CI in a smaller font, 2 dp w/o leading 0
    # so the 14-column table fits the IEEE two-column width.
    s = f"{mean:.3f}"
    if ci is None or np.isnan(ci):
        cell = s
    else:
        c = (f"{ci:.2f}".lstrip("0")) or ".00"
        cell = f"{s}\\,{{\\tiny$\\pm${c}}}"
    return f"\\textbf{{{cell}}}" if bold else cell


def coverage(df):
    return {m: int(df[df.model == m]["dataset"].nunique()) for m in MODELS}


def table1(df):
    cov = coverage(df)
    N = len(MODELS)
    # family group headers + cmidrules (first data column is 3)
    grp, cmids, col = [], [], 3
    for fam, ms in FAMILIES:
        grp.append(r"\multicolumn{%d}{c}{\textbf{%s}}" % (len(ms), fam))
        cmids.append(r"\cmidrule(lr){%d-%d}" % (col, col + len(ms) - 1))
        col += len(ms)
    L = [
        r"\begin{table*}[t]\centering\scriptsize",
        r"\setlength{\tabcolsep}{1.1pt}\renewcommand{\arraystretch}{1.15}",
        r"\caption{Benchmark summary: mean over all datasets with 95\% CI (same "
        r"cell, smaller font) for each metric and generator, grouped by model "
        r"family. \textbf{Bold} = best per row among the full-coverage models. "
        r"Arrows give the preferred direction. \emph{Ranges:} all scores lie in "
        r"$[0,1]$ except DCR, MMD and Wasserstein, which are unbounded "
        r"($\geq 0$, scale depends on the dataset), and PRDC density, which may "
        r"exceed $1$; MIA AUROC and C2ST are AUROCs for which $0.5$ is ideal "
        r"(synthetic indistinguishable from real). The \#datasets row gives "
        r"per-model coverage; `--' marks a metric not evaluated for that model.}",
        r"\label{tab:aggregate}",
        r"\begin{tabular}{@{}cl *{%d}{c}@{}}" % N,
        r"\toprule",
        r"& & " + " & ".join(grp) + r" \\",
        "".join(cmids),
        r"& \textbf{Metric} & " + " & ".join(r"\textbf{%s}" % MLAB[m] for m in MODELS) + r" \\",
        r"& \#datasets & " + " & ".join(str(cov.get(m, 0)) for m in MODELS) + r" \\",
    ]
    for dim, metrics in DIMENSIONS:
        L.append(r"\midrule")
        n = len(metrics)
        for j, (metric, direction, lab) in enumerate(metrics):
            sbm = agg(df, metric)
            bi = best_among(sbm, direction, FULLCOV)
            cell0 = (r"\multirow{%d}{*}{\rotatebox[origin=c]{90}{\shortstack{%s}}}" % (n, dim)
                     if j == 0 else "")
            row = [cell0, lab]
            for m in MODELS:
                if m in sbm:
                    mean, ci, _ = sbm[m]
                    row.append(fmt(mean, ci, bold=(m == bi)))
                else:
                    row.append("--")
            L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table2(df):
    L = [
        r"\begin{table}[t]\centering\small",
        r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.1}",
        r"\caption{DISCO-TAB vs.\ the best established competing generator per "
        r"metric (mean over all datasets, with 95\% CI). \textbf{Bold} = winner. "
        r"Scores are in $[0,1]$ except DCR/MMD/Wasserstein (unbounded $\geq 0$) "
        r"and PRDC density ($>1$ possible); MIA and C2ST are AUROCs with $0.5$ ideal.}",
        r"\label{tab:vsbest}",
        r"\begin{tabular}{@{}clccl@{}}",
        r"\toprule",
        r"& \textbf{Metric} & \textbf{DISCO-TAB} & \textbf{Best comp.} & \textbf{Method} \\",
    ]
    for dim, metrics in DIMENSIONS:
        L.append(r"\midrule")
        n = len(metrics)
        for j, (metric, direction, lab) in enumerate(metrics):
            sbm = agg(df, metric)
            if "disco" not in sbm:
                continue
            dmean, dci, _ = sbm["disco"]
            pool = [m for m in FULLCOV if m != "disco"]
            bi = best_among(sbm, direction, pool)
            cell0 = (r"\multirow{%d}{*}{\rotatebox[origin=c]{90}{\shortstack{%s}}}" % (n, dim)
                     if j == 0 else "")
            if bi is None:
                L.append(" & ".join([cell0, lab, fmt(dmean, dci), "--", "--"]) + r" \\")
                continue
            bmean, bci, _ = sbm[bi]
            if direction == "up":
                dwin = dmean >= bmean
            elif direction == "down":
                dwin = dmean <= bmean
            else:
                dwin = abs(dmean - 0.5) <= abs(bmean - 0.5)
            name = (r"\textbf{%s}" if not dwin else r"%s") % MLAB[bi]
            L.append(" & ".join([cell0, lab, fmt(dmean, dci, dwin),
                                 fmt(bmean, bci, not dwin), name]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def main():
    df = pd.read_csv(os.path.join(C, "long.csv"))
    df = df[df.dataset != "tab_credit_g"]   # dropped: duplicate of German Credit
    t1, t2 = table1(df), table2(df)
    open(os.path.join(OUT, "paper_table1_aggregate.tex"), "w").write(t1 + "\n")
    open(os.path.join(OUT, "paper_table2_vs_best.tex"), "w").write(t2 + "\n")
    # fit test with the real IEEE class (copy class + generated tex into OUT)
    for f in ("ieeecolor.cls", "generic.sty"):
        src = os.path.join(PAPER, f)
        if os.path.exists(src):
            shutil.copy(src, OUT)
    test = (r"\documentclass[journal,twoside,web]{ieeecolor}" "\n"
            r"\usepackage{graphicx,booktabs,multirow,array,amssymb,amsmath}" "\n"
            r"\begin{document}" "\n"
            r"\input{paper_table1_aggregate}" "\n\\clearpage\n"
            r"\input{paper_table2_vs_best}" "\n"
            r"\end{document}" "\n")
    open(os.path.join(OUT, "table_test.tex"), "w").write(test)
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "table_test.tex"],
                       cwd=OUT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote tables + table_test.pdf; coverage:", coverage(df))


if __name__ == "__main__":
    main()
