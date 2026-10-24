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

# Column grouping by model family (DISCO-TAB = ours, shown first). The set of
# full-coverage models eligible for "best per row" bolding is computed
# dynamically (fullcov_models): the models whose dataset coverage equals the
# maximum over all models (the full 23-dataset benchmark). Partial-coverage
# models (TabSyn, GReaT, RTF, TAGAL) are therefore excluded automatically.
FAMILIES = [
    ("Ours", ["disco"]),
    ("GAN", ["ctgan", "adsgan", "pategan", "taegan"]),
    ("VAE", ["tvae", "rtvae"]),
    ("Diff.", ["tabddpm", "tabsyn"]),
    ("Stat./Flow", ["nflow", "marginal"]),
    ("LLM", ["great", "realtabformer", "tagal"]),
]
MODELS = [m for _, ms in FAMILIES for m in ms]
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


def _key(direction):
    # returns a sort key where the smallest value is the BEST for that metric
    if direction == "up":
        return lambda v: -v
    if direction == "down":
        return lambda v: v
    return lambda v: abs(v - 0.5)   # "half": 0.5 is ideal (MIA AUROC, C2ST)


def rank_models(sbm, direction, pool=None):
    """Models ordered best-first for this metric (across `pool`, or all)."""
    cand = {m: v[0] for m, v in sbm.items()
            if (pool is None or m in pool) and not np.isnan(v[0])}
    return sorted(cand, key=lambda m: _key(direction)(cand[m]))


def best_among(sbm, direction, pool):
    order = rank_models(sbm, direction, pool)
    return order[0] if order else None


def fmt(mean, ci, rank=0):
    # value and CI on the SAME line; CI in a smaller font, 2 dp w/o leading 0
    # so the 14-column table fits the IEEE two-column width.
    # rank: 1 = best (bold+underline), 2 = second (bold), 3 = third (underline).
    s = f"{mean:.3f}"
    if ci is None or np.isnan(ci):
        cell = s
    else:
        c = (f"{ci:.2f}".lstrip("0")) or ".00"
        cell = f"{s}\\,{{\\tiny$\\pm${c}}}"
    if rank == 1:
        return f"\\textbf{{\\underline{{{cell}}}}}"
    if rank == 2:
        return f"\\textbf{{{cell}}}"
    if rank == 3:
        return f"\\underline{{{cell}}}"
    return cell


def coverage(df):
    return {m: int(df[df.model == m]["dataset"].nunique()) for m in MODELS}


def fullcov_models(df):
    """Models with full benchmark coverage (dataset count == the max observed).
    Only these are eligible for the per-row 'best' bolding."""
    cov = coverage(df)
    mx = max(cov.values()) if cov else 0
    return [m for m in MODELS if cov.get(m, 0) >= mx]


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
        r"\caption{\newexp{Benchmark summary: mean over datasets with 95\% CI (smaller "
        r"font) per metric and generator, grouped by family. Per row across all "
        r"generators, \textbf{\underline{bold+underline}}/\textbf{bold}/"
        r"\underline{underline} = best/second/third; means use each generator's covered "
        r"datasets (see the \#datasets row). Arrows give the preferred direction; all "
        r"scores lie in $[0,1]$ except DCR/MMD/Wasserstein (unbounded) and PRDC density "
        r"($>\!1$ possible); MIA and C2ST are AUROCs with $0.5$ ideal.}}",
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
            order = rank_models(sbm, direction)       # global best-first, all models
            rank_of = {m: i + 1 for i, m in enumerate(order[:3])}  # 1,2,3
            cell0 = (r"\multirow{%d}{*}{\rotatebox[origin=c]{90}{\shortstack{%s}}}" % (n, dim)
                     if j == 0 else "")
            row = [cell0, lab]
            for m in MODELS:
                if m in sbm:
                    mean, ci, _ = sbm[m]
                    row.append(fmt(mean, ci, rank=rank_of.get(m, 0)))
                else:
                    row.append("--")
            L.append(" & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def table2(df):
    fc = fullcov_models(df)
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
            pool = [m for m in fc if m != "disco"]
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
            L.append(" & ".join([cell0, lab, fmt(dmean, dci, 2 if dwin else 0),
                                 fmt(bmean, bci, 2 if not dwin else 0), name]) + r" \\")
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
