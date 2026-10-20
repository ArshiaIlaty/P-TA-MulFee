"""
Render every metric as a LaTeX datasets x models table showing MEAN +/- 95% CI
(t-based over seeds), dash for missing, bold = best per row (by mean, where a
direction is defined). Assemble into one landscape PDF. Reads
tables/comprehensive/long.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tables", "comprehensive")
LONG = os.path.join(OUT, "long.csv")

MODELS = ["disco", "ctgan", "tvae", "gaussian_copula", "tabddpm", "adsgan",
          "pategan", "rtvae", "nflow", "arf", "bayesian_network", "marginal",
          "great", "realtabformer"]
MLAB = {"disco": "DISCO-TAB", "ctgan": "CTGAN", "tvae": "TVAE",
        "gaussian_copula": "GaussCop", "tabddpm": "TabDDPM", "adsgan": "ADS-GAN",
        "pategan": "PATE-GAN", "rtvae": "RTVAE", "nflow": "NFlow", "arf": "ARF",
        "bayesian_network": "BayesNet", "marginal": "Marginals",
        "great": "GReaT", "realtabformer": "RealTabF"}
DS_ORDER = ["heart_failure", "breast_cancer", "liver_disorders", "parkinsons",
            "obesity", "diabetes", "german_credit", "bank_marketing", "census_income",
            "surv_melanoma", "surv_pbc", "surv_pbc3", "surv_retinopathy",
            "surv_whas500", "surv_gbsg2", "surv_colon", "surv_grace", "surv_trace",
            "surv_framingham", "tab_blood_transfusion", "tab_credit_g",
            "tab_qsar_biodeg", "tab_maternal_risk", "tab_website_phishing"]
METRICS = [
    ("tstr_f1_macro", True, "Utility (downstream TSTR)", "TSTR F1-macro ($\\uparrow$)"),
    ("tstr_f1_weighted", True, "Utility (downstream TSTR)", "TSTR F1-weighted ($\\uparrow$)"),
    ("tstr_accuracy", True, "Utility (downstream TSTR)", "TSTR accuracy ($\\uparrow$)"),
    ("fid_JS", False, "Statistical fidelity", "Jensen--Shannon div. ($\\downarrow$)"),
    ("fid_KS", False, "Statistical fidelity", "KS statistic ($\\downarrow$)"),
    ("fid_KL", False, "Statistical fidelity", "KL divergence ($\\downarrow$)"),
    ("fid_Hellinger", False, "Statistical fidelity", "Hellinger ($\\downarrow$)"),
    ("fid_Pearson", True, "Statistical fidelity", "Pearson corr.\\ preservation ($\\uparrow$)"),
    ("fid_JS_num", False, "Statistical fidelity", "JS numeric ($\\downarrow$)"),
    ("fid_JS_cat", False, "Statistical fidelity", "JS categorical ($\\downarrow$)"),
    ("dcr_median", True, "Privacy / disclosure", "DCR median ($\\uparrow$=less memorization)"),
    ("dcr_realreal_median", None, "Privacy / disclosure", "DCR real--real reference"),
    ("ext_dcr", True, "Privacy / disclosure", "DCR encoded ($\\uparrow$)"),
    ("ext_nndr", True, "Privacy / disclosure", "NNDR ($\\uparrow$)"),
    ("ext_nnaa", None, "Privacy / disclosure", "NNAA (0.5 ideal)"),
    ("ext_mia_auc", None, "Privacy / disclosure", "MIA AUROC (0.5 safe)"),
    ("ext_dup_rate", False, "Privacy / disclosure", "Duplicate rate ($\\downarrow$)"),
    ("ext_precision", True, "Diversity / coverage (PRDC)", "Precision ($\\uparrow$)"),
    ("ext_recall", True, "Diversity / coverage (PRDC)", "Recall ($\\uparrow$)"),
    ("ext_density", True, "Diversity / coverage (PRDC)", "Density ($\\uparrow$)"),
    ("ext_coverage", True, "Diversity / coverage (PRDC)", "Coverage ($\\uparrow$)"),
    ("ext_c2st", False, "Detection / distribution", "C2ST AUROC ($\\downarrow$, 0.5 best)"),
    ("ext_mmd", False, "Detection / distribution", "MMD$^2$ ($\\downarrow$)"),
    ("ext_fpcad", False, "Detection / distribution", "Fr\\'echet-PCA ($\\downarrow$)"),
    ("ext_wasserstein", False, "Detection / distribution", "Wasserstein ($\\downarrow$)"),
    ("ext_assoc_err", False, "Detection / distribution", "Association error ($\\downarrow$)"),
    ("ext_sdm_quality", True, "SDMetrics quality report", "Overall quality ($\\uparrow$)"),
    ("ext_sdm_shapes", True, "SDMetrics quality report", "Column shapes ($\\uparrow$)"),
    ("ext_sdm_pairs", True, "SDMetrics quality report", "Column-pair trends ($\\uparrow$)"),
    ("faith_FAITH", True, "FAITH framework", "FAITH overall ($\\uparrow$)"),
    ("faith_Factuality", True, "FAITH framework", "Factuality ($\\uparrow$)"),
    ("faith_Alignment", True, "FAITH framework", "Alignment ($\\uparrow$)"),
    ("faith_Integrity", True, "FAITH framework", "Integrity ($\\uparrow$)"),
    ("faith_Tracking", True, "FAITH framework", "Tracking ($\\uparrow$)"),
    ("valid_rate", True, "Other", "Parse valid-rate ($\\uparrow$)"),
]


def esc(s):
    return str(s).replace("_", "\\_")


def mean_ci(v):
    v = pd.Series(v).dropna().values
    n = len(v)
    if n == 0:
        return np.nan, np.nan, 0
    m = float(np.mean(v))
    if n < 2:
        return m, np.nan, n
    sd = float(np.std(v, ddof=1))
    ci = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return m, ci, n


def table_tex(df, metric, hib, label):
    g = df.groupby(["dataset", "model"])[metric]
    stat = g.apply(lambda s: mean_ci(s.values))
    meanP = stat.apply(lambda t: t[0]).unstack()
    ciP = stat.apply(lambda t: t[1]).unstack()
    cols = [m for m in MODELS if m in meanP.columns]
    rows = [d for d in DS_ORDER if d in meanP.index]
    if not cols or not rows:
        return None
    hdr = " & ".join(["\\textbf{Dataset}"] + [f"\\textbf{{{esc(MLAB[m])}}}" for m in cols])
    out = [f"\\begin{{longtable}}{{l*{{{len(cols)}}}{{c}}}}",
           f"\\caption{{{label} --- mean $\\pm$ 95\\% CI over seeds}}\\\\",
           "\\toprule", hdr + " \\\\", "\\midrule",
           "\\endfirsthead", "\\toprule", hdr + " \\\\", "\\midrule", "\\endhead",
           "\\bottomrule", "\\endfoot"]
    for ds in rows:
        mv = meanP.loc[ds, cols]
        valid = mv.dropna()
        best = (valid.max() if hib else valid.min()) if (hib is not None and len(valid)) else None
        cells = [esc(ds)]
        for m in cols:
            v = mv[m]
            if pd.isna(v):
                cells.append("--")
                continue
            ci = ciP.loc[ds, m]
            s = f"{v:.3f}" if pd.isna(ci) else f"{v:.3f}\\,$\\pm$\\,{ci:.3f}"
            if best is not None and abs(v - best) < 1e-9:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        out.append(" & ".join(cells) + " \\\\")
    out.append("\\end{longtable}")
    return "\n".join(out)


def main():
    df = pd.read_csv(LONG)
    body, cur = [], None
    for metric, hib, group, label in METRICS:
        if metric not in df.columns:
            continue
        t = table_tex(df, metric, hib, label)
        if t is None:
            continue
        if group != cur:
            body.append(f"\n\\section*{{{group}}}")
            cur = group
        body.append(t)
    doc = r"""\documentclass[8pt]{extarticle}
\usepackage[landscape,margin=0.7cm]{geometry}
\usepackage{longtable,booktabs,array}
\usepackage{amsmath,amssymb}
\usepackage[T1]{fontenc}
\renewcommand{\arraystretch}{1.08}
\setlength{\tabcolsep}{2.2pt}
\begin{document}
\begin{center}{\Large\textbf{DISCO-TAB --- Comprehensive Benchmark}}\\[2pt]
all generators $\times$ 24 datasets $\times$ all metrics; cells are
\textbf{mean $\pm$ 95\% confidence interval} over random seeds (Student-$t$,
up to 3 seeds); \textbf{bold} = best per row (by mean, where a direction is
defined); -- = missing run/metric; $\pm$ absent where only one seed is available.
\end{center}
\scriptsize
""" + "\n".join(body) + "\n\\end{document}\n"
    tex = os.path.join(OUT, "benchmark_tables.tex")
    open(tex, "w").write(doc)
    print("wrote", tex)


if __name__ == "__main__":
    main()
