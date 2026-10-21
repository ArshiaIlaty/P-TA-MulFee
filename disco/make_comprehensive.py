"""
Comprehensive benchmark tables: all models x all datasets x all metrics.
Reads results/*.json -> writes tables/comprehensive/:
  - long.csv                      (every run x every metric)
  - <metric>__datasets_x_models.csv  (mean over seeds, one per metric)
  - leaderboard.csv               (per-model aggregate: mean, avg-rank, #wins per metric)
  - REPORT.md                     (rendered headline tables + leaderboards)
"""
import os, json, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tables", "comprehensive")
os.makedirs(OUT, exist_ok=True)

# ARF and the Bayesian network dropped from the analysis (authors' choice for the
# revision). FAITH dropped too (not part of the original submission).
GEN_MODELS = ["disco", "ctgan", "tvae", "gaussian_copula", "tabddpm",
              "adsgan", "pategan", "rtvae", "nflow", "marginal",
              "great", "realtabformer", "taegan", "tagal", "tabsyn"]
MLAB = {"disco": "DISCO-TAB", "ctgan": "CTGAN", "tvae": "TVAE",
        "gaussian_copula": "GaussCop", "tabddpm": "TabDDPM", "adsgan": "ADS-GAN",
        "pategan": "PATE-GAN", "rtvae": "RTVAE", "nflow": "NFlow",
        "marginal": "Marginals",
        "great": "GReaT", "realtabformer": "RealTabF", "tabsyn": "TabSyn",
        "taegan": "TAEGAN", "tagal": "TAGAL"}
DS_ORDER = ["heart_failure", "breast_cancer", "liver_disorders", "parkinsons",
            "obesity", "diabetes", "german_credit", "bank_marketing", "census_income",
            "surv_melanoma", "surv_pbc", "surv_pbc3", "surv_retinopathy",
            "surv_whas500", "surv_gbsg2", "surv_colon", "surv_grace", "surv_trace",
            "surv_framingham", "tab_blood_transfusion",
            "tab_qsar_biodeg", "tab_maternal_risk", "tab_website_phishing"]

# metric -> (higher_is_better, group, label)
METRICS = {
    "tstr_f1_macro": (True, "Utility", "TSTR F1-macro"),
    "tstr_f1_weighted": (True, "Utility", "TSTR F1-weighted"),
    "tstr_accuracy": (True, "Utility", "TSTR accuracy"),
    "fid_JS": (False, "Fidelity", "JS divergence"),
    "fid_KS": (False, "Fidelity", "KS statistic"),
    "fid_KL": (False, "Fidelity", "KL divergence"),
    "fid_Hellinger": (False, "Fidelity", "Hellinger"),
    "fid_Pearson": (True, "Fidelity", "Pearson corr-preserve"),
    "fid_JS_num": (False, "Fidelity", "JS numeric"),
    "fid_JS_cat": (False, "Fidelity", "JS categorical"),
    "dcr_median": (True, "Privacy", "DCR median"),
    "dcr_realreal_median": (None, "Privacy", "DCR real-real ref"),
    "ext_dcr": (True, "Privacy", "DCR (ext)"),
    "ext_nndr": (True, "Privacy", "NNDR"),
    "ext_nnaa": (None, "Privacy", "NNAA (0.5 ideal)"),
    "ext_mia_auc": (None, "Privacy", "MIA AUROC (0.5 safe)"),
    "ext_dup_rate": (False, "Privacy", "Duplicate rate"),
    "ext_precision": (True, "Diversity", "PRDC precision"),
    "ext_recall": (True, "Diversity", "PRDC recall"),
    "ext_density": (True, "Diversity", "PRDC density"),
    "ext_coverage": (True, "Diversity", "PRDC coverage"),
    "ext_c2st": (False, "Detection", "C2ST AUROC"),
    "ext_mmd": (False, "Detection", "MMD"),
    "ext_fpcad": (False, "Detection", "Frechet-PCA dist"),
    "ext_wasserstein": (False, "Detection", "Wasserstein"),
    "ext_assoc_err": (False, "Detection", "Association error"),
    "ext_sdm_quality": (True, "SDMetrics", "SDM quality"),
    "ext_sdm_shapes": (True, "SDMetrics", "SDM column shapes"),
    "ext_sdm_pairs": (True, "SDMetrics", "SDM pair trends"),
    "valid_rate": (True, "Other", "Parse valid rate"),
}


def load_long():
    rows = []
    for p in glob.glob(os.path.join(HERE, "results", "*.json")):
        d = json.load(open(p))
        if d.get("tag") not in GEN_MODELS or d.get("n_syn", 0) < 10:
            continue
        rec = {"dataset": d["dataset"], "model": d["tag"], "seed": d.get("seed")}
        for m in METRICS:
            rec[m] = d.get(m, np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    df = load_long()
    df.to_csv(os.path.join(OUT, "long.csv"), index=False)
    models = [m for m in GEN_MODELS if m in set(df.model)]
    dss = [d for d in DS_ORDER if d in set(df.dataset)]

    lb_rows = []
    md = ["# DISCO-TAB — Comprehensive Benchmark (all models × datasets × metrics)\n",
          f"_{len(df)} runs | {len(models)} models | {len(dss)} datasets | "
          f"mean over up to 3 seeds. Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}._\n",
          "Models: " + ", ".join(MLAB[m] for m in models) + "\n"]

    for metric, (hib, grp, lab) in METRICS.items():
        sub = df[["dataset", "model", metric]].dropna(subset=[metric])
        if sub.empty:
            continue
        piv = sub.pivot_table(index="dataset", columns="model", values=metric, aggfunc="mean")
        piv = piv.reindex(index=[d for d in dss if d in piv.index],
                          columns=[m for m in models if m in piv.columns])
        piv.round(4).to_csv(os.path.join(OUT, f"{metric}__datasets_x_models.csv"))
        # 95% CI (Student-t) half-width per cell, over seeds
        from scipy import stats as _st

        def _ci(s):
            v = s.dropna().values
            if len(v) < 2:
                return np.nan
            return float(_st.t.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v)))
        ciP = sub.pivot_table(index="dataset", columns="model", values=metric, aggfunc=_ci)
        ciP = ciP.reindex(index=piv.index, columns=piv.columns)
        ciP.round(4).to_csv(os.path.join(OUT, f"{metric}__ci95.csv"))
        # human-readable "mean +/- ci" combined CSV
        comb = piv.copy().astype(object)
        for d in piv.index:
            for m in piv.columns:
                mv = piv.loc[d, m]; cv = ciP.loc[d, m]
                comb.loc[d, m] = ("" if pd.isna(mv) else
                                  (f"{mv:.3f}" if pd.isna(cv) else f"{mv:.3f}+/-{cv:.3f}"))
        comb.to_csv(os.path.join(OUT, f"{metric}__mean_ci.csv"))
        cov = int(sub.dataset.nunique())
        # leaderboard aggregate
        for m in piv.columns:
            col = piv[m].dropna()
            row = {"metric": metric, "group": grp, "label": lab, "model": m,
                   "mean": round(col.mean(), 4), "n_datasets": int(col.shape[0])}
            if hib is not None and piv.shape[1] > 1:
                ranks = piv.rank(axis=1, ascending=not hib)
                row["avg_rank"] = round(ranks[m].mean(), 2)
                best = piv.idxmax(axis=1) if hib else piv.idxmin(axis=1)
                row["n_best"] = int((best == m).sum())
            lb_rows.append(row)

    lb = pd.DataFrame(lb_rows)
    lb.to_csv(os.path.join(OUT, "leaderboard.csv"), index=False)

    # ---- render headline per-dataset table (TSTR F1-macro) ----
    def render(metric):
        piv = df.pivot_table(index="dataset", columns="model", values=metric, aggfunc="mean")
        piv = piv.reindex(index=[d for d in dss if d in piv.index],
                          columns=[m for m in models if m in piv.columns])
        head = "| Dataset | " + " | ".join(MLAB[m] for m in piv.columns) + " |"
        sep = "|" + "---|" * (len(piv.columns) + 1)
        lines = [head, sep]
        hib = METRICS[metric][0]
        for ds in piv.index:
            vals = piv.loc[ds]
            best = vals.max() if hib else vals.min()
            cells = []
            for m in piv.columns:
                v = vals[m]
                s = "--" if pd.isna(v) else f"{v:.3f}"
                if not pd.isna(v) and abs(v - best) < 1e-9:
                    s = f"**{s}**"
                cells.append(s)
            lines.append(f"| {ds} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    md.append("\n## Table 1 — TSTR F1-macro (utility; higher better; bold = best)\n")
    md.append(render("tstr_f1_macro"))
    md.append("\n## Table 2 — JS divergence (fidelity; lower better; bold = best)\n")
    md.append(render("fid_JS"))
    md.append("\n## Table 3 — DCR median (privacy proximity; bold = highest)\n")
    md.append(render("dcr_median"))
    md.append("\n## Table 4 — SDMetrics overall quality (bold = best)\n")
    md.append(render("ext_sdm_quality"))

    # ---- per-model summary leaderboard (mean + avg-rank over datasets) ----
    md.append("\n## Per-model summary (mean over datasets; avg-rank 1=best; #best = datasets ranked #1)\n")
    key = ["tstr_f1_macro", "fid_JS", "dcr_median", "ext_c2st", "ext_coverage",
           "ext_sdm_quality"]
    md.append("| Model | " + " | ".join(METRICS[k][2] for k in key) + " |")
    md.append("|" + "---|" * (len(key) + 1))
    for m in models:
        cells = [MLAB[m]]
        for k in key:
            r = lb[(lb.metric == k) & (lb.model == m)]
            if len(r):
                mv = r["mean"].values[0]; ar = r.get("avg_rank")
                arv = r["avg_rank"].values[0] if "avg_rank" in r and not pd.isna(r["avg_rank"].values[0]) else None
                cells.append(f"{mv:.3f} (r{arv})" if arv is not None else f"{mv:.3f}")
            else:
                cells.append("--")
        md.append("| " + " | ".join(cells) + " |")

    md.append("\n## Metric coverage note\n")
    cov = df[[c for c in METRICS]].notna().sum()
    md.append("Per-metric #runs scored (ext_* fill in as the enrichment pass completes):\n")
    md.append("```\n" + cov.to_string() + "\n```")

    open(os.path.join(OUT, "REPORT.md"), "w").write("\n".join(md))
    print(f"Wrote {OUT}/ : long.csv, {len(METRICS)} per-metric CSVs, leaderboard.csv, REPORT.md")
    print(f"models={len(models)} datasets={len(dss)} runs={len(df)}")


if __name__ == "__main__":
    main()
