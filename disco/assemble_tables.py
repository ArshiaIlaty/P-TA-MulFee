"""
Assemble paper tables/figures from results/*.json (DISCO-TAB + baselines).

Produces, under tables/:
  - table1_tstr_f1.tex        (TSTR F1-macro, datasets x methods, +Delta)
  - table2_fidelity.tex       (DISCO-TAB JS/KL/Hell/KS/Pearson + baseline KS)
  - table3_dcr.tex            (utility vs DCR + real-real baseline column)
  - table_faith.tex           (FAITH decomposition)
  - table_feattype_jsd.tex    (num vs cat JSD per method)
  - headline.json             (key numbers for the manuscript / figures)
Everything is mean over available seeds; SD reported where >1 seed.
"""
import os, json, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES, TAB = os.path.join(HERE, "results"), os.path.join(HERE, "tables")
os.makedirs(TAB, exist_ok=True)

MAIN_METHODS = {"disco", "ctgan", "tvae", "gaussian_copula"}
DS_ORDER = ["heart_failure", "breast_cancer", "liver_disorders", "parkinsons",
            "obesity", "diabetes", "german_credit", "bank_marketing", "census_income"]
DS_LABEL = {d: d.replace("_", " ").title() for d in DS_ORDER}
MED = set(DS_ORDER[:6])
METHOD_LABEL = {"gaussian_copula": "GaussCopula", "ctgan": "CTGAN", "tvae": "TVAE",
                "disco": "DISCO-TAB", "discoV1": "DISCO-TAB"}


def load_long():
    rows = []
    for p in glob.glob(os.path.join(RES, "*.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        rows.append({"dataset": d.get("dataset"), "method": d.get("tag"),
                     "seed": d.get("seed"),
                     **{k: d.get(k, np.nan) for k in (
                         "tstr_f1_macro", "tstr_f1_weighted", "fid_JS", "fid_KL",
                         "fid_Hellinger", "fid_KS", "fid_Pearson", "fid_JS_num",
                         "fid_JS_cat", "dcr_median", "dcr_realreal_median",
                         "faith_FAITH", "faith_Factuality", "faith_Alignment",
                         "faith_Integrity", "faith_Tracking", "valid_rate",
                         "trtr_f1_macro")}})
    return pd.DataFrame(rows)


def mean_pivot(df, metric):
    return df.pivot_table(index="dataset", columns="method", values=metric, aggfunc="mean")


def std_pivot(df, metric):
    return df.pivot_table(index="dataset", columns="method", values=metric, aggfunc="std")


def _methods(df, disco="disco"):
    present = [m for m in ["gaussian_copula", "ctgan", "tvae"] if m in df["method"].unique()]
    dmeth = "disco" if "disco" in df["method"].unique() else (
        "discoV1" if "discoV1" in df["method"].unique() else None)
    return present, dmeth


def table1(df):
    base, dmeth = _methods(df)
    piv = mean_pivot(df, "tstr_f1_macro")
    sd = std_pivot(df, "tstr_f1_macro")
    lines = []
    cols = base + ([dmeth] if dmeth else [])
    header = " & ".join(["Dataset"] + [METHOD_LABEL.get(m, m) for m in base]
                        + (["\\textbf{DISCO-TAB}", "$\\Delta$"] if dmeth else []))
    lines.append(header + " \\\\")
    for ds in [d for d in DS_ORDER if d in piv.index]:
        cells = [DS_LABEL[ds]]
        bvals = []
        for m in base:
            v = piv.loc[ds, m] if m in piv.columns else np.nan
            bvals.append(v)
            cells.append("--" if pd.isna(v) else f"{v:.3f}")
        if dmeth:
            dv = piv.loc[ds, dmeth] if dmeth in piv.columns else np.nan
            dstd = sd.loc[ds, dmeth] if dmeth in sd.columns else np.nan
            cell = "--" if pd.isna(dv) else (f"\\textbf{{{dv:.3f}}}" +
                   (f"$\\pm${dstd:.3f}" if pd.notna(dstd) else ""))
            cells.append(cell)
            valid_b = [x for x in bvals if pd.notna(x)]
            delta = (dv - max(valid_b)) if (valid_b and pd.notna(dv)) else np.nan
            cells.append("--" if pd.isna(delta) else f"{delta:+.3f}")
        lines.append(" & ".join(cells) + " \\\\")
    tex = "\n".join(lines)
    open(os.path.join(TAB, "table1_tstr_f1.tex"), "w").write(tex)
    return piv, sd, tex


def generic_disco_table(df, fname, metrics, better):
    _, dmeth = _methods(df)
    if not dmeth:
        return None
    sub = df[df["method"] == dmeth]
    g = sub.groupby("dataset")
    lines = ["Dataset & " + " & ".join(metrics) + " \\\\"]
    for ds in [d for d in DS_ORDER if d in sub["dataset"].unique()]:
        row = [DS_LABEL[ds]]
        for m in metrics:
            v = g.get_group(ds)[m].mean()
            row.append("--" if pd.isna(v) else f"{v:.3f}")
        lines.append(" & ".join(row) + " \\\\")
    tex = "\n".join(lines)
    open(os.path.join(TAB, fname), "w").write(tex)
    return tex


def main():
    df_all = load_long()
    if df_all.empty:
        print("no results yet"); return
    df_all.to_csv(os.path.join(TAB, "all_results_long.csv"), index=False)
    # Main tables consider only canonical generators; ablation tags handled by
    # assemble_ablation.py.
    df = df_all[df_all["method"].isin(MAIN_METHODS)].copy()
    print("methods:", sorted(df["method"].unique()),
          "\ndatasets:", sorted(df["dataset"].unique()),
          "\nseeds per cell:\n",
          df.groupby(["dataset", "method"]).size().unstack(fill_value=0), "\n")
    piv, sd, _ = table1(df)
    print("=== TSTR F1-macro (mean) ===")
    print(piv.round(3).to_string(), "\n")

    generic_disco_table(df, "table2_fidelity.tex",
                        ["fid_JS", "fid_KL", "fid_Hellinger", "fid_KS", "fid_Pearson"],
                        better="low")
    generic_disco_table(df, "table3_dcr.tex",
                        ["tstr_f1_macro", "dcr_median", "dcr_realreal_median"],
                        better="mix")
    generic_disco_table(df, "table_faith.tex",
                        ["faith_FAITH", "faith_Factuality", "faith_Alignment",
                         "faith_Integrity", "faith_Tracking"], better="high")

    # feature-type JSD per method (num vs cat)
    ft = df.groupby("method")[["fid_JS_num", "fid_JS_cat"]].mean()
    ft.to_csv(os.path.join(TAB, "feattype_jsd.csv"))
    print("=== feature-type JSD (mean per method) ===")
    print(ft.round(3).to_string(), "\n")

    # headline numbers
    base, dmeth = _methods(df)
    headline = {}
    if dmeth:
        for ds in piv.index:
            cols = [c for c in base if c in piv.columns]
            bvals = piv.loc[ds, cols].dropna()
            if len(bvals) and pd.notna(piv.loc[ds, dmeth]):
                headline[ds] = {"disco": float(piv.loc[ds, dmeth]),
                                "best_base": float(bvals.max()),
                                "delta": float(piv.loc[ds, dmeth] - bvals.max())}
    json.dump(headline, open(os.path.join(TAB, "headline.json"), "w"), indent=2)
    print("=== headline Delta (disco - best baseline) ===")
    for ds, h in headline.items():
        print(f"  {ds:16s} disco={h['disco']:.3f} best_base={h['best_base']:.3f} "
              f"Delta={h['delta']:+.3f}")


if __name__ == "__main__":
    main()
