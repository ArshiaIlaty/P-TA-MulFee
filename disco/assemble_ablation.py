"""
Assemble the ablation table (RED #6) from results/*.json for the ablation tags.

Relative performance = mean_dataset( F1_tag / F1_full ), where full = 'disco',
averaged over the ablation datasets (heart_failure, obesity) and seeds. Reports
mean +/- SD of the per-(dataset,seed) ratio and the mean runtime.
"""
import os, glob, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES, TAB = os.path.join(HERE, "results"), os.path.join(HERE, "tables")
FULL = "disco"
TAGS = ["disco", "no_token", "no_sent", "no_feat", "no_row", "no_ifrs",
        "row_only", "no_rl"]
LABEL = {"disco": "Full DISCO-TAB", "no_token": "w/o Token Disc",
         "no_sent": "w/o Sentence Disc", "no_feat": "w/o Feature Disc",
         "no_row": "w/o Row Disc", "no_ifrs": "w/o IFRS",
         "row_only": "Row-only (no hierarchy)", "no_rl": "No RL (SFT only)"}
DATASETS = ["heart_failure", "obesity"]


def load():
    rows = []
    for p in glob.glob(os.path.join(RES, "*.json")):
        d = json.load(open(p))
        if d.get("tag") in TAGS and d.get("dataset") in DATASETS:
            rows.append({"dataset": d["dataset"], "tag": d["tag"],
                         "seed": d.get("seed"),
                         "f1": d.get("tstr_f1_macro", np.nan),
                         "runtime": d.get("runtime_s", np.nan)})
    return pd.DataFrame(rows)


def main():
    df = load()
    if df.empty:
        print("no ablation results yet"); return
    full = (df[df.tag == FULL].groupby("dataset")["f1"].mean())
    recs = []
    for tag in TAGS:
        sub = df[df.tag == tag]
        if sub.empty:
            continue
        ratios = []
        for _, r in sub.iterrows():
            f = full.get(r["dataset"], np.nan)
            if f and np.isfinite(f) and f > 0:
                ratios.append(r["f1"] / f)
        if not ratios:
            continue
        recs.append({"config": LABEL[tag], "tag": tag,
                     "rel_perf_%": round(100 * np.mean(ratios), 1),
                     "rel_sd_%": round(100 * np.std(ratios), 1) if len(ratios) > 1 else 0.0,
                     "mean_f1": round(sub["f1"].mean(), 3),
                     "runtime_min": round(sub["runtime"].mean() / 60, 1),
                     "n": len(sub)})
    out = pd.DataFrame(recs)
    out.to_csv(os.path.join(TAB, "ablation.csv"), index=False)
    print(out.to_string(index=False))
    # LaTeX body
    lines = []
    for _, r in out.iterrows():
        lines.append(f"{r['config']} & {r['rel_perf_%']:.1f}\\% $\\pm$ {r['rel_sd_%']:.1f} "
                     f"& {r['runtime_min']:.1f} \\\\")
    open(os.path.join(TAB, "table_ablation.tex"), "w").write("\n".join(lines))
    print("\nwrote tables/table_ablation.tex")


if __name__ == "__main__":
    main()
