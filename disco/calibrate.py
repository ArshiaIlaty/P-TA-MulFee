"""
Post-hoc numeric calibration for DISCO-TAB (improvement 1).

Rank/quantile ("copula") calibration: for each numeric column, map the synthetic
values onto the REAL training column's empirical quantile function via their
ranks. Because the mapping is monotonic per column, it leaves the synthetic
ranks -- and therefore the empirical copula (the rank-correlation / dependence
structure DISCO-TAB learned) -- unchanged, while forcing each marginal to match
the real marginal exactly in distribution. This directly improves marginal
fidelity (JS/KS/Hellinger), SDMetrics column-shapes, and Wasserstein/detection,
without retraining and without disturbing the learned joint structure.

Pure post-processing: `calibrate_numerics(df_syn, df_real, num_cols, target)`.
Calibration is done CLASS-CONDITIONALLY (within each target class) so the
class-conditional distributions that drive downstream utility are preserved --
a global (class-agnostic) remap fixes marginals but destroys the feature-label
structure and tanks TSTR. Classes with too few real samples fall back to the
pooled mapping.
"""
import numpy as np
import pandas as pd


def _map_col(real, syn_vals, mask):
    """Rank-map the masked synthetic values onto the real empirical quantiles."""
    ranks = pd.Series(syn_vals[mask]).rank(method="average").values
    q = (ranks - 0.5) / mask.sum()
    cal = np.quantile(real, np.clip(q, 0.0, 1.0))
    if len(real) and np.all(np.mod(real, 1) == 0):
        cal = np.round(cal)
    return np.clip(cal, real.min(), real.max())


def calibrate_numerics(df_syn, df_real, num_cols, target=None, min_per_class=8):
    out = df_syn.copy()
    use_cc = target is not None and target in out.columns and target in df_real.columns
    sy_lab = out[target].astype(str).values if use_cc else None
    re_lab = df_real[target].astype(str).values if use_cc else None
    for c in num_cols:
        if c not in out.columns:
            continue
        real_all = pd.to_numeric(df_real[c], errors="coerce").dropna().values
        if len(real_all) < 5:
            continue
        syn = pd.to_numeric(out[c], errors="coerce").values
        vals = out[c].to_numpy(dtype="float64", copy=True)
        done = np.zeros(len(syn), dtype=bool)
        if use_cc:
            for lab in np.unique(sy_lab):
                real_c = pd.to_numeric(
                    pd.Series(df_real[c].values[re_lab == lab]), errors="coerce").dropna().values
                sel = (sy_lab == lab) & ~np.isnan(syn)
                if sel.sum() < 1:
                    continue
                if len(real_c) >= min_per_class:        # class-conditional map
                    vals[sel] = _map_col(real_c, syn, sel)
                    done |= sel
        rem = ~done & ~np.isnan(syn)                     # pooled fallback
        if rem.sum() >= 1:
            vals[rem] = _map_col(real_all, syn, rem)
        out[c] = vals
    return out


if __name__ == "__main__":
    # Demo/validation: calibrate existing DISCO synthetic CSVs, re-score, and
    # report the fidelity delta vs. the uncalibrated version.
    import os, glob, json, argparse, warnings
    warnings.filterwarnings("ignore")
    HERE = os.path.dirname(os.path.abspath(__file__))
    from data import load_dataset
    import metrics as M

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["heart_failure", "obesity", "diabetes",
                             "surv_whas500", "surv_pbc3", "tab_maternal_risk"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write", action="store_true",
                    help="write calibrated CSVs as tag 'discocal'")
    args = ap.parse_args()

    keys = ["fid_JS", "fid_KS", "fid_Hellinger", "ext_sdm_shapes",
            "ext_wasserstein", "tstr_f1_macro", "tstr_accuracy"]
    print(f"{'dataset':20} {'metric':16} {'base':>8} {'calib':>8} {'delta':>8}")
    for ds in args.datasets:
        p = os.path.join(HERE, "synthetic", f"{ds}__disco__seed{args.seed}.csv")
        if not os.path.exists(p):
            print(f"skip {ds}: no DISCO csv"); continue
        b = load_dataset(ds, seed=args.seed)
        syn = pd.read_csv(p)
        base = M.evaluate_all(b, syn, seed=args.seed)
        cal = calibrate_numerics(syn, b.df_train, b.num_cols, target=b.target)
        calm = M.evaluate_all(b, cal, seed=args.seed)
        for k in keys:
            bv, cv = base.get(k), calm.get(k)
            if bv is None or cv is None:
                continue
            print(f"{ds:20} {k:16} {bv:8.3f} {cv:8.3f} {cv-bv:+8.3f}")
        if args.write:
            cal.to_csv(os.path.join(HERE, "synthetic",
                       f"{ds}__discocal__seed{args.seed}.csv"), index=False)
