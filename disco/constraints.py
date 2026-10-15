"""
DISCO-TAB: Automated Constraint Discovery (Section IV-B).

Builds a global association matrix C over feature pairs using
  - Pearson |r|      for numeric-numeric,
  - Cramer's V       for categorical-categorical,
  - correlation ratio for mixed numeric/categorical,
then returns P_crit = Top-k { (f_a,f_b) : |C_ab| >= delta_thresh }.
"""
import numpy as np
import pandas as pd
from itertools import combinations


def cramers_v(a, b):
    a = a.astype(str)
    b = b.astype(str)
    ct = pd.crosstab(a, b)
    if ct.size == 0 or ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2 = _chi2(ct.values)
    n = ct.values.sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = ct.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def _chi2(obs):
    obs = obs.astype(float)
    row = obs.sum(1, keepdims=True)
    col = obs.sum(0, keepdims=True)
    tot = obs.sum()
    exp = row @ col / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(exp > 0, (obs - exp) ** 2 / exp, 0.0)
    return float(term.sum())


def correlation_ratio(categories, values):
    """eta: association between a categorical and a numeric variable, in [0,1]."""
    cats = pd.Series(categories).astype(str).values
    vals = pd.to_numeric(pd.Series(values), errors="coerce").values
    ok = ~np.isnan(vals)
    cats, vals = cats[ok], vals[ok]
    if len(vals) < 2:
        return 0.0
    grand = vals.mean()
    ss_between, ss_total = 0.0, ((vals - grand) ** 2).sum()
    for c in np.unique(cats):
        g = vals[cats == c]
        ss_between += len(g) * (g.mean() - grand) ** 2
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


def pearson_abs(a, b):
    a = pd.to_numeric(pd.Series(a), errors="coerce")
    b = pd.to_numeric(pd.Series(b), errors="coerce")
    ok = a.notna() & b.notna()
    if ok.sum() < 2:
        return 0.0
    sa, sb = a[ok], b[ok]
    if sa.std() < 1e-12 or sb.std() < 1e-12:
        return 0.0
    return float(abs(np.corrcoef(sa, sb)[0, 1]))


def association(df, fa, fb, num_cols):
    a_num, b_num = fa in num_cols, fb in num_cols
    if a_num and b_num:
        return pearson_abs(df[fa], df[fb])
    if (not a_num) and (not b_num):
        return cramers_v(df[fa], df[fb])
    if a_num:
        return correlation_ratio(df[fb], df[fa])
    return correlation_ratio(df[fa], df[fb])


def discover_constraints(df, num_cols, cat_cols, delta_thresh=0.3, top_k=8):
    feats = list(num_cols) + list(cat_cols)
    scored = []
    for fa, fb in combinations(feats, 2):
        s = association(df, fa, fb, set(num_cols))
        scored.append((fa, fb, s))
    scored.sort(key=lambda t: t[2], reverse=True)
    pcrit = [(fa, fb, round(s, 4)) for fa, fb, s in scored if s >= delta_thresh][:top_k]
    # Guarantee at least one pair so D_feat is always defined.
    if not pcrit and scored:
        fa, fb, s = scored[0]
        pcrit = [(fa, fb, round(s, 4))]
    return pcrit, scored


if __name__ == "__main__":
    from data import load_dataset
    for name in ["heart_failure", "obesity", "diabetes"]:
        b = load_dataset(name, seed=0)
        pcrit, scored = discover_constraints(b.df_train, b.num_cols, b.cat_cols,
                                             delta_thresh=0.3, top_k=8)
        print(f"\n=== {name} === P_crit ({len(pcrit)} pairs, thresh 0.3):")
        for fa, fb, s in pcrit:
            print(f"   {s:.3f}  {fa} <-> {fb}")
        if not pcrit:
            print("   (none above threshold; strongest:", scored[0] if scored else None, ")")
