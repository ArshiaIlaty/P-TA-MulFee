"""
Enriched evaluation battery for the DISCO-TAB benchmark.

Ports the general-purpose (non-survival) metrics from the survivor project's
`privacy_fidelity_eval.py` (c2st, mmd, wasserstein, KL/TVD/Hellinger, Frechet-PCA,
NNAA, DCR/NNDR, black-box MIA, duplicate-rate, type-aware association-error),
and adds SDMetrics QualityReport (column shapes / pair trends) and PRDC
precision/recall/density/coverage. All encoding reuses the data.py Preprocessor
(`b.pp.transform_X`) so distances are consistent with the rest of the pipeline.

`evaluate_ext(b, df_syn)` -> flat dict of ext_* metrics (NaN on any failure).
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance, chi2_contingency
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

RNG = np.random.RandomState(12345)
CAP = 2000  # subsample cap for O(n^2) distance metrics


def _sub(X, cap=CAP):
    if len(X) <= cap:
        return X
    return X[RNG.choice(len(X), cap, replace=False)]


def _nn_dists(A, B, k=1):
    out = np.empty((len(A), k), dtype=np.float32)
    step = 512
    for s in range(0, len(A), step):
        d = cdist(A[s:s + step], B)
        d.sort(axis=1)
        out[s:s + step] = d[:, :k]
    return out


# ----------------------- fidelity / detection -----------------------
def c2st(Rtr_e, S_e):
    n = min(len(Rtr_e), len(S_e))
    if n < 20:
        return np.nan
    X = np.vstack([Rtr_e[RNG.choice(len(Rtr_e), n, replace=False)],
                   S_e[RNG.choice(len(S_e), n, replace=False)]])
    y = np.r_[np.ones(n), np.zeros(n)]
    clf = GradientBoostingClassifier(random_state=0)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def mmd_rbf(Rtr_e, S_e, cap=1200):
    X = _sub(Rtr_e, cap); Y = _sub(S_e, cap)
    if len(X) < 2 or len(Y) < 2:
        return np.nan
    Z = np.vstack([X, Y])
    d2 = cdist(Z, Z, "sqeuclidean")
    med = np.median(d2[d2 > 0]) or 1.0
    g = 1.0 / med
    Kxx = np.exp(-g * cdist(X, X, "sqeuclidean"))
    Kyy = np.exp(-g * cdist(Y, Y, "sqeuclidean"))
    Kxy = np.exp(-g * cdist(X, Y, "sqeuclidean"))
    m, n = len(X), len(Y)
    np.fill_diagonal(Kxx, 0); np.fill_diagonal(Kyy, 0)
    mmd2 = Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2 * Kxy.mean()
    return float(max(mmd2, 0.0))


def fpcad(Rtr_e, S_e, dim=10):
    from scipy.linalg import sqrtm
    d = min(dim, S_e.shape[1], len(Rtr_e) - 1, len(S_e) - 1)
    if d < 2:
        return np.nan
    pca = PCA(n_components=d, random_state=0).fit(np.vstack([Rtr_e, S_e]))
    Xr, Xs = pca.transform(Rtr_e), pca.transform(S_e)
    mu_r, mu_s = Xr.mean(0), Xs.mean(0)
    Cr, Cs = np.cov(Xr, rowvar=False), np.cov(Xs, rowvar=False)
    diff = mu_r - mu_s
    covmean = sqrtm(Cr @ Cs)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(max(diff @ diff + np.trace(Cr + Cs - 2 * covmean), 0.0))


def wasserstein_feats(real_df, syn_df, num_cols):
    ds = []
    for c in num_cols:
        r = pd.to_numeric(real_df[c], errors="coerce").dropna().values
        s = pd.to_numeric(syn_df[c], errors="coerce").dropna().values
        if len(r) and len(s):
            sd = r.std() or 1.0
            ds.append(wasserstein_distance(r / sd, s / sd))
    return float(np.mean(ds)) if ds else np.nan


def _hist_pair(r, s, is_num, bins=20):
    if is_num:
        r = pd.to_numeric(r, errors="coerce").dropna().values
        s = pd.to_numeric(s, errors="coerce").dropna().values
        if len(r) < 2:
            return None
        edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return None
        pr, _ = np.histogram(r, bins=edges)
        ps, _ = np.histogram(np.clip(s, edges[0], edges[-1]), bins=edges)
    else:
        cats = pd.Index(pd.Series(r).astype(str).unique())
        pr = pd.Series(r).astype(str).value_counts().reindex(cats, fill_value=0).values
        ps = pd.Series(s).astype(str).value_counts().reindex(cats, fill_value=0).values
    pr = pr / max(pr.sum(), 1); ps = ps / max(ps.sum(), 1)
    eps = 1e-9
    return pr + eps, ps + eps


def divergences(real_df, syn_df, feat_cols, cat_cols):
    kl, tv, he = [], [], []
    cat_set = set(cat_cols)
    for c in feat_cols:
        is_num = c not in cat_set and pd.api.types.is_numeric_dtype(real_df[c])
        pair = _hist_pair(real_df[c], syn_df[c], is_num)
        if pair is None:
            continue
        p, q = pair
        p /= p.sum(); q /= q.sum()
        kl.append(0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))))
        tv.append(0.5 * np.sum(np.abs(p - q)))
        he.append(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))
    f = lambda a: float(np.mean(a)) if a else np.nan
    return f(kl), f(tv), f(he)


# ----------------------- privacy / disclosure -----------------------
def dcr_nndr(Rtr, Rho, S):
    d_tr = _nn_dists(S, Rtr, k=2)
    d_ho = _nn_dists(S, Rho, k=1)[:, 0]
    dcr = float(np.median(d_tr[:, 0]))
    ratio = d_tr[:, 0] / np.clip(d_tr[:, 1], 1e-9, None)
    nndr = float(np.median(ratio))
    closer = float(np.mean(d_tr[:, 0] < d_ho))
    return dcr, nndr, closer


def nnaa(Rtr, S):
    n = min(len(Rtr), len(S))
    if n < 10:
        return np.nan
    R = Rtr[RNG.choice(len(Rtr), n, replace=False)]
    Sx = S[RNG.choice(len(S), n, replace=False)]
    d_ts = _nn_dists(R, Sx, k=1)[:, 0]
    d_tt = _nn_dists(R, R, k=2)[:, 1]
    d_st = _nn_dists(Sx, R, k=1)[:, 0]
    d_ss = _nn_dists(Sx, Sx, k=2)[:, 1]
    return float(0.5 * (np.mean(d_ts > d_tt) + np.mean(d_st > d_ss)))


def mia_blackbox(Rtr_e, Rho_e, S_e):
    dm = _nn_dists(Rtr_e, S_e, k=1)[:, 0]
    dn = _nn_dists(Rho_e, S_e, k=1)[:, 0]
    y = np.r_[np.ones(len(dm)), np.zeros(len(dn))]
    score = -np.r_[dm, dn]
    return float(roc_auc_score(y, score))


def duplicate_rate(real_df, syn_df, feat_cols):
    r = real_df[feat_cols].round(6).astype(str).agg("|".join, axis=1)
    s = syn_df[feat_cols].round(6).astype(str).agg("|".join, axis=1)
    return float(s.isin(set(r)).mean())


# ----------------------- association error -----------------------
def _cramers_v(x, y):
    x = pd.Series(x).astype(str); y = pd.Series(y).astype(str)
    tab = pd.crosstab(x, y).values
    if tab.shape[0] < 2 or tab.shape[1] < 2 or tab.sum() == 0:
        return 0.0
    try:
        chi2 = chi2_contingency(tab, correction=False)[0]
    except Exception:
        return 0.0
    n = tab.sum(); r, k = tab.shape
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1); kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def _corr_ratio(cats, values):
    cats = pd.Series(cats).astype(str).values
    values = pd.to_numeric(pd.Series(values), errors="coerce").values
    ok = ~np.isnan(values); cats, values = cats[ok], values[ok]
    if len(values) < 2:
        return 0.0
    grand = values.mean()
    ssb = sum(len(v) * (v.mean() - grand) ** 2
              for v in (values[cats == lv] for lv in np.unique(cats)) if len(v))
    sst = float(((values - grand) ** 2).sum())
    return float(np.sqrt(max(0.0, ssb / sst))) if sst > 0 else 0.0


def _assoc_matrix(df, num_cols, cat_cols):
    cols = list(num_cols) + list(cat_cols)
    p = len(cols); M = np.eye(p); num_set = set(num_cols)
    numv = {c: pd.to_numeric(df[c], errors="coerce").values for c in num_cols}
    for i in range(p):
        for j in range(i + 1, p):
            ci, cj = cols[i], cols[j]
            in_i, in_j = ci in num_set, cj in num_set
            if in_i and in_j:
                a, b = numv[ci], numv[cj]
                ok = ~(np.isnan(a) | np.isnan(b))
                v = (abs(float(np.corrcoef(a[ok], b[ok])[0, 1]))
                     if ok.sum() > 2 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0 else 0.0)
            elif in_i:
                v = _corr_ratio(df[cj], numv[ci])
            elif in_j:
                v = _corr_ratio(df[ci], numv[cj])
            else:
                v = _cramers_v(df[ci], df[cj])
            M[i, j] = M[j, i] = v if np.isfinite(v) else 0.0
    return M, cols


def association_error(real_df, syn_df, num_cols, cat_cols):
    if len(num_cols) + len(cat_cols) < 2:
        return np.nan
    Ar, cols = _assoc_matrix(real_df, num_cols, cat_cols)
    As, _ = _assoc_matrix(syn_df, num_cols, cat_cols)
    D = np.abs(Ar - As)
    off = D[~np.eye(len(cols), dtype=bool)]
    return float(np.nanmean(off))


# ----------------------- PRDC (Naeem et al. 2020) -----------------------
def prdc(real_e, fake_e, k=5):
    real_e, fake_e = _sub(real_e), _sub(fake_e)
    if len(real_e) <= k or len(fake_e) <= k:
        return {"precision": np.nan, "recall": np.nan, "density": np.nan, "coverage": np.nan}
    def kth_radii(X):
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        d, _ = nn.kneighbors(X)
        return d[:, k]
    rr = kth_radii(real_e); fr = kth_radii(fake_e)
    d_rf = cdist(real_e, fake_e)  # real x fake
    precision = float((d_rf.min(0) <= rr[d_rf.argmin(0)]).mean()) if False else \
        float((d_rf <= rr[:, None]).any(0).mean())            # fake within some real ball
    recall = float((d_rf <= fr[None, :]).any(1).mean())        # real within some fake ball
    density = float((1.0 / k) * (d_rf <= rr[:, None]).sum(0).mean())
    coverage = float((d_rf.min(1) <= rr).mean())               # real has a fake in its ball
    return {"precision": precision, "recall": recall, "density": density, "coverage": coverage}


# ----------------------- SDMetrics QualityReport -----------------------
def sdmetrics_quality(real_df, syn_df, num_cols, cat_cols, target):
    from sdmetrics.reports.single_table import QualityReport
    meta = {"columns": {}}
    for c in list(num_cols):
        meta["columns"][c] = {"sdtype": "numerical"}
    for c in list(cat_cols) + [target]:
        meta["columns"][c] = {"sdtype": "categorical"}
    cols = list(num_cols) + list(cat_cols) + [target]
    rep = QualityReport()
    rep.generate(real_df[cols], syn_df[cols], meta, verbose=False)
    props = rep.get_properties().set_index("Property")["Score"].to_dict()
    return (float(rep.get_score()),
            float(props.get("Column Shapes", np.nan)),
            float(props.get("Column Pair Trends", np.nan)))


# ----------------------- top-level -----------------------
def _run(name, fn, out, keys):
    try:
        v = fn()
        if isinstance(v, tuple):
            for k, val in zip(keys, v):
                out[k] = float(val) if val is not None and np.isfinite(val) else np.nan
        else:
            out[keys[0]] = float(v) if v is not None and np.isfinite(v) else np.nan
    except Exception as e:
        for k in keys:
            out[k] = np.nan
        warnings.warn(f"ext metric {name} failed: {e}")


def evaluate_ext(b, df_syn):
    """b = data.py DatasetBundle; df_syn = synthetic DataFrame (same columns)."""
    out = {}
    tr, te = b.df_train, b.df_test
    feat = b.feature_cols
    df_syn = df_syn.copy()
    for c in feat:
        if c not in df_syn.columns:
            df_syn[c] = np.nan
    # encoded mixed-space (consistent with the rest of the pipeline)
    Rtr = _sub(b.pp.transform_X(tr)); Rho = b.pp.transform_X(te); S = _sub(b.pp.transform_X(df_syn))

    _run("c2st", lambda: c2st(Rtr, S), out, ["ext_c2st"])
    _run("mmd", lambda: mmd_rbf(Rtr, S), out, ["ext_mmd"])
    _run("fpcad", lambda: fpcad(Rtr, S), out, ["ext_fpcad"])
    _run("wasserstein", lambda: wasserstein_feats(tr, df_syn, b.num_cols), out, ["ext_wasserstein"])
    _run("divergences", lambda: divergences(tr, df_syn, feat, b.cat_cols), out,
         ["ext_kl", "ext_tvd", "ext_hellinger"])
    _run("dcr_nndr", lambda: dcr_nndr(S, Rtr, Rho), out,
         ["ext_dcr", "ext_nndr", "ext_dcr_closer_train"])
    _run("nnaa", lambda: nnaa(Rtr, S), out, ["ext_nnaa"])
    _run("mia", lambda: mia_blackbox(Rtr, Rho, S), out, ["ext_mia_auc"])
    _run("dup", lambda: duplicate_rate(tr, df_syn, feat), out, ["ext_dup_rate"])
    _run("assoc", lambda: association_error(tr, df_syn, b.num_cols, b.cat_cols), out, ["ext_assoc_err"])
    _run("prdc", lambda: _prdc_tuple(prdc(Rtr, S)), out,
         ["ext_precision", "ext_recall", "ext_density", "ext_coverage"])
    _run("sdm", lambda: sdmetrics_quality(tr, df_syn, b.num_cols, b.cat_cols, b.target), out,
         ["ext_sdm_quality", "ext_sdm_shapes", "ext_sdm_pairs"])
    return out


def _prdc_tuple(d):
    return (d["precision"], d["recall"], d["density"], d["coverage"])


if __name__ == "__main__":
    from data import load_dataset
    b = load_dataset("heart_failure", 0)
    syn = b.df_train.sample(len(b.df_train), replace=True, random_state=1).reset_index(drop=True)
    for c in b.num_cols:
        syn[c] = pd.to_numeric(syn[c], errors="coerce") + np.random.normal(0, 0.05, len(syn))
    out = evaluate_ext(b, syn)
    print("heart_failure bootstrap synthetic -- enriched metrics:")
    for k, v in out.items():
        print(f"  {k:22s} {round(v,4) if v==v else 'NaN'}")
