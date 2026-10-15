"""
DISCO-TAB evaluation metrics.

All functions take a `DatasetBundle` `b` (see data.py) and a synthetic
`pandas.DataFrame` with the same columns as `b.df_train` (feature cols + "target"),
and return dicts of real computed numbers.

Covers:
  * TSTR / TRTR downstream utility (RandomForest F1)
  * marginal statistical fidelity (KS, JSD, KL, Hellinger, Pearson-corr similarity)
  * empirical disclosure risk (Distance to Closest Record, + real-real baseline)
  * the FAITH decomposition (Factuality, Alignment, Integrity, Tracking)
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, accuracy_score

EPS = 1e-8


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _num_series(df, col):
    """Synthetic numeric column coerced to float, NaNs dropped."""
    return pd.to_numeric(df[col], errors="coerce")


def _prob_vectors_numeric(real_vals, syn_vals, n_bins):
    real_vals = np.asarray(real_vals, dtype=float)
    syn_vals = np.asarray(syn_vals, dtype=float)
    real_vals = real_vals[np.isfinite(real_vals)]
    syn_vals = syn_vals[np.isfinite(syn_vals)]
    lo, hi = np.min(real_vals), np.max(real_vals)
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    # clip synthetic into the real range so out-of-support mass lands in end bins
    syn_clip = np.clip(syn_vals, lo, hi)
    p, _ = np.histogram(real_vals, bins=edges)
    q, _ = np.histogram(syn_clip, bins=edges)
    p = p.astype(float) + EPS
    q = q.astype(float) + EPS
    p /= p.sum()
    q /= q.sum()
    return p, q


def _prob_vectors_categorical(real_vals, syn_vals):
    real_vals = real_vals.astype(str)
    syn_vals = syn_vals.astype(str)
    support = sorted(set(real_vals) | set(syn_vals))
    rc = real_vals.value_counts()
    sc = syn_vals.value_counts()
    p = np.array([rc.get(c, 0) for c in support], dtype=float) + EPS
    q = np.array([sc.get(c, 0) for c in support], dtype=float) + EPS
    p /= p.sum()
    q /= q.sum()
    return p, q


def _jsd(p, q):
    # scipy returns the JS *distance*; square it to get the divergence in [0,1].
    d = jensenshannon(p, q, base=2)
    return float(d * d)


def _kl(p, q):
    return float(np.sum(p * np.log(p / q)))


def _hellinger(p, q):
    return float(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2.0))


def _encode(b, df):
    """Mixed standardized-numeric + one-hot encoding (fit on real train)."""
    return b.pp.transform_X(df)


def _nn_distances(reference_X, query_X, n_encoded):
    """Per-query Euclidean distance to nearest reference row, / sqrt(dims)."""
    if reference_X.shape[0] == 0 or query_X.shape[0] == 0:
        return np.array([])
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(reference_X)
    d, _ = nn.kneighbors(query_X)
    denom = np.sqrt(max(n_encoded, 1))
    return d[:, 0] / denom


# --------------------------------------------------------------------------- #
# 1. downstream utility
# --------------------------------------------------------------------------- #
def _fit_clf(X, y, clf, n_estimators, seed):
    if clf == "rf":
        m = RandomForestClassifier(n_estimators=n_estimators, random_state=seed,
                                   n_jobs=-1)
    else:
        raise ValueError(f"unknown clf {clf}")
    m.fit(X, y)
    return m


def tstr_f1(b, df_syn, clf="rf", n_estimators=200, seed=0):
    Xtr, ytr = b.pp.xy(df_syn)
    keep = ytr >= 0
    Xtr, ytr = Xtr[keep], ytr[keep]
    Xte, yte = b.pp.xy(b.df_test)
    n_used = int(keep.sum())
    if n_used == 0 or len(np.unique(ytr)) < 2:
        return {"f1_macro": 0.0, "f1_weighted": 0.0, "accuracy": 0.0,
                "n_train_used": n_used}
    m = _fit_clf(Xtr, ytr, clf, n_estimators, seed)
    pred = m.predict(Xte)
    return {
        "f1_macro": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(yte, pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(yte, pred)),
        "n_train_used": n_used,
    }


def trtr_f1(b, clf="rf", n_estimators=200, seed=0):
    Xtr, ytr = b.pp.xy(b.df_train)
    Xte, yte = b.pp.xy(b.df_test)
    m = _fit_clf(Xtr, ytr, clf, n_estimators, seed)
    pred = m.predict(Xte)
    return {
        "f1_macro": float(f1_score(yte, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(yte, pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(yte, pred)),
    }


# --------------------------------------------------------------------------- #
# 2. marginal statistical fidelity
# --------------------------------------------------------------------------- #
def fidelity(b, df_syn, n_bins=20):
    js, kl, hell, ks = [], [], [], []
    js_num, js_cat = [], []

    for col in b.num_cols:
        real = pd.to_numeric(b.df_train[col], errors="coerce").dropna().values
        syn = _num_series(df_syn, col).dropna().values
        if len(real) == 0 or len(syn) == 0:
            continue
        p, q = _prob_vectors_numeric(real, syn, n_bins)
        j = _jsd(p, q)
        js.append(j); js_num.append(j)
        kl.append(_kl(p, q))
        hell.append(_hellinger(p, q))
        try:
            ks.append(float(ks_2samp(real, syn).statistic))
        except Exception:
            pass

    for col in b.cat_cols:
        real = b.df_train[col].astype(str)
        syn = df_syn[col].astype(str)
        if len(real) == 0 or len(syn) == 0:
            continue
        p, q = _prob_vectors_categorical(real, syn)
        j = _jsd(p, q)
        js.append(j); js_cat.append(j)
        kl.append(_kl(p, q))
        hell.append(_hellinger(p, q))

    # Pearson correlation-structure similarity on numeric columns.
    pearson = np.nan
    if len(b.num_cols) >= 2:
        real_num = b.df_train[b.num_cols].apply(pd.to_numeric, errors="coerce")
        syn_num = df_syn[b.num_cols].apply(pd.to_numeric, errors="coerce")
        cr = real_num.corr().values
        cs = syn_num.corr().values
        iu = np.triu_indices_from(cr, k=1)
        diff = np.abs(cr[iu] - cs[iu])
        diff = diff[np.isfinite(diff)]
        if diff.size:
            pearson = float(np.clip(1.0 - diff.mean(), -1.0, 1.0))

    mean = lambda a: float(np.mean(a)) if len(a) else np.nan
    return {
        "JS": mean(js), "KL": mean(kl), "Hellinger": mean(hell), "KS": mean(ks),
        "Pearson": pearson, "JS_num": mean(js_num), "JS_cat": mean(js_cat),
    }


# --------------------------------------------------------------------------- #
# 3. empirical disclosure risk (DCR)
# --------------------------------------------------------------------------- #
def dcr(b, df_syn, ref_frac=0.5, seed=0):
    n_enc = b.pp.n_encoded
    ref_X = _encode(b, b.df_test)              # reference disjoint from gen data
    syn_X = _encode(b, df_syn)
    d_syn = _nn_distances(ref_X, syn_X, n_enc)

    # real-real baseline: two disjoint halves of the held-out real test set.
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(b.df_test))
    cut = int(len(idx) * ref_frac)
    a_idx, c_idx = idx[:cut], idx[cut:]
    rr_median = np.nan
    if len(a_idx) and len(c_idx):
        Xa = _encode(b, b.df_test.iloc[a_idx])
        Xc = _encode(b, b.df_test.iloc[c_idx])
        d_rr = _nn_distances(Xa, Xc, n_enc)
        if d_rr.size:
            rr_median = float(np.median(d_rr))

    if d_syn.size == 0:
        return {"dcr_median": np.nan, "dcr_mean": np.nan, "dcr_p5": np.nan,
                "dcr_realreal_median": rr_median}
    return {
        "dcr_median": float(np.median(d_syn)),
        "dcr_mean": float(np.mean(d_syn)),
        "dcr_p5": float(np.percentile(d_syn, 5)),
        "dcr_realreal_median": rr_median,
    }


# --------------------------------------------------------------------------- #
# 4. FAITH decomposition
# --------------------------------------------------------------------------- #
def _support(b):
    num_rng, cat_set = {}, {}
    for c in b.num_cols:
        v = pd.to_numeric(b.df_train[c], errors="coerce")
        num_rng[c] = (float(v.min()), float(v.max()))
    for c in b.cat_cols:
        cat_set[c] = set(b.df_train[c].astype(str).unique())
    return num_rng, cat_set


def _in_support_mask(b, df_syn, num_rng, cat_set):
    """Boolean per-row: True iff every cell lies within the real-train support."""
    ok = np.ones(len(df_syn), dtype=bool)
    for c in b.num_cols:
        lo, hi = num_rng[c]
        v = pd.to_numeric(df_syn[c], errors="coerce").values
        ok &= np.isfinite(v) & (v >= lo) & (v <= hi)
    for c in b.cat_cols:
        v = df_syn[c].astype(str).values
        ok &= np.array([x in cat_set[c] for x in v])
    return ok


def faith(b, df_syn, eps=None):
    num_rng, cat_set = _support(b)

    # Factuality: all cells within support (data-driven domain constraints).
    fact_mask = _in_support_mask(b, df_syn, num_rng, cat_set)
    s_fact = float(fact_mask.mean()) if len(df_syn) else 0.0

    # Tracking: 1 - fraction outside support (= same mask here).
    s_track = float(fact_mask.mean()) if len(df_syn) else 0.0

    # Alignment: numeric correlation-structure match.
    s_align = np.nan
    if len(b.num_cols) >= 2:
        d = len(b.num_cols)
        cr = b.df_train[b.num_cols].apply(pd.to_numeric, errors="coerce").corr().values
        cs = df_syn[b.num_cols].apply(pd.to_numeric, errors="coerce").corr().values
        fro = np.linalg.norm(np.nan_to_num(cr - cs), ord="fro")
        s_align = float(np.clip(1.0 - fro / d, 0.0, 1.0))

    # Integrity: fraction of synthetic rows farther than eps from any real record.
    n_enc = b.pp.n_encoded
    ref_X = _encode(b, b.df_test)
    syn_X = _encode(b, df_syn)
    d_syn = _nn_distances(ref_X, syn_X, n_enc)
    if eps is None:
        eps = dcr(b, df_syn).get("dcr_realreal_median", np.nan)
    if d_syn.size and np.isfinite(eps):
        s_integ = float(np.mean(d_syn > eps))
    else:
        s_integ = np.nan

    parts = [s_fact, s_align, s_integ, s_track]
    valid = [p for p in parts if np.isfinite(p)]
    overall = float(0.25 * np.nansum([p if np.isfinite(p) else 0.0 for p in parts])) \
        if valid else np.nan
    return {
        "FAITH": overall, "Factuality": s_fact,
        "Alignment": s_align, "Integrity": s_integ, "Tracking": s_track,
    }


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
def evaluate_all(b, df_syn, seed=0):
    out = {}

    def run(name, fn, prefix):
        try:
            res = fn()
            for k, v in res.items():
                out[f"{prefix}{k}"] = v
        except Exception as e:  # noqa
            warnings.warn(f"metric '{name}' failed: {e}")
            # best-effort: still register the expected keys as NaN downstream

    run("tstr", lambda: tstr_f1(b, df_syn, seed=seed), "tstr_")
    run("fidelity", lambda: fidelity(b, df_syn), "fid_")
    run("dcr", lambda: dcr(b, df_syn, seed=seed), "")
    run("faith", lambda: faith(b, df_syn), "faith_")
    out["n_syn"] = int(len(df_syn))
    return out


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
def _bootstrap_synth(b, n=None, jitter=0.05, seed=0):
    rng = np.random.RandomState(seed)
    n = n or len(b.df_train)
    s = b.df_train.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
    for c in b.num_cols:
        v = pd.to_numeric(s[c], errors="coerce").astype(float)
        sd = pd.to_numeric(b.df_train[c], errors="coerce").std()
        s[c] = v + rng.normal(0, jitter * (sd if np.isfinite(sd) and sd > 0 else 1.0), size=len(s))
    return s


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pprint import pprint
    import data

    for name in ["heart_failure", "obesity"]:
        print("=" * 70)
        print("DATASET:", name)
        b = data.load_dataset(name, seed=0)
        syn = _bootstrap_synth(b, seed=0)
        print("  TRTR ceiling:", trtr_f1(b))
        res = evaluate_all(b, syn, seed=0)
        pprint(res)
