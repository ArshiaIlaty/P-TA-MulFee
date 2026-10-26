"""
Synthcity model-zoo baselines for the DISCO-TAB benchmark.

Adds a diverse generative zoo (beyond SDV ctgan/tvae/gaussian_copula and the
synthcity tabddpm) via the GENERIC synthcity loader, writing CSVs named exactly
like the other baselines so evaluate_baselines.py scores them unchanged.

Run with the `discodiff` env:
  conda run -n discodiff python baselines_synthcity.py --datasets ... --models ... --seeds 0 1 2
Importable from base for the helpers below (synthcity imported lazily).
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data_raw")
OUT = os.path.join(HERE, "synthetic", "baselines")
os.makedirs(OUT, exist_ok=True)

# synthcity plugin id -> our method label (all via the generic loader)
MODELS = {
    "adsgan": "adsgan",
    "pategan": "pategan",
    "rtvae": "rtvae",
    "nflow": "nflow",
    "arf": "arf",
    "bayesian_network": "bayesian_network",
    "marginal_distributions": "marginal",
}
# privbayes dropped: hangs on structure learning and is redundant with
# bayesian_network (same Bayesian/graphical family).
ALL_DATASETS = [
    "heart_failure", "breast_cancer", "liver_disorders", "parkinsons", "obesity",
    "diabetes", "german_credit", "bank_marketing", "census_income",
    "surv_melanoma", "surv_pbc", "surv_pbc3", "surv_retinopathy", "surv_whas500",
    "surv_gbsg2", "surv_colon", "surv_grace", "surv_trace", "surv_framingham",
    "tab_blood_transfusion", "tab_credit_g", "tab_qsar_biodeg", "tab_maternal_risk",
    "tab_website_phishing",
]
LARGE = {"diabetes", "bank_marketing", "census_income"}


def split(df, seed):
    from sklearn.model_selection import train_test_split
    try:
        tr, _ = train_test_split(df, test_size=0.2, random_state=seed,
                                 stratify=df["target"])
    except ValueError:
        tr, _ = train_test_split(df, test_size=0.2, random_state=seed)
    return tr.reset_index(drop=True)


def infer_cats(df):
    return [c for c in df.columns if c != "target"
            and not pd.api.types.is_numeric_dtype(df[c])]


def epoch_schedule(n):
    if n <= 500:
        return 300
    if n <= 2000:
        return 200
    if n <= 10000:
        return 150
    return 100


def coerce_target(syn, real_target):
    classes = pd.unique(real_target)
    cset = set(str(c) for c in classes)
    numeric = all(_is_num(c) for c in classes)
    svals = syn["target"]
    out = []
    numclasses = [float(c) for c in classes] if numeric else None
    mode = real_target.value_counts().index[0]
    for v in svals:
        s = str(v)
        if s in cset:
            out.append(_match(v, classes))
        elif numeric and _is_num(v):
            # nearest numeric class
            fv = float(v)
            out.append(classes[int(np.argmin([abs(fv - nc) for nc in numclasses]))])
        else:
            out.append(mode)
    syn = syn.copy()
    syn["target"] = out
    return syn


def _is_num(x):
    try:
        float(x); return True
    except (ValueError, TypeError):
        return False


def _match(v, classes):
    for c in classes:
        if str(c) == str(v):
            return c
    return classes[0]


def generate_one(dataset, model_id, label, seed, log):
    path = os.path.join(OUT, f"{dataset}__{label}__seed{seed}.csv")
    if os.path.exists(path):
        log(f"skip (exists) {dataset} {label} seed{seed}")
        return
    from synthcity.plugins import Plugins
    from synthcity.plugins.core.dataloader import GenericDataLoader
    df = pd.read_csv(os.path.join(RAW, f"{dataset}.csv"))
    df["target"] = df["target"].astype(str)
    tr = split(df, seed)
    n_gen = len(tr)
    fit_df = tr
    if len(tr) > 6000:
        fit_df = tr.sample(6000, random_state=0).reset_index(drop=True)
    if dataset in LARGE:
        n_gen = min(n_gen, 5000)
    cats = infer_cats(fit_df)
    epochs = epoch_schedule(len(fit_df))
    loader = GenericDataLoader(fit_df, discrete_columns=cats + ["target"])

    def _fit_gen(device=None):
        kw = {"n_iter": epochs}
        if device is not None:
            kw["device"] = device
        try:
            m = Plugins().get(model_id, **kw)
        except TypeError:
            kw.pop("n_iter", None)
            try:
                m = Plugins().get(model_id, **kw)
            except TypeError:
                m = Plugins().get(model_id)
        m.fit(loader)
        return m.generate(count=n_gen).dataframe()

    try:
        syn = _fit_gen()
    except Exception as e:
        if "cuda" in str(e).lower() or "memory" in str(e).lower():
            log(f"  OOM {dataset} {label}: retry CPU")
            syn = _fit_gen(device="cpu")
        else:
            raise
    # align to schema
    for c in df.columns:
        if c not in syn.columns:
            syn[c] = df[c].iloc[0]
    syn = syn[df.columns]
    syn = coerce_target(syn, df["target"])
    for c in df.columns:
        if c != "target" and pd.api.types.is_numeric_dtype(df[c]):
            syn[c] = pd.to_numeric(syn[c], errors="coerce")
    syn = syn.dropna(axis=0, how="all").reset_index(drop=True)
    syn.to_csv(path, index=False)
    log(f"[OK] {dataset:22s} {label:14s} seed{seed} shape={syn.shape}")


def main():
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    def log(m):
        print(m, flush=True)

    # breadth-first: seed outer so seed 0 lands across everything first
    import signal

    class _TO(Exception):
        pass

    def _alarm(signum, frame):
        raise _TO()
    signal.signal(signal.SIGALRM, _alarm)
    TIMEOUT = int(os.environ.get("SYNTHCITY_TIMEOUT", "900"))  # per (dataset, model) cap so no plugin stalls the batch

    for seed in args.seeds:
        for ds in args.datasets:
            for mid in args.models:
                label = MODELS.get(mid, mid)
                try:
                    signal.alarm(TIMEOUT)
                    generate_one(ds, mid, label, seed, log)
                except _TO:
                    log(f"[TIMEOUT] {ds} {label} seed{seed} (> {TIMEOUT}s), skipped")
                except Exception as e:
                    log(f"[FAIL] {ds} {label} seed{seed}: {type(e).__name__}: {str(e)[:160]}")
                finally:
                    signal.alarm(0)


# base-env-importable helpers (no synthcity import)
def list_synthcity_csvs(dataset, seed):
    out = {}
    for mid, label in MODELS.items():
        p = os.path.join(OUT, f"{dataset}__{label}__seed{seed}.csv")
        if os.path.exists(p):
            out[label] = p
    return out


if __name__ == "__main__":
    main()
