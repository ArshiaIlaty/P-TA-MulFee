"""
Fetch the 15 expansion datasets as data_raw/<name>.csv with a `target` column,
so the existing pipeline (data.py / run_experiment / baselines / metrics) picks
them up unchanged. 10 SurvSet clinical cohorts (target = event indicator) + 5
TabArena/OpenML classification sets.

Run from disco/:  python fetch_new_datasets.py
"""
import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_raw")
os.makedirs(RAW, exist_ok=True)

# name -> domain, for registering in data.py
SURVSET = {
    "surv_melanoma": ("Melanoma", "Medical"),
    "surv_pbc": ("pbc", "Medical"),
    "surv_pbc3": ("Pbc3", "Medical"),
    "surv_retinopathy": ("retinopathy", "Medical"),
    "surv_whas500": ("whas500", "Medical"),
    "surv_gbsg2": ("GBSG2", "Medical"),
    "surv_colon": ("colon", "Medical"),
    "surv_grace": ("grace", "Medical"),
    "surv_trace": ("TRACE", "Medical"),
    "surv_framingham": ("Framingham", "Medical"),
}
# openml name -> (our name, version); use sklearn.fetch_openml by name
OPENML = {
    "tab_blood_transfusion": ("blood-transfusion-service-center", 1),
    "tab_credit_g": ("credit-g", 1),
    "tab_qsar_biodeg": ("qsar-biodeg", 1),
    "tab_maternal_risk": ("Maternal-Health-Risk", 1),
    "tab_website_phishing": ("website-phishing", 1),
}


def clean_survset(df):
    # drop identifiers / survival time; event is the classification target.
    df = df.drop(columns=[c for c in ["pid", "time", "time2"] if c in df.columns])
    df["target"] = (pd.to_numeric(df["event"], errors="coerce") > 0).astype(int)
    df = df.drop(columns=["event"])
    for c in df.columns:
        if c == "target":
            continue
        if str(c).startswith("num_"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = df[c].astype(str)
    df = df.dropna(axis=1, how="all")
    # median-fill numeric, mode/__NA__ for categoricals
    for c in df.columns:
        if c == "target":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].replace("nan", "__NA__").fillna("__NA__")
    return df


def fetch_survset():
    from SurvSet.data import SurvLoader
    L = SurvLoader()
    out = []
    for name, (ds, domain) in SURVSET.items():
        try:
            df = L.load_dataset(ds)["df"]
            df = clean_survset(df)
            df = df.drop_duplicates().reset_index(drop=True)
            df.to_csv(os.path.join(RAW, f"{name}.csv"), index=False)
            vc = df["target"].value_counts().to_dict()
            out.append((name, domain, len(df), df.shape[1] - 1, vc))
            print(f"[OK] {name:22s} N={len(df):>5d} feat={df.shape[1]-1:>2d} target={vc}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
    return out


def fetch_openml():
    from sklearn.datasets import fetch_openml
    out = []
    for name, (oml, ver) in OPENML.items():
        got = None
        for args in ([{"name": oml, "version": ver}, {"name": oml}]):
            try:
                got = fetch_openml(as_frame=True, parser="auto", **args)
                break
            except Exception as e:
                last = e
        if got is None:
            print(f"[FAIL] {name} ({oml}): {last}")
            continue
        X = got.data.copy()
        y = got.target
        X["target"] = y.astype(str).values
        # coerce obvious numerics; leave the rest categorical
        for c in X.columns:
            if c == "target":
                continue
            if X[c].dtype.name in ("category", "object"):
                conv = pd.to_numeric(X[c], errors="coerce")
                if conv.notna().mean() > 0.95:
                    X[c] = conv
        X = X.dropna(axis=1, how="all")
        for c in X.columns:
            if c == "target":
                continue
            if pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].fillna(X[c].median())
            else:
                X[c] = X[c].astype(str).replace("nan", "__NA__").fillna("__NA__")
        X = X.dropna(subset=["target"]).drop_duplicates().reset_index(drop=True)
        X.to_csv(os.path.join(RAW, f"{name}.csv"), index=False)
        vc = X["target"].value_counts().to_dict()
        out.append((name, "Tabular", len(X), X.shape[1] - 1, vc))
        print(f"[OK] {name:22s} N={len(X):>5d} feat={X.shape[1]-1:>2d} target={vc}")
    return out


if __name__ == "__main__":
    print("=== SurvSet ==="); s = fetch_survset()
    print("=== OpenML/TabArena ==="); o = fetch_openml()
    print(f"\nFetched {len(s)+len(o)} new datasets into data_raw/.")
    # emit DOMAIN registration snippet
    reg = {**{k: v[1] for k, v in SURVSET.items()}, **{k: "Tabular" for k in OPENML}}
    print("\nDOMAIN additions:", reg)
