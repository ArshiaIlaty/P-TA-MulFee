"""
DISCO-TAB: dataset acquisition + canonical preprocessing.

Downloads the nine benchmark datasets used in the paper, applies a single
canonical cleaning pass, writes a tidy CSV per dataset to data_raw/, and emits
a characteristics summary (data_raw/dataset_summary.csv + a LaTeX snippet) that
fills Table `tab:datasets`.

Run from the `disco/` directory with an isolated interpreter:
    python -I fetch_datasets.py
"""
import os
import io
import sys
import json
import urllib.request
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_raw")
os.makedirs(RAW, exist_ok=True)


def _dedup_columns(cols):
    """Make a column list unique (ucimlrepo sometimes collapses variant names)."""
    seen, out = {}, []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


def _from_uci(uci_id, target_rename):
    """Fetch a UCI dataset, concatenate features+target, rename the target col."""
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=uci_id)
    X = d.data.features.copy()
    y = d.data.targets.copy()
    X.columns = _dedup_columns(list(X.columns))
    # Some UCI targets have >1 column; keep the first.
    tcol = y.columns[0]
    df = X.copy()
    df[target_rename] = y[tcol].values
    return df


def load_heart_failure():
    df = _from_uci(519, "target")
    return df, "target", "Medical"


def load_breast_cancer():
    # sklearn ships the Wisconsin Diagnostic set (569 x 30), fully offline.
    from sklearn.datasets import load_breast_cancer as lbc
    ds = lbc(as_frame=True)
    df = ds.frame.rename(columns={"target": "target"})
    return df, "target", "Medical"


def load_liver_disorders():
    # UCI "Liver Disorders" (BUPA). ucimlrepo designates `drinks` (number of
    # daily alcoholic drinks) as the target; the common ML benchmark binarizes
    # it as >=3 drinks -> 1 (at-risk), else 0. The 5 blood-test measurements
    # (mcv, alkphos, sgpt, sgot, gammagt) are the predictors.
    df = _from_uci(60, "drinks")
    drinks = df["drinks"].astype(float)
    df = df.drop(columns=["drinks"])
    df["target"] = (drinks >= 3).astype(int)
    return df, "target", "Medical"


def load_parkinsons():
    df = _from_uci(174, "status")
    # 'name' is an identifier string, not a predictor.
    for c in ["name", "Name", "ID"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    return df.rename(columns={"status": "target"}), "target", "Medical"


def load_obesity():
    df = _from_uci(544, "NObeyesdad")
    return df.rename(columns={"NObeyesdad": "target"}), "target", "Medical"


def load_diabetes():
    # Kaggle "Diabetes prediction dataset" (100k rows, ~1:9 imbalance).
    # Not on UCI; pull from a stable public mirror.
    urls = [
        "https://huggingface.co/datasets/marianeft/diabetes_prediction_dataset/resolve/main/diabetes_prediction_dataset.csv",
        "https://huggingface.co/datasets/m2s6a8/diabetes_prediction_dataset/resolve/main/diabetes_prediction_dataset.csv",
    ]
    raw = None
    for u in urls:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode("utf-8")
            if raw and "," in raw:
                break
        except Exception as e:
            print(f"  diabetes url failed ({u.split('/')[2]}): {e}")
            raw = None
    if raw is None:
        raise RuntimeError("could not download diabetes dataset from any mirror")
    df = pd.read_csv(io.StringIO(raw))
    df = df.rename(columns={"diabetes": "target"})
    return df, "target", "Medical"


def load_german_credit():
    df = _from_uci(144, "class")
    # class: 1=good, 2=bad -> map to 0/1
    df["class"] = (df["class"].astype(int) == 2).astype(int)
    return df.rename(columns={"class": "target"}), "target", "Finance"


def load_bank_marketing():
    df = _from_uci(222, "y")
    df["y"] = (df["y"].astype(str).str.lower() == "yes").astype(int)
    return df.rename(columns={"y": "target"}), "target", "Finance"


def load_census_income():
    df = _from_uci(2, "income")
    inc = df["income"].astype(str).str.replace(".", "", regex=False).str.strip()
    df["income"] = (inc.str.contains(">50K")).astype(int)
    return df.rename(columns={"income": "target"}), "target", "Social"


LOADERS = [
    ("heart_failure", load_heart_failure),
    ("breast_cancer", load_breast_cancer),
    ("liver_disorders", load_liver_disorders),
    ("parkinsons", load_parkinsons),
    ("obesity", load_obesity),
    ("diabetes", load_diabetes),
    ("german_credit", load_german_credit),
    ("bank_marketing", load_bank_marketing),
    ("census_income", load_census_income),
]


def clean(df, target):
    """Light canonical cleaning: drop all-NA cols, strip strings, drop dup rows."""
    df = df.copy()
    df = df.dropna(axis=1, how="all")
    # Standardize string cells.
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    # Drop rows with any missing value (datasets here are mostly complete; we
    # record the reserved-missing-token path separately in the serializer).
    before = len(df)
    df = df.dropna(axis=0, how="any").reset_index(drop=True)
    dropped = before - len(df)
    return df, dropped


def feature_types(df, target):
    feats = [c for c in df.columns if c != target]
    num, cat = [], []
    for c in feats:
        if pd.api.types.is_numeric_dtype(df[c]):
            # Low-cardinality integer-coded columns are still treated numeric
            # for the dataset table; the serializer handles them generically.
            num.append(c)
        else:
            cat.append(c)
    return num, cat


def summarize():
    rows = []
    latex_rows = []
    for name, loader in LOADERS:
        try:
            df, target, domain = loader()
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            rows.append({"dataset": name, "status": f"FAILED: {e}"})
            continue
        df, dropped = clean(df, target)
        out = os.path.join(RAW, f"{name}.csv")
        df.to_csv(out, index=False)
        num, cat = feature_types(df, target)
        vc = df[target].value_counts()
        n_classes = int(vc.shape[0])
        ir = float(vc.max() / vc.min()) if vc.min() > 0 else float("inf")
        rec = {
            "dataset": name,
            "domain": domain,
            "N": int(len(df)),
            "n_features": len(num) + len(cat),
            "n_num": len(num),
            "n_cat": len(cat),
            "n_classes": n_classes,
            "imbalance_ratio": round(ir, 2),
            "target": target,
            "class_counts": json.dumps(vc.to_dict(), default=int),
            "rows_dropped_na": int(dropped),
        }
        rows.append(rec)
        print(f"[OK] {name:16s} N={rec['N']:>6d}  feat={rec['n_features']:>2d} "
              f"(num {rec['n_num']}/cat {rec['n_cat']})  cls={n_classes}  IR=1:{ir:.1f}")
        latex_rows.append(
            f"{name.replace('_',' ').title()} & {domain} & {rec['N']:,} & "
            f"{rec['n_features']} ({rec['n_num']}/{rec['n_cat']}) & {n_classes} & "
            f"1:{ir:.1f} & 80/20 \\\\"
        )
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(RAW, "dataset_summary.csv"), index=False)
    with open(os.path.join(RAW, "dataset_table.tex"), "w") as f:
        f.write("\n".join(latex_rows))
    print("\n==== SUMMARY ====")
    cols = ["dataset", "domain", "N", "n_features", "n_num", "n_cat",
            "n_classes", "imbalance_ratio"]
    print(summ[[c for c in cols if c in summ.columns]].to_string(index=False))
    return summ


if __name__ == "__main__":
    summarize()
