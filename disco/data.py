"""
DISCO-TAB: dataset bundle + canonical preprocessing (shared contract).

`load_dataset(name, seed)` returns a DatasetBundle with a stratified 80/20
train/test split and a Preprocessor fit on the REAL TRAIN partition only
(standardize numeric by train stats, one-hot categoricals). The same
Preprocessor is used for TSTR classification and for DCR distances so every
method is scored identically.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_raw")

DOMAIN = {
    "heart_failure": "Medical", "breast_cancer": "Medical",
    "liver_disorders": "Medical", "parkinsons": "Medical",
    "obesity": "Medical", "diabetes": "Medical",
    "german_credit": "Finance", "bank_marketing": "Finance",
    "census_income": "Social",
    # --- expansion: SurvSet clinical cohorts (target = event indicator) ---
    "surv_melanoma": "Medical", "surv_pbc": "Medical", "surv_pbc3": "Medical",
    "surv_retinopathy": "Medical", "surv_whas500": "Medical",
    "surv_gbsg2": "Medical", "surv_colon": "Medical", "surv_grace": "Medical",
    "surv_trace": "Medical", "surv_framingham": "Medical",
    # --- expansion: TabArena/OpenML classification sets ---
    "tab_blood_transfusion": "Tabular", "tab_credit_g": "Finance",
    "tab_qsar_biodeg": "Chemistry", "tab_maternal_risk": "Medical",
    "tab_website_phishing": "Web",
}
EXPANSION_DATASETS = [k for k in DOMAIN if k.startswith(("surv_", "tab_"))]
ALL_DATASETS = list(DOMAIN.keys())
TARGET = "target"


def _ohe():
    # sklearn >=1.2 renamed `sparse` -> `sparse_output`
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_types(df, target=TARGET):
    feats = [c for c in df.columns if c != target]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in feats if c not in num]
    return num, cat


class Preprocessor:
    """Fit on real train; transform real or synthetic frames to a numeric matrix."""

    def __init__(self, num_cols, cat_cols, target=TARGET):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.target = target
        self.scaler = StandardScaler()
        self.ohe = _ohe()
        self.le = LabelEncoder()
        self.num_means_ = None

    def fit(self, df):
        if self.num_cols:
            num = df[self.num_cols].apply(pd.to_numeric, errors="coerce")
            self.num_means_ = num.mean()
            self.scaler.fit(num.fillna(self.num_means_))
        if self.cat_cols:
            self.ohe.fit(df[self.cat_cols].astype(str))
        self.le.fit(df[self.target].astype(str))
        self.classes_ = list(self.le.classes_)
        return self

    def transform_X(self, df):
        parts = []
        if self.num_cols:
            num = df[self.num_cols].apply(pd.to_numeric, errors="coerce")
            num = num.fillna(self.num_means_)
            parts.append(self.scaler.transform(num))
        if self.cat_cols:
            parts.append(self.ohe.transform(df[self.cat_cols].astype(str)))
        return np.hstack(parts) if parts else np.zeros((len(df), 0))

    def transform_y(self, df):
        # Unseen labels in synthetic data map to -1 (filtered by callers).
        y = df[self.target].astype(str)
        known = set(self.classes_)
        mapped = y.where(y.isin(known), other=None)
        out = np.full(len(df), -1, dtype=int)
        mask = mapped.notna().values
        if mask.any():
            out[mask] = self.le.transform(mapped[mask].astype(str))
        return out

    def xy(self, df):
        return self.transform_X(df), self.transform_y(df)

    @property
    def n_encoded(self):
        d = len(self.num_cols)
        if self.cat_cols:
            d += sum(len(c) for c in self.ohe.categories_)
        return d


class DatasetBundle:
    def __init__(self, name, seed, test_size=0.2):
        self.name = name
        self.seed = seed
        self.domain = DOMAIN[name]
        self.target = TARGET
        path = os.path.join(RAW, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing — run fetch_datasets.py first")
        df = pd.read_csv(path)
        df[self.target] = df[self.target].astype(str)
        self.num_cols, self.cat_cols = infer_types(df, self.target)
        self.feature_cols = self.num_cols + self.cat_cols
        # Stratified split; fall back to unstratified if a class is too rare.
        strat = df[self.target]
        try:
            tr, te = train_test_split(df, test_size=test_size, random_state=seed,
                                      stratify=strat)
        except ValueError:
            tr, te = train_test_split(df, test_size=test_size, random_state=seed)
        self.df_train = tr.reset_index(drop=True)
        self.df_test = te.reset_index(drop=True)
        self.pp = Preprocessor(self.num_cols, self.cat_cols, self.target).fit(self.df_train)
        self.n_classes = len(self.pp.classes_)
        self.class_counts = self.df_train[self.target].value_counts().to_dict()

    def __repr__(self):
        return (f"<DatasetBundle {self.name} seed={self.seed} "
                f"train={len(self.df_train)} test={len(self.df_test)} "
                f"num={len(self.num_cols)} cat={len(self.cat_cols)} "
                f"cls={self.n_classes}>")


def load_dataset(name, seed=0, test_size=0.2):
    return DatasetBundle(name, seed, test_size)


if __name__ == "__main__":
    for n in ALL_DATASETS:
        try:
            b = load_dataset(n, seed=0)
            print(b, "| encoded dims:", b.pp.n_encoded)
        except Exception as e:
            print(n, "ERR", e)
