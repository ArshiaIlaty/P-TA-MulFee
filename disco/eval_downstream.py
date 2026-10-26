"""Extended downstream (TSTR) evaluation: three classifiers
(RandomForest, LogisticRegression, XGBoost) x four metrics
(macro-F1, AUROC, AUPRC, minority-class recall). Minority-class recall directly
probes whether inverse-frequency reward shaping (IFRS) preserves the rare class.

Re-scores the synthetic CSVs we already have (no regeneration). CPU-only; run in
the `discodiff` env (xgboost + sklearn). Resumable: appends to
tables/comprehensive/downstream_ext.csv and skips (dataset,model,seed,clf) rows
already present.

  conda run -n discodiff python eval_downstream.py            # all synthetic CSVs
  conda run -n discodiff python eval_downstream.py --glob "synthetic/*__disco__*.csv"
"""
import os, re, glob, argparse, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from data import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
                             recall_score, precision_score)
from sklearn.preprocessing import label_binarize
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tables", "comprehensive", "downstream_ext.csv")
PAT = re.compile(r"(?P<ds>.+?)__(?P<model>.+?)__seed(?P<seed>\d+)\.csv$")
# Cap worker threads so this CPU job can run alongside the LLM campaign / sweep
# on an 8-core box instead of grabbing every core with n_jobs=-1.
THREADS = int(os.environ.get("EVAL_THREADS", "3"))
COLS = ["dataset", "model", "seed", "clf", "f1_macro", "auroc", "auprc",
        "minority_recall", "minority_precision", "minority_f1"]


def make_clfs(seed):
    return {
        "rf": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=THREADS),
        "logreg": LogisticRegression(max_iter=2000, random_state=seed),
        "xgb": xgb.XGBClassifier(n_estimators=200, random_state=seed, n_jobs=THREADS,
                                 tree_method="hist", eval_metric="logloss",
                                 verbosity=0),
    }


def eval_one(Xtr, ytr, Xte, yte, clf, minority):
    r = {"f1_macro": np.nan, "auroc": np.nan, "auprc": np.nan,
         "minority_recall": np.nan, "minority_precision": np.nan,
         "minority_f1": np.nan}
    if len(np.unique(ytr)) < 2:            # collapsed synthetic -> cannot learn
        r["f1_macro"] = 0.0
        r["minority_recall"] = 0.0; r["minority_precision"] = 0.0; r["minority_f1"] = 0.0
        return r
    try:
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
    except Exception:
        return r
    r["f1_macro"] = float(f1_score(yte, pred, average="macro", zero_division=0))
    rec = recall_score(yte, pred, labels=[minority], average=None, zero_division=0)
    r["minority_recall"] = float(rec[0]) if len(rec) else np.nan
    prec = precision_score(yte, pred, labels=[minority], average=None, zero_division=0)
    r["minority_precision"] = float(prec[0]) if len(prec) else np.nan
    mf1 = f1_score(yte, pred, labels=[minority], average=None, zero_division=0)
    r["minority_f1"] = float(mf1[0]) if len(mf1) else np.nan
    try:
        proba = clf.predict_proba(Xte)
        trained = list(clf.classes_)
        test_classes = np.unique(yte)
        if len(trained) < len(test_classes):      # a real class was never trained
            return r
        if len(trained) == 2:
            pi = trained.index(minority) if minority in trained else 1
            ybin = (yte == trained[pi]).astype(int)
            r["auroc"] = float(roc_auc_score(ybin, proba[:, pi]))
            r["auprc"] = float(average_precision_score(ybin, proba[:, pi]))
        else:
            yb = label_binarize(yte, classes=trained)
            r["auroc"] = float(roc_auc_score(yb, proba, average="macro", multi_class="ovr"))
            r["auprc"] = float(average_precision_score(yb, proba, average="macro"))
    except Exception:
        pass
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", default=[
        os.path.join(HERE, "synthetic", "baselines", "*.csv"),
        os.path.join(HERE, "synthetic", "*__seed*.csv")])
    args = ap.parse_args()

    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.dataset, prev.model, prev.seed, prev.clf))
    if not os.path.exists(OUT):
        pd.DataFrame(columns=COLS).to_csv(OUT, index=False)  # header once
    files = sorted({f for g in args.glob for f in glob.glob(g)})
    bundles = {}
    total = 0
    for path in files:
        m = PAT.search(os.path.basename(path))
        if not m:
            continue
        ds, model, seed = m["ds"], m["model"], int(m["seed"])
        if all((ds, model, seed, c) in done for c in ("rf", "logreg", "xgb")):
            continue
        key = (ds, seed)
        try:
            b = bundles.get(key) or bundles.setdefault(key, load_dataset(ds, seed=seed))
            df_syn = pd.read_csv(path)
            Xtr, ytr = b.pp.xy(df_syn); keep = ytr >= 0; Xtr, ytr = Xtr[keep], ytr[keep]
            Xte, yte = b.pp.xy(b.df_test)
            _, ytr_real = b.pp.xy(b.df_train)
            minority = int(pd.Series(ytr_real[ytr_real >= 0]).value_counts().idxmin())
        except Exception as e:
            print(f"[skip] {ds} {model} seed{seed}: {e}", flush=True)
            continue
        # per-file rows, appended immediately so the run is resumable, monitorable,
        # and never loses hours of work if interrupted.
        file_rows = []
        for cname, clf in make_clfs(seed).items():
            if (ds, model, seed, cname) in done:
                continue
            r = eval_one(Xtr, ytr, Xte, yte, clf, minority)
            file_rows.append({"dataset": ds, "model": model, "seed": seed,
                              "clf": cname, **r})
            done.add((ds, model, seed, cname))
        if file_rows:
            pd.DataFrame(file_rows)[COLS].to_csv(OUT, mode="a", header=False, index=False)
            total += len(file_rows)
        print(f"[ok] {ds} {model} seed{seed} (+{len(file_rows)} rows, {total} total)",
              flush=True)
    print(f"done: wrote {total} new rows -> {OUT}")


if __name__ == "__main__":
    main()
