"""
Evaluate baseline synthetic CSVs under the SAME protocol as DISCO-TAB.

Reads synthetic/baselines/<dataset>__<method>__seed<seed>.csv, rebuilds the
matching DatasetBundle (same seed/split), runs metrics.evaluate_all, and writes
results/<dataset>__<method>__seed<seed>.json so stats.py can aggregate every
method uniformly.
"""
import os, re, json, glob, warnings
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))

from data import load_dataset
import metrics as M

PAT = re.compile(r"(?P<ds>.+?)__(?P<method>.+?)__seed(?P<seed>\d+)\.csv$")


def evaluate_csv(path, log=print):
    m = PAT.search(os.path.basename(path))
    if not m:
        log(f"skip (bad name): {path}")
        return None
    ds, method, seed = m["ds"], m["method"], int(m["seed"])
    try:
        b = load_dataset(ds, seed=seed)
    except Exception as e:
        log(f"skip {ds}: {e}")
        return None
    df_syn = pd.read_csv(path)
    # Align columns to the schema (baselines should already match).
    cols = [c for c in b.df_train.columns if c in df_syn.columns]
    df_syn = df_syn[cols]
    out = os.path.join(HERE, "results", f"{ds}__{method}__seed{seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # Merge into any existing JSON so previously-computed ext_* enrichment is
    # NOT clobbered by a re-score (evaluate_ext and this can run concurrently).
    res = {}
    if os.path.exists(out):
        try:
            res = json.load(open(out))
        except Exception:
            res = {}
    res.update({"dataset": ds, "seed": seed, "tag": method, "n_syn": len(df_syn)})
    if len(df_syn) >= 10:
        res.update(M.evaluate_all(b, df_syn, seed=seed))
        res["trtr_f1_macro"] = M.trtr_f1(b, seed=seed)["f1_macro"]
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    log(f"[{ds} {method} seed{seed}] TSTR_F1={res.get('tstr_f1_macro')} "
        f"JS={res.get('fid_JS')} DCR={res.get('dcr_median')}")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=os.path.join(HERE, "synthetic", "baselines", "*.csv"))
    args = ap.parse_args()
    files = sorted(glob.glob(args.glob))
    print(f"evaluating {len(files)} baseline CSVs")
    for p in files:
        try:
            evaluate_csv(p)
        except Exception as e:
            print(f"ERROR {p}: {e}")
