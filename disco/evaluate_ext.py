"""
Merge the enriched metrics battery (metrics_ext.evaluate_ext) into every
results/<dataset>__<method>__seed<seed>.json, by re-reading the synthetic CSV
for each run. Resumable: skips a run whose JSON already has an "ext_c2st" key
unless --force.

  python evaluate_ext.py            # all synthetic CSVs
  python evaluate_ext.py --force    # recompute even if present
"""
import os, re, json, glob, argparse, warnings
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))

from data import load_dataset
import metrics_ext as MX

PAT = re.compile(r"(?P<ds>.+?)__(?P<method>.+?)__seed(?P<seed>\d+)\.csv$")
_BUNDLE = {}


def bundle(ds, seed):
    key = (ds, seed)
    if key not in _BUNDLE:
        _BUNDLE[key] = load_dataset(ds, seed=seed)
    return _BUNDLE[key]


def process(csv_path, force=False, log=print):
    m = PAT.search(os.path.basename(csv_path))
    if not m:
        return "skip-name"
    ds, method, seed = m["ds"], m["method"], int(m["seed"])
    jpath = os.path.join(HERE, "results", f"{ds}__{method}__seed{seed}.json")
    if not os.path.exists(jpath):
        return "no-json"
    try:
        rec = json.load(open(jpath))
    except Exception:
        return "bad-json"
    if "ext_c2st" in rec and not force:
        return "done"
    try:
        b = bundle(ds, seed)
    except Exception as e:
        log(f"  [bundle fail] {ds} seed{seed}: {e}")
        return "bundle-fail"
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        log(f"  [csv fail] {csv_path}: {e}")
        return "csv-fail"
    if len(df) < 10:
        return "too-few"
    cols = [c for c in b.df_train.columns if c in df.columns]
    ext = MX.evaluate_ext(b, df[cols])
    rec.update(ext)
    json.dump(rec, open(jpath, "w"), indent=2, default=float)
    log(f"  [ok] {ds:20s} {method:16s} seed{seed}  c2st={ext.get('ext_c2st'):.3f} "
        f"sdm={ext.get('ext_sdm_quality'):.3f} dcr={ext.get('ext_dcr'):.3f}")
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--glob", default=None)
    args = ap.parse_args()
    pats = ([args.glob] if args.glob else
            [os.path.join(HERE, "synthetic", "*.csv"),
             os.path.join(HERE, "synthetic", "baselines", "*.csv")])
    files = sorted(f for p in pats for f in glob.glob(p))
    print(f"enriching {len(files)} synthetic CSVs...")
    from collections import Counter
    c = Counter()
    for f in files:
        try:
            c[process(f, force=args.force)] += 1
        except Exception as e:
            c["error"] += 1
            print(f"  ERROR {f}: {e}")
    print("summary:", dict(c))


if __name__ == "__main__":
    main()
