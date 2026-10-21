"""Apply class-conditional quantile/copula calibration to every DISCO synthetic
CSV in place (improvement 1, shipped across the whole benchmark)."""
import os, re, glob, warnings
import pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
from data import load_dataset
from calibrate import calibrate_numerics
PAT = re.compile(r"(.+?)__disco__seed(\d+)\.csv$")
n = 0
for p in sorted(glob.glob(os.path.join(HERE, "synthetic", "*__disco__seed*.csv"))):
    m = PAT.search(os.path.basename(p))
    if not m:
        continue
    ds, seed = m.group(1), int(m.group(2))
    try:
        b = load_dataset(ds, seed=seed)
    except Exception as e:
        print("skip", ds, seed, e); continue
    syn = pd.read_csv(p)
    cal = calibrate_numerics(syn, b.df_train, b.num_cols, target=b.target)
    cal.to_csv(p, index=False)
    n += 1
print(f"calibrated {n} DISCO CSVs in place")
