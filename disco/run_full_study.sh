#!/bin/bash
# Autonomous, resumable driver for the entire DISCO-TAB study (memory-safe:
# batch 16, expandable segments). Each step skips already-produced outputs, so
# a restart re-enters safely. Shares the T4 with an external process, so only
# ~10 GB is available -> small batches.
set -u
cd /home/ec2-user/P-TA-MulFee/disco
export HF_HOME=/home/ec2-user/P-TA-MulFee/disco/.hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

LOG "starting study."

# Delete any DISCO result that parsed too few valid rows (<10) so it re-runs
# with the relaxed parser / memory-safe config.
python - <<'PY'
import json, glob, os
for p in glob.glob("results/*__disco__seed*.json"):
    try:
        d = json.load(open(p))
    except Exception:
        os.remove(p); continue
    if d.get("n_syn", 0) < 10:
        os.remove(p)
        syn = p.replace("results/", "synthetic/").replace(".json", ".csv")
        if os.path.exists(syn):
            os.remove(syn)
        print("cleared bad result:", os.path.basename(p))
PY

ALL="heart_failure liver_disorders parkinsons breast_cancer obesity diabetes german_credit bank_marketing census_income"
MEDG="heart_failure liver_disorders parkinsons breast_cancer obesity diabetes german_credit"

clean_bad() {
python - <<'PY'
import json, glob, os
for p in glob.glob("results/*__disco__seed*.json"):
    try:
        d = json.load(open(p))
    except Exception:
        os.remove(p); continue
    if d.get("n_syn", 0) < 10:
        os.remove(p)
        syn = p.replace("results/", "synthetic/").replace(".json", ".csv")
        if os.path.exists(syn): os.remove(syn)
        print("cleared bad:", os.path.basename(p))
PY
}

LOG "STEP 1: seed-0 matrix (all 9, resumable, up to 2 passes)"
for pass in 1 2; do
  clean_bad
  python -u run_all.py --mode main --datasets $ALL --seeds 0
done
LOG "STEP 1 done. assembling partial tables..."
python -u evaluate_baselines.py >/dev/null 2>&1 || true
python -u assemble_tables.py || true
python -u stats.py || true

LOG "STEP 2: DISCO seeds 1,2 for medical+german"
python -u run_all.py --mode main --datasets $MEDG --seeds 1 2

LOG "STEP 3: ablations on heart_failure + obesity (seed 0)"
python -u run_all.py --mode ablation --datasets heart_failure obesity --seeds 0

LOG "STEP 4: baseline ctgan/tvae seeds 1,2 (medical+german)"
conda run -n discobase python baselines.py --datasets $MEDG --methods ctgan tvae --seeds 1 2 --epochs 100 || LOG "baseline seed gen had failures (non-fatal)"

LOG "STEP 5: final evaluation + assembly"
python -u evaluate_baselines.py
python -u stats.py
python -u assemble_tables.py
python -u assemble_ablation.py
python -u make_figs.py --copy-to-paper
LOG "FULL STUDY COMPLETE"
