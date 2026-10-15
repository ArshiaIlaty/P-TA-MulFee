#!/bin/bash
# "Strengthen" pass: fill the gaps identified after the first full study.
# Resumable (skips existing results). No pgrep wait-loops (avoids the earlier
# deadlock). Logs to logs/strengthen.log.
set -u
cd /home/ec2-user/P-TA-MulFee/disco
export HF_HOME=/home/ec2-user/P-TA-MulFee/disco/.hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

LOG "1/6 german_credit DISCO re-run (confusable-name alias fix), seeds 0 1 2"
python -u run_all.py --mode main --datasets german_credit --seeds 0 1 2

LOG "2/6 diabetes ablation (IFRS where severe imbalance matters), seed 0"
python -u run_all.py --mode ablation --datasets diabetes --seeds 0

LOG "3/6 DISCO bank/census seeds 1 2 (complete R=3)"
python -u run_all.py --mode main --datasets bank_marketing census_income --seeds 1 2

LOG "4/6 baselines ctgan/tvae/gaussian for bank/census seeds 1 2"
conda run -n discobase python baselines.py --datasets bank_marketing census_income \
  --methods gaussian_copula ctgan tvae --seeds 1 2 --epochs 60 || LOG "baseline bank/census seeds had failures"

LOG "5/6 score all baselines"
python -u evaluate_baselines.py

LOG "6/6 reassemble tables + stats + ablation + figures"
python -u stats.py
python -u assemble_tables.py
python -u assemble_ablation.py
python -u make_figs.py --copy-to-paper
LOG "STRENGTHEN COMPLETE"
