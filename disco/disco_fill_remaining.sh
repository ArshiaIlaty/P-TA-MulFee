#!/bin/bash
# Fill the remaining datasets for a 23/23 benchmark. RTF + TabSyn are already
# running (GPU-sharing OK). TAGAL (Qwen-4bit) needs ~6GB contiguous, so it OOMs
# at load under contention -> run it only AFTER RTF+TabSyn free the GPU. Then
# score all three (core+ext) and regenerate the tables.
cd /home/ec2-user/P-TA-MulFee/disco
L=logs/fill_remaining.log
: > "$L"

echo "waiting for RTF_FILL_DONE + TABSYN_FILL_DONE... $(date)" >> "$L"
for i in $(seq 1 600); do
  grep -q RTF_FILL_DONE logs/rtf_fill.log 2>/dev/null && grep -q TABSYN_FILL_DONE logs/tabsyn_fill.log 2>/dev/null && break
  sleep 60
done
echo "RTF+TabSyn done -> TAGAL 7 (GPU free) $(date)" >> "$L"

conda run -n discollm python run_tagal.py \
  --datasets obesity breast_cancer parkinsons bank_marketing surv_retinopathy surv_gbsg2 tab_qsar_biodeg \
  --seeds 0 >> "$L" 2>&1
echo "TAGAL fill done -> scoring all three $(date)" >> "$L"

for m in tagal realtabformer tabsyn; do
  conda run -n discodiff python evaluate_baselines.py --glob "synthetic/baselines/*__${m}__seed0.csv" >> "$L" 2>&1
  conda run -n discobase  python evaluate_ext.py        --glob "synthetic/baselines/*__${m}__seed0.csv" >> "$L" 2>&1
done

conda run -n discobase python make_comprehensive.py >> "$L" 2>&1
conda run -n discobase python make_paper_tables.py >> "$L" 2>&1
echo "FILL_ALL_DONE $(date)" >> "$L"
