#!/bin/bash
# GPU is free (RW-kShape finished) -> retry the datasets that OOM'd under
# contention. RTF(5) + TAGAL(5) in parallel (Qwen 6GB + RTF 3GB fit in 15GB),
# then score all + regenerate.
cd /home/ec2-user/P-TA-MulFee/disco
L=logs/retry_stuck.log; : > "$L"
( conda run -n discoRTF python baselines_llm.py --models realtabformer \
    --datasets obesity parkinsons bank_marketing surv_framingham tab_qsar_biodeg --seeds 0 >> "$L" 2>&1
  echo "RTF_RETRY_DONE $(date)" >> "$L" ) &
( conda run -n discollm python run_tagal.py \
    --datasets obesity breast_cancer parkinsons bank_marketing tab_qsar_biodeg --seeds 0 >> "$L" 2>&1
  echo "TAGAL_RETRY_DONE $(date)" >> "$L" ) &
wait
echo "scoring + regen $(date)" >> "$L"
for m in tagal realtabformer tabsyn; do
  conda run -n discodiff python evaluate_baselines.py --glob "synthetic/baselines/*__${m}__seed0.csv" >> "$L" 2>&1
  conda run -n discobase  python evaluate_ext.py        --glob "synthetic/baselines/*__${m}__seed0.csv" >> "$L" 2>&1
done
conda run -n discobase python make_comprehensive.py >> "$L" 2>&1
conda run -n discobase python make_paper_tables.py >> "$L" 2>&1
echo "RETRY_ALL_DONE $(date)" >> "$L"
