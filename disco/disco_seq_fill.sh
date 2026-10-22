#!/bin/bash
# Fill remaining gaps SEQUENTIALLY (one GPU job at a time). RW-kShape holds ~6.7GB
# of the 15GB GPU, so parallel jobs OOM'd; sequential leaves ~8GB per job (fits
# TAGAL's 6GB Qwen). Never touches RW-kShape. Scores + regenerates at the end.
cd /home/ec2-user/P-TA-MulFee/disco
L=logs/seq_fill.log; : > "$L"

echo "RTF (5) $(date)" >> "$L"
conda run -n discoRTF python baselines_llm.py --models realtabformer \
  --datasets obesity parkinsons bank_marketing surv_framingham tab_qsar_biodeg --seeds 0 >> "$L" 2>&1
echo "RTF_SEQ_DONE $(date)" >> "$L"

echo "TabSyn (3) $(date)" >> "$L"
conda run -n discobase python run_tabsyn.py --datasets diabetes bank_marketing census_income >> "$L" 2>&1
echo "TABSYN_SEQ_DONE $(date)" >> "$L"

echo "TAGAL (5) $(date)" >> "$L"
conda run -n discollm python run_tagal.py \
  --datasets obesity breast_cancer parkinsons bank_marketing tab_qsar_biodeg --seeds 0 >> "$L" 2>&1
echo "TAGAL_SEQ_DONE $(date)" >> "$L"

echo "scoring + regen $(date)" >> "$L"
for m in tagal realtabformer tabsyn; do
  conda run -n discodiff python evaluate_baselines.py --glob "synthetic/baselines/*__${m}__seed0.csv" >> "$L" 2>&1
  conda run -n discobase  python evaluate_ext.py        --glob "synthetic/baselines/*__${m}__seed0.csv" >> "$L" 2>&1
done
conda run -n discobase python make_comprehensive.py >> "$L" 2>&1
conda run -n discobase python make_paper_tables.py >> "$L" 2>&1
echo "SEQ_ALL_DONE $(date)" >> "$L"
