#!/bin/bash
# Expand GReaT, TAEGAN, TAGAL to all 23 datasets (seed 0), resumable. Scores
# core+ext and refreshes the aggregate tables after each model so partial
# results are usable. TAEGAN first (fast), then GReaT (moderate), then TAGAL (slow).
cd /home/ec2-user/P-TA-MulFee/disco
ALL="heart_failure obesity diabetes surv_whas500 surv_pbc3 tab_maternal_risk breast_cancer liver_disorders parkinsons german_credit bank_marketing census_income surv_melanoma surv_pbc surv_retinopathy surv_gbsg2 surv_colon surv_grace surv_trace surv_framingham tab_blood_transfusion tab_qsar_biodeg tab_website_phishing"
L=logs/expand_llm.log

score() {  # $1 = tag
  DISCO_FORCE_RESCORE=1 conda run -n discodiff python evaluate_baselines.py --glob "synthetic/baselines/*__$1__seed0.csv" >> "$L" 2>&1
  conda run -n discobase python evaluate_ext.py --force --glob "synthetic/baselines/*__$1__seed0.csv" >> "$L" 2>&1
}
regen() {
  conda run -n discobase python make_comprehensive.py >> "$L" 2>&1
  conda run -n discobase python make_paper_tables.py >> "$L" 2>&1
}

echo "=== TAEGAN all23 start $(date) ===" >> "$L"
conda run -n discobase python run_taegan.py --datasets $ALL --seeds 0 >> "$L" 2>&1
score taegan; regen; echo "TAEGAN_PHASE_DONE $(date)" >> "$L"

echo "=== GReaT all23 start $(date) ===" >> "$L"
conda run -n discollm python baselines_llm.py --models great --datasets $ALL --seeds 0 >> "$L" 2>&1
score great; regen; echo "GREAT_PHASE_DONE $(date)" >> "$L"

echo "=== TAGAL all23 start $(date) ===" >> "$L"
conda run -n discollm python run_tagal.py --datasets $ALL --seeds 0 >> "$L" 2>&1
score tagal; regen; echo "TAGAL_PHASE_DONE $(date)" >> "$L"

echo "ALL_PHASES_DONE $(date)" >> "$L"
