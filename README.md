# DISCO-TAB: Discriminator-Guided Control for Clinically Constrained Tabular Synthesis

DISCO-TAB fine-tunes a GPT-2 generator with a **hierarchy of discriminators**
(token / sentence / feature / row) coupled to reward-shaped PPO, plus
**inverse-frequency reward shaping (IFRS)** and post-hoc calibration, to generate
high-fidelity, utility-preserving synthetic tabular data for data-scarce,
imbalanced clinical settings. It builds on the P-TA framework (original P-TA/HELOC
code retained under version history).

This repository accompanies the IEEE JBHI submission *"DISCO-TAB: A Hierarchical
Reinforcement Learning Framework for Clinically Constrained Tabular Data
Synthesis."* The manuscript sources live in [`2026_JBHI___DISCO_Tab/`](2026_JBHI___DISCO_Tab);
the experiment code lives in [`disco/`](disco).

---

## 📊 Headline results

Benchmark: **14 generators × 23 datasets × 21 metrics**, three seeds, one shared
evaluation pipeline (clinical, financial, chemical, and web domains; SurvSet,
TabArena, and OpenML sources). Baselines span GAN (CTGAN, ADS-GAN, PATE-GAN,
TAEGAN), VAE (TVAE, RTVAE), diffusion/latent (TabDDPM, TabSyn), flow/marginal
(NFlow, Marginals), and language-model (GReaT, RealTabFormer, TAGAL) families.

| Dimension | Metric | DISCO-TAB | Notes |
|---|---|---|---|
| Downstream utility | TSTR macro-F1 | **0.64** | top-tier; statistically on par with the strongest baseline (TAEGAN) |
| Minority recovery | minority-class recall | **0.61** | best full-coverage generator (next best ≈ 0.50); holds across RF / LogReg / XGBoost |
| Statistical fidelity | correlation preservation | **0.92** | leads the full-coverage field |
| Statistical fidelity | Kolmogorov–Smirnov | **0.078** | strongest learned full-coverage generator |
| Disclosure risk | DCR ratio (syn / real–real) | **≈ 1.0** | near-calibrated; neither copying nor off-manifold |
| Disclosure risk | membership-inference AUROC | **0.52** | just above the 0.5 no-leakage value |

DISCO-TAB's distinction is **breadth and robustness**: it stays near the top on
utility, fidelity, and detectability at once, generates the minority class at
≈1.45× its real frequency (IFRS), and never collapses to a single class.

### Statistical testing

- **Friedman test** over the full-coverage generators rejects equal mean ranks.
- **Pairwise Wilcoxon signed-rank** tests, pairing unit = one per-dataset mean
  TSTR macro-F1 per dataset, with **Holm–Bonferroni** correction across the
  family. DISCO-TAB is statistically on par with the strongest baselines and
  significantly above the weaker GAN / VAE / flow / marginal ones.
- A **critical-difference (Nemenyi) diagram** summarizes the ranking, generated
  per metric.

Visualizations (generated into `disco/tables/comprehensive/figures/` and
`.../nemenyi/`):

| Figure | Script |
|---|---|
| Wilcoxon vs. each generator (Holm-corrected, IQR whiskers) | `disco/fig_wilcoxon.py` → `fig_wilcoxon_holm.pdf` |
| Pairwise tournament heatmap (median ΔF1, Holm-marked) | `disco/fig_stats_heatmap.py` → `fig_stats_heatmap.pdf` |
| Critical-difference diagrams, one per metric (21) | `disco/fig_nemenyi.py` → `nemenyi/` |
| Utility leaderboard / per-dataset margin / utility–fidelity / privacy | `disco/make_figures.py` |

---

## 📂 Repository layout

```
disco/                     DISCO-TAB experiment code (training, baselines, scoring, figures)
  run_experiment.py        DISCO-TAB: SFT + hierarchical-discriminator PPO + IFRS
  baselines_synthcity.py   synthcity baselines (CTGAN/ADS-GAN/PATE-GAN/TVAE/RTVAE/NFlow/TabDDPM)
  baselines_llm.py         GReaT / RealTabFormer      run_tabsyn.py  TabSyn
  run_taegan.py  run_tagal.py                         gen_marginal.py  independent-marginals sampler
  metrics.py  evaluate_baselines.py  evaluate_ext.py  eval_downstream.py   scoring pipeline
  make_comprehensive.py    aggregate per-run JSON → long.csv + per-metric tables
  make_paper_tables.py  make_downstream_ext_table.py  make_privacy_train_table.py
  make_wilcoxon.py  fig_nemenyi.py  fig_wilcoxon.py  fig_stats_heatmap.py   statistics
  tables/comprehensive/    long.csv, per-metric CSVs, figures/, nemenyi/
2026_JBHI___DISCO_Tab/     LaTeX manuscript (IEEE JBHI)
```

### Conda environments

| Env | Used for |
|---|---|
| `base` | DISCO-TAB training (`run_experiment.py`) |
| `discobase` | SDV/SDMetrics, TabSyn, TAEGAN, `evaluate_ext.py`, figures |
| `discodiff` | synthcity baselines, XGBoost, `evaluate_baselines.py`, `eval_downstream.py` |
| `discollm` | GReaT, TAGAL |
| `discoRTF` | RealTabFormer |

---

## 🔁 Reproduce

```bash
# 1. DISCO-TAB on one dataset
conda run -n base python disco/run_experiment.py --dataset heart_failure --seed 0

# 2. Baselines (example: synthcity zoo on all datasets, three seeds)
conda run -n discodiff python disco/baselines_synthcity.py --seeds 0 1 2

# 3. Score everything + aggregate
conda run -n discodiff python disco/evaluate_baselines.py --glob "disco/synthetic/baselines/*.csv"
conda run -n discobase python disco/make_comprehensive.py

# 4. Tables, figures, and statistics for the paper
conda run -n discobase python disco/make_figures.py
conda run -n discobase python disco/fig_nemenyi.py
python disco/make_wilcoxon.py
```

Per-run outputs are written to `disco/results/*.json` and
`disco/synthetic/`; everything in the paper is computed from the released
`disco/tables/comprehensive/long.csv` under one shared pipeline.

---

## 📄 Citation

If you use this code, please cite the DISCO-TAB paper (IEEE JBHI, under review).
See [`2026_JBHI___DISCO_Tab/`](2026_JBHI___DISCO_Tab) for the manuscript.
