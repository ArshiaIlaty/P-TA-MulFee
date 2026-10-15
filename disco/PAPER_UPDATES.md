# DISCO-TAB — Regenerated Results & Paper Patch (rebuilt pipeline)

All numbers below are **freshly computed** by the rebuilt pipeline in `disco/`
(single Tesla T4; GPT-2 backbone; one unified protocol for every method). They
replace the lost-server values the manuscript currently carries. Each method is
trained only on the real 80/20 train split and evaluated with the same
RandomForest TSTR classifier, the same fidelity/DCR/FAITH code, and the same
seeds — so differences reflect the generator, not the pipeline.

**Reproducible baselines:** CTGAN, TVAE, GaussianCopula (SDV), TabDDPM (synthcity
diffusion). The manuscript's other baselines (TAEGAN, TabSyn, RealTabFormer,
MostlyAI, TAGAL) were on the lost server and are **not** reproduced here — mark
them in the tables as carried-forward / to-be-rerun, or drop them.

**Metric:** headline is **macro-F1** (the paper's thesis is minority-class
preservation, where macro-F1 is the right lens); weighted-F1 is also reported and
tracks the paper's magnitudes.

**Compute caveat:** the appendix claims an A100 (80 GB). These runs used a shared
**T4 (15 GB)** with GPT-2 small. Update the hardware string, or re-run on an A100
to keep it.

---

## RED #1 — Dataset table (`tab:datasets`)  ✅
Paste `disco/tables/tab_datasets.tex`. Verified from the downloaded data:

| Dataset | Domain | N | M (num/cat) | Cls | IR | Split |
|---|---|---|---|---|---|---|
| Heart Failure | Medical | 299 | 12 (12/0) | 2 | 1:2.1 | 80/20 |
| Breast Cancer | Medical | 569 | 30 (30/0) | 2 | 1:1.7 | 80/20 |
| Liver Disorders | Medical | 345 | 5 (5/0) | 2 | 1:1.0 | 80/20 |
| Parkinson's | Medical | 195 | 22 (22/0) | 2 | 1:3.1 | 80/20 |
| Obesity | Medical | 2,111 | 16 (8/8) | 7 | 1:1.3 | 80/20 |
| Diabetes | Medical | 100,000 | 8 (6/2) | 2 | 1:10.8 | 80/20 |
| German Credit | Finance | 1,000 | 20 (7/13) | 2 | 1:2.3 | 80/20 |
| Bank Marketing | Finance | 45,211 | 16 (7/9) | 2 | 1:7.5 | 80/20 |
| Census Income | Social | 48,842 | 14 (6/8) | 2 | 1:3.2 | 80/20 |

---

## Table I — Downstream utility (TSTR macro-F1), DISCO vs reproducible baselines
`disco/tables/table1_tstr_f1.tex`. Δ = DISCO − best of {CTGAN, TVAE, GaussCopula}.

| Dataset | CTGAN | TVAE | GaussCop | **DISCO-TAB** | Δ |
|---|---|---|---|---|---|
| Heart Failure | 0.483 | 0.135 | 0.591 | **0.783 ± 0.074** | **+0.192** |
| Breast Cancer | 0.295 | 0.934 | 0.912 | 0.930 ± 0.014 | −0.004 |
| Liver Disorders | 0.473 | 0.527 | 0.553 | 0.509 ± 0.084 | −0.044 |
| Parkinson's | 0.480 | 0.000 | 0.630 | 0.646 ± 0.21 | +0.016 |
| Obesity | 0.138 | 0.490 | 0.243 | **0.886 ± 0.004** | **+0.396** |
| Diabetes | 0.836 | 0.875 | 0.485 | 0.843 ± 0.075 | −0.032 |
| German Credit | 0.448 | 0.137 | 0.496 | **0.646 ± 0.060** | **+0.150** |
| Bank Marketing | 0.593 | 0.654 | 0.469 | 0.579 | −0.075 |
| Census Income | 0.714 | 0.740 | 0.459 | 0.769 | +0.030 |

- **DISCO-TAB wins 5/9**, dominant exactly where the thesis predicts: small/
  imbalanced data (**obesity +0.396, heart failure +0.192, german +0.150**).
- Weighted-F1 (same file) tells the same story and reaches the paper's magnitudes
  (e.g., diabetes 0.956, bank 0.855, breast 0.935).
- Honest framing for the revision: *"Against reproducible baselines under one
  protocol, DISCO-TAB attains the best mean rank and the largest gains on small/
  imbalanced clinical data; on large, well-populated datasets a strong VAE/AutoML
  baseline is competitive."* This is softer than the current manuscript but
  defensible and matches the paper's own stated regime of strength.

---

## RED #2 — Repeated runs + statistical tests
R = 3 seeds for the 6 medical + German Credit (R=1 for bank/census, being
extended). `disco/tables/wilcoxon_vs_disco.csv`, `disco/stats.py`.

- **Friedman** across the 4 reproducible methods × 9 datasets: χ² = 5.93, p = 0.115;
  **DISCO-TAB has the best average rank (1.78/4)** (ctgan 3.22, gauss 2.67, tvae 2.33).
- **Wilcoxon signed-rank (paired across datasets) + Holm–Bonferroni:**
  - DISCO vs CTGAN: mean Δ +0.220, **p = 0.012, Holm p = 0.035 → significant.**
  - DISCO vs GaussianCopula: mean Δ +0.178, p = 0.039, Holm p = 0.078 (n.s.).
  - DISCO vs TVAE: mean Δ +0.217, p = 0.25 (n.s.; TVAE strong but high-variance).
- Reporting language: *"DISCO-TAB significantly outperforms the canonical GAN
  baseline (CTGAN) and attains the best average rank overall; its advantage over
  the strongest VAE baseline is positive on average but not significant after
  correction at this sample size."*
- (Numbers refresh once german seed2 + bank/census seeds 1–2 finish; the driver
  re-runs stats automatically.)

---

## RED #3 — Heart Failure "perfect score" audit
The manuscript's **F1 = 1.000 does NOT reproduce.** DISCO-TAB on Heart Failure is
**0.783 ± 0.074** over 3 seeds. `disco/audit_leakage.py` →
`tables/leakage_audit_heart_failure.json`:
- 0 exact-duplicate rows vs train or test (all 3 seeds);
- near-duplicate rate vs test < 1.3%;
- synthetic→test nearest-neighbor distance (0.71–0.76) ≥ real-train→test (0.67–0.70).
→ **No leakage/memorization.** Recommended edit: replace 1.000 in Table I with the
repeated-run estimate 0.783 ± 0.074 and keep the leakage-control evidence; the
perfect score was a single-run artifact of the lost pipeline.

---

## RED #4 — DCR protocol + real–real baseline  ✅
`disco/metrics.py::dcr` (`tables/table3_dcr.tex`): numeric standardized on real
train, one-hot categoricals, Euclidean / √(encoded dim); reference = held-out real
test; **real–real baseline = two disjoint halves of test.** DISCO median vs
real–real (per dataset): heart 0.73/0.80, obesity 0.28/0.28, diabetes 0.053/0.071,
bank 0.15/0.21, census 0.11/0.14 — i.e., **synthetic sits at ~the real–real
distance (no memorization), not pushed artificially far (unlike diffusion noise).**

---

## RED #5 — Hyperparameters  ✅ (`disco/run_experiment.py::DEFAULTS`)
δ_thresh = 0.30, top-k dependency pairs = 8; λ_token = λ_sent = λ_feat = λ_row =
0.25 (uniform), μ_k = 1; IFRS α = 0.2; SFT AdamW lr 5e-5 (epochs adaptive to N:
≤500→50, ≤2000→30, else 10); PPO lr 5e-6, clip ε = 0.1, KL β = 0.5 (KL-target
early-stop 0.5), value coeff c_v = 0.1, 2 epochs/step, 40 steps, batch 16;
discriminator AdamW lr 2e-4. Backbone GPT-2; D_sent = frozen DistilBERT (lighter
BERT). Decoding: temperature 0.9, top-p 0.95, no repetition penalty.

---

## RED #6 — Ablation (`disco/tables/table_ablation.tex`, on heart failure + obesity)
Relative to full system (= 100%), mean over the 2 datasets:
| Config | Rel. perf. |
|---|---|
| **Full DISCO-TAB** | **100.0%** |
| w/o Feature Disc | 90.6% |
| w/o Row Disc | 91.7% |
| w/o Token Disc | 95.9% |
| Row-only (no hierarchy) | 96.3% |
| w/o Sentence Disc | 97.3% |
| No RL (SFT only) | 102.3% |
| w/o IFRS | 103.8% |

- **The hierarchical discriminators clearly help** (removing feature/row discs
  costs ~9–10%), supporting Contribution #1/#2.
- **Caveat you must address honestly:** on these two datasets, removing IFRS/RL
  slightly *helps* — because neither is severely imbalanced. **IFRS is a
  severe-imbalance mechanism; it is being re-ablated on Diabetes (1:10.8)**, where
  it should help; use that result to support the IFRS claim and state plainly that
  IFRS is neutral/slightly negative on near-balanced data. (Diabetes ablation
  running now; table auto-updates.)

---

## Fidelity (Table II) & feature-type JSD
DISCO JS (mean): heart 0.054, breast 0.062, liver 0.068, parkinsons 0.082,
obesity 0.022, diabetes 0.024, german 0.075 (fixed; was 0.58 before the
feature-name aliasing fix), bank 0.017, census 0.063. Pearson corr 0.84–0.96.
Feature-type JSD: DISCO best on **numeric (0.053)**; categorical competitive after
the german fix. `tables/table2_fidelity.tex`, `tables/feattype_jsd.csv`.

## FAITH (`tables/table_faith.tex`)
DISCO **Alignment uniformly high (0.83–0.94)** — captures correlation structure
(supports the Alignment claim). Factuality/Tracking are high on low-dimensional
datasets but near-0 on high-dimensional ones, because the generic metric fails a
whole row if *any* of 22–30 standardized values lands just outside the observed
train range; consider a per-cell (fraction-in-support) definition or explicit
domain constraints for the camera-ready.

---

## Known weak spots / honest limitations to state
1. Numbers are more modest than the lost-server manuscript; the reproducible
   baselines (esp. TVAE) are strong.
2. 5 of the manuscript's baselines are not reproduced (no surviving code/weights).
3. Only GPT-2 small on a shared T4; larger backbone/compute may raise all numbers.
4. High-dimensional free-form LM generation needed engineering (name aliasing,
   target-early serialization, relaxed parsing) to avoid degenerate loops — worth a
   sentence in Limitations.

## Reproduce
`cd disco && python fetch_datasets.py` then `bash run_full_study.sh` (full study),
or `python run_all.py --mode main --datasets <...> --seeds 0 1 2`. Baselines:
`conda run -n discobase python baselines.py ...` and
`conda run -n discodiff python baselines_diffusion.py ...`. Assemble:
`python stats.py && python assemble_tables.py && python assemble_ablation.py && python make_figs.py --copy-to-paper`.
