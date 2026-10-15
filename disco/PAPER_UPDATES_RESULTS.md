# DISCO-TAB — Regenerated Results (real, reproducible)

_Generated from 107 scored runs. Single Tesla T4; GPT-2 backbone. All methods scored under one protocol (`disco/metrics.py`)._

> These are freshly computed numbers from the rebuilt pipeline. They replace the lost-server values. Baselines reproduced here: GaussCopula, CTGAN, TVAE. Paper baselines not regenerated (no surviving code/weights): TAEGAN, TabSyn, RealTabFormer, MostlyAI, TAGAL.

## Table I — Downstream utility, TSTR macro-F1 (mean over seeds)

| Dataset | GaussCopula | CTGAN | TVAE | DISCO-TAB | Δ(best base) |
|---|---|---|---|---|---|
| Heart Failure | 0.591 | 0.483 | 0.135 | 0.783 | **+0.192** |
| Breast Cancer | 0.912 | 0.295 | 0.934 | 0.930 | **-0.004** |
| Liver Disorders | 0.553 | 0.473 | 0.527 | 0.509 | **-0.044** |
| Parkinson's | 0.630 | 0.480 | 0.000 | 0.646 | **+0.016** |
| Obesity | 0.243 | 0.138 | 0.490 | 0.886 | **+0.396** |
| Diabetes | 0.485 | 0.836 | 0.875 | 0.843 | **-0.032** |
| German Credit | 0.496 | 0.448 | 0.137 | -- | -- |
| Bank Marketing | 0.469 | 0.593 | 0.654 | 0.579 | **-0.075** |
| Census Income | 0.459 | 0.714 | 0.740 | 0.769 | **+0.030** |

## Table I(b) — TSTR weighted-F1 (mean)

| Dataset | GaussCopula | CTGAN | TVAE | DISCO-TAB | Δ(best base) |
|---|---|---|---|---|---|
| Heart Failure | 0.679 | 0.559 | 0.185 | 0.799 | **+0.120** |
| Breast Cancer | 0.920 | 0.284 | 0.939 | 0.935 | **-0.003** |
| Liver Disorders | 0.553 | 0.472 | 0.529 | 0.511 | **-0.042** |
| Parkinson's | 0.729 | 0.632 | 0.000 | 0.732 | **+0.003** |
| Obesity | 0.250 | 0.138 | 0.493 | 0.888 | **+0.394** |
| Diabetes | 0.876 | 0.952 | 0.962 | 0.956 | **-0.006** |
| German Credit | 0.624 | 0.585 | 0.192 | -- | -- |
| Bank Marketing | 0.828 | 0.824 | 0.866 | 0.855 | **-0.011** |
| Census Income | 0.671 | 0.805 | 0.811 | 0.831 | **+0.021** |

## Table II — Statistical fidelity (DISCO-TAB, mean; lower JS/KL/Hell/KS better, Pearson higher better)

| Dataset | JS | KL | Hellinger | KS | Pearson |
|---|---|---|---|---|---|
| Heart Failure | 0.054 | 0.396 | 0.183 | 0.190 | 0.927 |
| Breast Cancer | 0.062 | 0.499 | 0.209 | 0.206 | 0.868 |
| Liver Disorders | 0.068 | 0.432 | 0.219 | 0.229 | 0.917 |
| Parkinson's | 0.082 | 1.233 | 0.238 | 0.184 | 0.845 |
| Obesity | 0.022 | 0.109 | 0.098 | 0.258 | 0.945 |
| Diabetes | 0.024 | 0.104 | 0.118 | 0.150 | 0.928 |
| German Credit | -- | -- | -- | -- | -- |
| Bank Marketing | 0.017 | 0.181 | 0.104 | 0.112 | 0.960 |
| Census Income | 0.063 | 0.481 | 0.204 | 0.477 | 0.942 |

## Table III — Empirical disclosure risk (DCR, DISCO-TAB) vs real–real baseline

| Dataset | Utility (F1) | DCR (median) | real–real DCR |
|---|---|---|---|
| Heart Failure | 0.783 | 0.731 | 0.801 |
| Breast Cancer | 0.930 | 0.550 | 0.465 |
| Liver Disorders | 0.509 | 0.412 | 0.380 |
| Parkinson's | 0.646 | 0.483 | 0.524 |
| Obesity | 0.886 | 0.277 | 0.279 |
| Diabetes | 0.843 | 0.053 | 0.071 |
| German Credit | -- | -- | -- |
| Bank Marketing | 0.579 | 0.152 | 0.210 |
| Census Income | 0.769 | 0.107 | 0.136 |

## FAITH decomposition (DISCO-TAB, mean)

| Dataset | FAITH | Factuality | Alignment | Integrity | Tracking |
|---|---|---|---|---|---|
| Heart Failure | 0.813 | 0.978 | 0.910 | 0.386 | 0.978 |
| Breast Cancer | 0.388 | 0.000 | 0.882 | 0.670 | 0.000 |
| Liver Disorders | 0.870 | 0.999 | 0.917 | 0.565 | 0.999 |
| Parkinson's | 0.298 | 0.000 | 0.832 | 0.361 | 0.000 |
| Obesity | 0.561 | 0.416 | 0.936 | 0.478 | 0.416 |
| Diabetes | 0.745 | 0.859 | 0.912 | 0.350 | 0.859 |
| German Credit | -- | -- | -- | -- | -- |
| Bank Marketing | 0.805 | 1.000 | 0.935 | 0.287 | 1.000 |
| Census Income | 0.364 | 0.076 | 0.928 | 0.374 | 0.076 |

## Feature-type JSD (mean per method; lower better)

| Method | Numerical JSD | Categorical JSD |
|---|---|---|
| GaussCopula | 0.059 | 0.000 |
| CTGAN | 0.100 | 0.011 |
| TVAE | 0.129 | 0.076 |
| DISCO-TAB | 0.054 | 0.014 |

## Ablation (relative TSTR F1 vs full system)

| Config | Rel. perf. | SD | mean F1 | datasets |
|---|---|---|---|---|
| Full DISCO-TAB | 100.0% | 5.5 | 0.834 | 6 |
| w/o Token Disc | 95.9% | 2.8 | 0.802 | 2 |
| w/o Sentence Disc | 97.3% | 1.9 | 0.813 | 2 |
| w/o Feature Disc | 90.6% | 7.6 | 0.76 | 2 |
| w/o Row Disc | 91.7% | 4.9 | 0.767 | 2 |
| w/o IFRS | 103.8% | 8.8 | 0.862 | 2 |
| Row-only (no hierarchy) | 96.3% | 3.6 | 0.805 | 2 |
| No RL (SFT only) | 102.3% | 3.7 | 0.852 | 2 |

## Significance (DISCO-TAB vs baselines; Wilcoxon signed-rank, Holm-corrected)

| Baseline | n | DISCO mean | base mean | Δ | wins | p | p(Holm) | sig@.05 |
|---|---|---|---|---|---|---|---|---|
| CTGAN | 9 | 0.716 | 0.496 | 0.220 | 8 | 0.0117 | 0.0352 | True |
| GaussCopula | 9 | 0.716 | 0.538 | 0.178 | 7 | 0.0391 | 0.0781 | False |
| TVAE | 9 | 0.716 | 0.499 | 0.217 | 5 | 0.2500 | 0.2500 | False |
