# DISCO-TAB — Regenerated Results (real, reproducible)

_Generated from 133 scored runs. Single Tesla T4; GPT-2 backbone. All methods scored under one protocol (`disco/metrics.py`)._

> These are freshly computed numbers from the rebuilt pipeline. They replace the lost-server values. Baselines reproduced here: GaussCopula, CTGAN, TVAE, TabDDPM. Paper baselines not regenerated (no surviving code/weights): TAEGAN, TabSyn, RealTabFormer, MostlyAI, TAGAL.

## Table I — Downstream utility, TSTR macro-F1 (mean over seeds)

| Dataset | GaussCopula | CTGAN | TVAE | TabDDPM | DISCO-TAB | Δ(best base) |
|---|---|---|---|---|---|---|
| Heart Failure | 0.591 | 0.483 | 0.135 | 0.406 | 0.783 | **+0.192** |
| Breast Cancer | 0.912 | 0.295 | 0.934 | 0.924 | 0.930 | **-0.004** |
| Liver Disorders | 0.553 | 0.473 | 0.527 | 0.592 | 0.509 | **-0.083** |
| Parkinson's | 0.630 | 0.480 | 0.000 | 0.928 | 0.646 | **-0.281** |
| Obesity | 0.243 | 0.138 | 0.490 | 0.795 | 0.886 | **+0.091** |
| Diabetes | 0.485 | 0.836 | 0.875 | -- | 0.843 | **-0.032** |
| German Credit | 0.496 | 0.448 | 0.137 | -- | 0.646 | **+0.150** |
| Bank Marketing | 0.469 | 0.604 | 0.673 | -- | 0.673 | **+0.001** |
| Census Income | 0.459 | 0.727 | 0.746 | -- | 0.765 | **+0.018** |

## Table I(b) — TSTR weighted-F1 (mean)

| Dataset | GaussCopula | CTGAN | TVAE | TabDDPM | DISCO-TAB | Δ(best base) |
|---|---|---|---|---|---|---|
| Heart Failure | 0.679 | 0.559 | 0.185 | 0.555 | 0.799 | **+0.120** |
| Breast Cancer | 0.920 | 0.284 | 0.939 | 0.929 | 0.935 | **-0.003** |
| Liver Disorders | 0.553 | 0.472 | 0.529 | 0.592 | 0.511 | **-0.081** |
| Parkinson's | 0.729 | 0.632 | 0.000 | 0.947 | 0.732 | **-0.215** |
| Obesity | 0.250 | 0.138 | 0.493 | 0.796 | 0.888 | **+0.092** |
| Diabetes | 0.876 | 0.952 | 0.962 | -- | 0.956 | **-0.006** |
| German Credit | 0.624 | 0.585 | 0.192 | -- | 0.668 | **+0.044** |
| Bank Marketing | 0.828 | 0.846 | 0.854 | -- | 0.865 | **+0.011** |
| Census Income | 0.671 | 0.812 | 0.818 | -- | 0.830 | **+0.012** |

## Table II — Statistical fidelity (DISCO-TAB, mean; lower JS/KL/Hell/KS better, Pearson higher better)

| Dataset | JS | KL | Hellinger | KS | Pearson |
|---|---|---|---|---|---|
| Heart Failure | 0.054 | 0.396 | 0.183 | 0.190 | 0.927 |
| Breast Cancer | 0.062 | 0.499 | 0.209 | 0.206 | 0.868 |
| Liver Disorders | 0.068 | 0.432 | 0.219 | 0.229 | 0.917 |
| Parkinson's | 0.082 | 1.233 | 0.238 | 0.184 | 0.845 |
| Obesity | 0.022 | 0.109 | 0.098 | 0.258 | 0.945 |
| Diabetes | 0.024 | 0.104 | 0.118 | 0.150 | 0.928 |
| German Credit | 0.074 | 1.517 | 0.169 | 0.118 | 0.937 |
| Bank Marketing | 0.016 | 0.180 | 0.102 | 0.104 | 0.952 |
| Census Income | 0.050 | 0.450 | 0.182 | 0.445 | 0.951 |

## Table III — Empirical disclosure risk (DCR, DISCO-TAB) vs real–real baseline

| Dataset | Utility (F1) | DCR (median) | real–real DCR |
|---|---|---|---|
| Heart Failure | 0.783 | 0.731 | 0.801 |
| Breast Cancer | 0.930 | 0.550 | 0.465 |
| Liver Disorders | 0.509 | 0.412 | 0.380 |
| Parkinson's | 0.646 | 0.483 | 0.524 |
| Obesity | 0.886 | 0.277 | 0.279 |
| Diabetes | 0.843 | 0.053 | 0.071 |
| German Credit | 0.646 | 0.407 | 0.423 |
| Bank Marketing | 0.673 | 0.160 | 0.209 |
| Census Income | 0.765 | 0.087 | 0.136 |

## FAITH decomposition (DISCO-TAB, mean)

| Dataset | FAITH | Factuality | Alignment | Integrity | Tracking |
|---|---|---|---|---|---|
| Heart Failure | 0.813 | 0.978 | 0.910 | 0.386 | 0.978 |
| Breast Cancer | 0.388 | 0.000 | 0.882 | 0.670 | 0.000 |
| Liver Disorders | 0.870 | 0.999 | 0.917 | 0.565 | 0.999 |
| Parkinson's | 0.298 | 0.000 | 0.832 | 0.361 | 0.000 |
| Obesity | 0.561 | 0.416 | 0.936 | 0.478 | 0.416 |
| Diabetes | 0.745 | 0.859 | 0.912 | 0.350 | 0.859 |
| German Credit | 0.333 | 0.000 | 0.942 | 0.388 | 0.000 |
| Bank Marketing | 0.810 | 1.000 | 0.932 | 0.309 | 1.000 |
| Census Income | 0.350 | 0.091 | 0.938 | 0.280 | 0.091 |

## Feature-type JSD (mean per method; lower better)

| Method | Numerical JSD | Categorical JSD |
|---|---|---|
| GaussCopula | 0.059 | 0.000 |
| CTGAN | 0.087 | 0.011 |
| TVAE | 0.118 | 0.066 |
| TabDDPM | 0.163 | 0.015 |
| DISCO-TAB | 0.049 | 0.030 |

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
| CTGAN | 9 | 0.742 | 0.498 | 0.244 | 9 | 0.0039 | 0.0117 | True |
| GaussCopula | 9 | 0.742 | 0.538 | 0.205 | 8 | 0.0195 | 0.0391 | True |
| TVAE | 9 | 0.742 | 0.502 | 0.240 | 6 | 0.1641 | 0.1641 | False |
