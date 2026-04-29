# Learning Nonlinear Test Assets: Dynamic Unsupervised Clustering of Firm Characteristics

Replication code — *Learning Nonlinear Test Assets: Dynamic Unsupervised Clustering of Firm Characteristics*.

---

## Overview

This repository contains the full replication package for the paper. The core methodology applies a temporally-regularised K-means clustering algorithm to 45 firm characteristics, producing 50 dynamic cluster portfolios that serve as test assets for asset pricing.

The baseline model uses **K = 50 clusters** and **λ = 1,000,000** (temporal regularisation strength).

---

## Repository Structure

```
├── notebooks/
│   ├── 00_setup_and_data_guide.ipynb          — Path verification and data checklist
│   ├── 01_Table1_Cluster_Returns.ipynb        — Table 1: Monthly cluster returns
│   ├── 02_Figure3_Grid_Search.ipynb           — Figure 3: H-L and SDF SR across K x lambda
│   ├── 03_Figures4_5_6_SDF_Diagnostics.ipynb  — Figures 4-6: SDF diagnostics
│   ├── 04_Tables2_3_SpanningTests.ipynb       — Tables 2-3: Factor spanning tests
│   ├── 05_Tables5_6_Figures9_10.ipynb         — Tables 5-6, Figures 9-10: Centroids
│   ├── 06_Figures11_12_13_ML.ipynb            — Figures 11-13: ML explainability
│   ├── 07_Table4_HML_DoubleSort.ipynb         — Table 4: HML double-sort
│   ├── 08_Figures7_8_Industry.ipynb           — Figures 7-8: Industry decomposition
│   ├── 09_Figure14_Table7_Stability.ipynb     — Figure 14, Table 7: Cluster stability
│   ├── 10_Figures15_16_MVE_Frontier.ipynb     — Figures 15-16: MVE frontiers
│   ├── 11_Table8_Macro_Regime.ipynb           — Table 8: Macro regime switches
│   ├── 12_Build_Centroid_Firm_Panel.ipynb     — Build centroid_chars.csv
│   └── 13_Run_All_Clustering_Models.ipynb     — Run all clustering models
│
├── utils/
│   ├── data_utils.py       — Data loading, path resolution, statistics helpers
│   └── portfolio_utils.py  — SDF estimation, MVE frontier, portfolio construction
│
├── data/
│   ├── Cluster_Ranking.csv — Official L01-L50 cluster rankings per lambda
│   └── README.md           — Instructions for placing data files
│
└── requirements.txt
```

---

## Notebooks at a Glance

| Notebook | Paper Output | Key Data Required |
|----------|-------------|-------------------|
| 00 | Setup verification | — |
| 01 | Table 1 | Cluster panels (K=50, λ=10⁶) |
| 02 | Figure 3 | Full K × λ grid |
| 03 | Figures 4, 5, 6 | Cluster panels (K=50, λ=10⁶) |
| 04 | Tables 2 & 3 | Cluster panels + `All_factors_JFE.csv` |
| 05 | Tables 5, 6 / Figures 9, 10 | `centroid_chars.csv` |
| 06 | Figures 11, 12, 13 | Entropy CSV from `entropy.ipynb` |
| 07 | Table 4 | `hml_panel_K_50_lambda_1000000.csv` |
| 08 | Figures 7 & 8 | `characteristics_clustering_results_K_50_lambda_1000000.csv` |
| 09 | Figure 14, Table 7 | `characteristics_clustering_results_K_50_lambda_1000000.csv` |
| 10 | Figures 15 & 16 | Cluster panels + factor files + Ken French portfolios |
| 11 | Table 8 | Cluster panels + `All_factors_JFE.csv` + Goyal-Welch macro |
| 12 | `centroid_chars.csv` | `characteristics_clustering_results_K_50_lambda_1000000.csv` |
| 13 | All clustering results | `Imputed__characteristics_winsorized99.csv` |

---

## Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/BhaskarGoswami1998/Learning-Nonlinear-Test-Assets-Dynamic-Unsupervised-Clustering-of-Firm-Characteristics.git
cd Learning-Nonlinear-Test-Assets-Dynamic-Unsupervised-Clustering-of-Firm-Characteristics
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place data files
See `data/README.md` for the full list of required files and where to place them.

### 4. Run notebooks in order
Start with `00_setup_and_data_guide.ipynb` to verify all paths are reachable, then run notebooks 01-11 in sequence.

---

## Key Model Parameters

| Parameter | Value |
|-----------|-------|
| Number of clusters K | 50 |
| Temporal regularisation λ | 1,000,000 |
| Return column | `r_t1_wmean` (1-month-ahead VW excess return) |
| Sample period | 1977:01 – 2020:12 |
| Firm characteristics | 45 (A2ME, AC, AT, ..., VAR) |
| Newey-West lags | 3 |
| Rolling SDF window | 72 months |

---

## Algorithm

The dynamic K-means clustering minimises at each month t:

$$\mathcal{L} = \sum_{i} \|x_i - \mu_{c_i}\|^2 + \lambda \cdot \|\mu - \mu_{\text{prev}}\|^2 \cdot \frac{N}{K}$$

where $\mu_{\text{prev}}$ are the cluster centroids from month $t-1$ and $N/K$ normalises for sample size relative to the number of clusters.

---

## Data Sources

| File | Source |
|------|--------|
| `All_factors_JFE.csv` | Authors (FF5, Q5, DHS, RPPCA factors) |
| `IPCA_factors.csv` | Kelly, Pruitt, Su (2019) |
| `RP-PCA Factors 2.csv` | Lettau & Pelger (2020) |
| `25_Portfolios_5x5.csv` | Ken French Data Library |
| `49_Industry_Portfolios.csv` | Ken French Data Library |
| `Portfolios_Formed_on_*.csv` | Ken French Data Library |
| `Macro_economic_PredictorData2021_Amit_goyal.xlsx` | Goyal & Welch (2008), updated 2021 |
| `Imputed__characteristics_winsorized99.csv` | Authors (CRSP/Compustat) |

---

## Requirements

```
numpy>=1.21
pandas>=1.3
matplotlib>=3.4
scikit-learn>=0.24
statsmodels>=0.13
scipy>=1.7
seaborn>=0.11
xgboost>=1.5
lightgbm>=3.2
openpyxl>=3.0
```

---

## Contact

Bhaskar Goswami — [@BhaskarGoswami1998](https://github.com/BhaskarGoswami1998)
