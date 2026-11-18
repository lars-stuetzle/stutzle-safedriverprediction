# stutzle-safedriverprediction

Project work for **B3155 – Project Work in Machine Learning and Data Mining** (University of Bologna,
A.Y. 2026/2027, Prof. Claudio Sartori).

Kaggle competition: [Porto Seguro's Safe Driver Prediction](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction)

## Problem

Porto Seguro, a Brazilian car insurer, wants to predict the probability that a driver will file an insurance
claim in the next year. Better estimates allow fairer prices: safe drivers pay less, risky drivers pay more.

- **Task:** binary classification (`target` = 1 if the driver filed a claim), with probability outputs.
- **Data:** 595,212 policies, 57 anonymised features in four groups (`ind`, `reg`, `car`, `calc`),
  binary / categorical / numeric; missing values coded as `-1`.
- **Main challenge:** strong class imbalance (only 3.64% claims) and many weak features.
- **Metric:** normalised Gini coefficient (= 2·AUC − 1), the official competition metric; the PR-AUC is
  reported as well.

As required by the course, only the Kaggle `train.csv` is used (the Kaggle `test.csv` has no target).
It is split into a stratified 80% training part (model selection with 5-fold cross-validation) and a 20%
held-out test part (used once, for the final evaluation).

## Objectives

1. Understand the data and its quality issues (exploratory data analysis).
2. Build preprocessing pipelines that follow the EDA findings, without leakage.
3. Compare several classifiers (logistic regression, decision tree, Naive Bayes, random forest,
   gradient boosting, neural network) against a dummy baseline with cross-validation.
4. Tune the best models and evaluate the final model once on the held-out test part
   (Gini, ROC / PR curves, lift, confidence intervals, feature importance).
5. Interpret the results from the insurer's point of view.

## Approach (CRISP-DM)

| Step | Notebook | Status |
|---|---|---|
| Data understanding (EDA) | `project/notebooks/01_eda.ipynb` | done |
| Data preparation, evaluation protocol, baseline | `project/notebooks/02_preparation.ipynb` | in progress |
| Modelling and model selection | `project/notebooks/03_modeling.ipynb` | to do |
| Final evaluation on the test part | `project/notebooks/04_evaluation.ipynb` | to do |
| Presentation (LaTeX Beamer) | `doc/` | to do |

## Results so far

**Exploratory data analysis** (notebook 01)
- Missing values are informative: depending on the column, the claim rate among missing rows is
  20–27% lower or about twice as high as among present rows, so "missing" is kept as information.
- No single feature is strong: the best univariate Gini is 0.15 (`ps_car_13`); the signal is spread
  over many weak features.
- The 20 `calc` features behave like random noise (none beats the best of 20 random noise features).
- Structural redundancy: `ps_ind_14` is exactly the sum of `ps_ind_10_bin`–`ps_ind_13_bin`;
  `ps_ind_06–09_bin` and `ps_ind_16–18_bin` are one-hot encoded hidden categorical variables.
- No duplicates across the train/test split, so there is no leakage risk from the split.

**Model comparison** (5-fold CV Gini on the training part)

| Model | CV Gini | CV PR-AUC |
|---|---|---|
| Dummy (constant claim rate) | 0.000 | 0.036 |
| _further models: to do_ | | |

## Repository structure

```
stutzle-safedriverprediction/
├── README.md              this file
├── project/               software and data
│   ├── README.md          description, prerequisites, installation and startup
│   ├── requirements.txt
│   ├── Data/              train.csv goes here (not in the repository, see Data/README.md)
│   ├── src/data_utils.py  loading, recoding and splitting, shared by all notebooks
│   ├── notebooks/         01_eda, 02_preparation, ... (+ PDF exports)
│   └── results/           tables and figures produced by the notebooks
└── doc/                   Beamer presentation (.tex and .pdf)
```

## References

- Kaggle competition: <https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction>
- C. Sartori, *Machine Learning and Data Mining*, lecture slides, University of Bologna, 2026.
- P. Chapman et al., *CRISP-DM 1.0: Step-by-step data mining guide*, SPSS, 2000.
- F. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825–2830, 2011.
- P.-N. Tan, M. Steinbach, V. Kumar, *Introduction to Data Mining*, Addison Wesley, 2006.
