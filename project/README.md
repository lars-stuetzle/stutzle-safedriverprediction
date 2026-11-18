# Porto Seguro Safe Driver Prediction – software and data

Prediction of the probability that a car insurance policy holder files a claim within one year, based on
the Kaggle competition
[Porto Seguro's Safe Driver Prediction](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction).
See the [main README](../README.md) for objectives and results.

## Contents

| Path | Content |
|---|---|
| `notebooks/01_eda.ipynb` | Exploratory data analysis and EDA log |
| `notebooks/02_preparation.ipynb` | Preprocessing pipelines, cross-validation protocol, dummy baseline |
| `notebooks/03_modeling.ipynb` | Comparison and tuning of the classifiers, choice of the final model |
| `notebooks/04_evaluation.ipynb` | Final evaluation on the held-out test part: confidence intervals, curves, lift, threshold, feature importance |
| `notebooks/*.pdf` | PDF export of each executed notebook (A4 landscape) |
| `src/data_utils.py` | Shared functions: load `train.csv`, recode `-1` as missing, stratified 80/20 split, feature groups |
| `src/preprocessing.py` | Preprocessing pipelines for linear models and for tree models |
| `src/evaluation.py` | Shared evaluation protocol: stratified 5-fold CV, Gini and PR-AUC |
| `src/models.py` | Builds the final model from the configuration saved by notebook 03 |
| `tools/export_pdf.py` | Exports the executed notebooks to landscape PDF files |
| `Data/` | Place for `train.csv` (not stored in the repository, see below) |
| `results/` | Tables, figures and the chosen model configuration written by the notebooks (e.g. `03_final_configuration.json`) |
| `requirements.txt` | Python packages with the tested versions |

Each notebook starts with a numbered workflow; its sections and code comments use the same numbers.

## Prerequisites

- Python 3.11 or newer (tested with Python 3.14.4)
- About 2 GB of free RAM (the training file has 595,212 rows × 59 columns)
- A Kaggle account to download the data

## Installation

```bash
git clone https://github.com/lars-stuetzle/stutzle-safedriverprediction.git
cd stutzle-safedriverprediction/project

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

## Data

The data file is too large for GitHub (about 110 MB), so it is not part of the repository.

1. Accept the competition rules on the
   [Kaggle data page](https://www.kaggle.com/competitions/porto-seguro-safe-driver-prediction/data).
2. Download `train.csv` (in the browser, or with the Kaggle API:
   `kaggle competitions download -c porto-seguro-safe-driver-prediction -f train.csv`).
3. Put it in `project/Data/train.csv` (unzip it if necessary).

Only `train.csv` is used. The Kaggle `test.csv` has no target and is therefore not part of this project.

## Startup

Open the notebooks in Jupyter, JupyterLab or VS Code **from the `notebooks/` folder** (the paths are relative
to it) and run them in numerical order, each with *Restart & Run All*:

```bash
cd notebooks
jupyter lab            # or open the folder in VS Code
```

To execute a notebook from the command line instead:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

On a laptop (8 cores, 16 GB RAM) the run times are:

| Notebook | Run time |
|---|---|
| 01 | about 3–4 minutes |
| 02 | about 2 minutes |
| 03 | about 80 minutes (hyperparameter searches over all models) |
| 04 | about 20 minutes |

The cross-validation uses all CPU cores (`n_jobs=-1`).

## PDF export

The PDF files next to the notebooks are exports of the executed notebooks in A4 landscape format. To recreate
them after running the notebooks (needs Microsoft Edge, Google Chrome or Chromium; no LaTeX):

```bash
python tools/export_pdf.py            # all notebooks in notebooks/
``` Notebook 03 reads `results/02_first_models_cv.csv`, so
run notebook 02 first; notebook 04 reads the results of notebook 03.
