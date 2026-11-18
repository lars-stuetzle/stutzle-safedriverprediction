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
| `src/data_utils.py` | Shared functions: load `train.csv`, recode `-1` as missing, stratified 80/20 split, feature groups |
| `Data/` | Place for `train.csv` (not stored in the repository, see below) |
| `results/` | Tables and figures written by the notebooks |
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

Notebook 01 runs in about 3–4 minutes on a laptop.
