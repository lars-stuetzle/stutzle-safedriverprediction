"""Shared data loading, recoding and splitting for all project notebooks.

Every notebook imports from this module, so that all of them work on exactly
the same training and test parts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Constants shared by all notebooks
RANDOM_STATE = 42
TEST_SIZE = 0.2
MISSING_CODE = -1  # missing values are coded as -1 in the competition data
DATA_PATH = Path(__file__).resolve().parent.parent / "Data" / "train.csv"


def load_data(path=DATA_PATH):
    """Read the raw Kaggle train.csv and recode the missing-value code -1 as NaN.

    The raw file is only read, never modified.
    """
    raw_data = pd.read_csv(path)
    return raw_data.replace(MISSING_CODE, np.nan)


def split_features_target(data):
    """Separate the 57 features from the target (the id column is dropped)."""
    features = data.drop(columns=["id", "target"])
    target = data["target"]
    return features, target


def train_test_parts(features, target):
    """Stratified 80/20 split, so that both parts keep the same claim rate.

    The test part is held out until the final evaluation (notebook 04).
    """
    return train_test_split(features, target, test_size=TEST_SIZE,
                            stratify=target, random_state=RANDOM_STATE)


def feature_groups(columns):
    """Group the feature names by type, using the suffixes from the competition.

    _bin = binary, _cat = categorical, no suffix = numeric (ordinal or continuous).
    The calc features are listed separately, since the EDA found them to be noise.
    """
    binary_cols = [col for col in columns if col.endswith("_bin")]
    categorical_cols = [col for col in columns if col.endswith("_cat")]
    numeric_cols = [col for col in columns
                    if col not in binary_cols and col not in categorical_cols]
    calc_cols = [col for col in columns if "_calc_" in col]
    return {"binary": binary_cols, "categorical": categorical_cols,
            "numeric": numeric_cols, "calc": calc_cols}
