"""Construction of the final model chosen in notebook 03, from its saved configuration.

The model definitions are the same as in notebook 03 (sections 6, 8 and 10), so that notebook 04 (final
evaluation) and notebook 05 (Kaggle submission) refit exactly the model that was selected there.
"""

import json
from pathlib import Path

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_utils import RANDOM_STATE
from src.preprocessing import categorical_columns, make_linear_preprocessor, make_tree_preprocessor

CONFIGURATION_PATH = Path(__file__).resolve().parent.parent / "results" / "03_final_configuration.json"


def load_configuration(path=CONFIGURATION_PATH):
    """Hyperparameters of the final model, written by notebook 03 (section 11)."""
    with open(path) as file:
        return json.load(file)


def make_logistic_regression(columns, params, drop_calc=True):
    """Linear preparation + L2-regularized logistic regression (params e.g. {'model__C': 0.01})."""
    return Pipeline([
        ("prepare", make_linear_preprocessor(columns, drop_calc=drop_calc)),
        ("model", LogisticRegression(max_iter=2000)),
    ]).set_params(**params)


def make_random_forest(columns, params, drop_calc=True):
    """Random forest of 300 trees, each grown on 30% of the rows (params: max_features, min_samples_leaf)."""
    return Pipeline([
        ("prepare", make_tree_preprocessor(columns, drop_calc=drop_calc)),
        ("model", RandomForestClassifier(n_estimators=300, max_samples=0.3, n_jobs=-1,
                                         random_state=RANDOM_STATE, **params)),
    ])


def make_gradient_boosting(columns, params, drop_calc=True):
    """Histogram gradient boosting with early stopping on an internal 10% split (params: model__...)."""
    return Pipeline([
        ("prepare", make_tree_preprocessor(columns, drop_calc=drop_calc)),
        ("model", HistGradientBoostingClassifier(
            max_iter=1000, early_stopping=True, scoring="roc_auc", validation_fraction=0.1,
            n_iter_no_change=50, categorical_features=categorical_columns(columns, drop_calc=drop_calc),
            random_state=RANDOM_STATE)),
    ]).set_params(**params)


def make_final_model(columns, configuration):
    """Soft voting of the tuned logistic regression, random forest and gradient boosting."""
    drop_calc = configuration["drop_calc"]
    return VotingClassifier([
        ("logistic_regression",
         make_logistic_regression(columns, configuration["logistic_regression"], drop_calc)),
        ("random_forest", make_random_forest(columns, configuration["random_forest"], drop_calc)),
        ("gradient_boosting",
         make_gradient_boosting(columns, configuration["gradient_boosting"], drop_calc)),
    ], voting="soft")
