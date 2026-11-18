"""Preprocessing pipelines built from the EDA log (notebook 01, section 11).

Two families of models need different preparation:
- linear models (logistic regression; also kNN, SVM, MLP): every feature must be numeric, complete and on a
  comparable scale, so missing values are imputed, categories one-hot encoded and numbers standardized;
- tree models (decision tree, random forest, gradient boosting): they handle missing values and unscaled
  features natively, so the data is passed on almost unchanged.

All steps are wrapped in scikit-learn objects, so that they are fitted on the training folds only and no
information leaks from the validation fold into the preparation.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_utils import feature_groups

# value used for the "missing" category of categorical features (all real codes are >= 0)
MISSING_CATEGORY = -1
# categories with fewer rows than this (in the data the encoder is fitted on) are merged into one
# "infrequent" category: about 0.1% of the training part
MIN_CATEGORY_COUNT = 500
# exact linear combination of ps_ind_10_bin ... ps_ind_13_bin (EDA section 9)
REDUNDANT_FOR_LINEAR = ["ps_ind_14"]


def select_columns(columns, drop_calc=False, drop_columns=()):
    """Feature lists by type after removing the calc group (optional) and any extra columns."""
    groups = feature_groups(columns)
    excluded = set(drop_columns) | (set(groups["calc"]) if drop_calc else set())
    return {kind: [col for col in cols if col not in excluded]
            for kind, cols in groups.items() if kind != "calc"}


def make_linear_preprocessor(columns, drop_calc=False, min_category_count=MIN_CATEGORY_COUNT):
    """Preparation for linear and distance-based models.

    - numeric: median imputation plus a 0/1 "was missing" indicator (missingness is informative),
      then standardization (features are skewed and on different scales);
    - categorical: missing becomes its own category, then one-hot encoding; rare categories are grouped
      so that the model does not fit weights to a handful of rows; categories never seen during fitting
      are mapped to the infrequent group;
    - binary: no missing values, already 0/1, passed through unchanged;
    - ps_ind_14 is dropped (perfect multicollinearity), and optionally the calc features.
    """
    selected = select_columns(columns, drop_calc=drop_calc, drop_columns=REDUNDANT_FOR_LINEAR)

    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
        ("one_hot", OneHotEncoder(min_frequency=min_category_count,
                                  handle_unknown="infrequent_if_exist")),
    ])
    # the output is a sparse matrix (about 25% non-zero values): 3x less memory than a dense array,
    # which matters when the 5 cross-validation folds run in parallel
    return ColumnTransformer([
        ("numeric", numeric_steps, selected["numeric"]),
        ("categorical", categorical_steps, selected["categorical"]),
        ("binary", "passthrough", selected["binary"]),
    ], remainder="drop", sparse_threshold=0.3, verbose_feature_names_out=False)


def make_tree_preprocessor(columns, drop_calc=False):
    """Preparation for tree models: only column selection, NaN kept for the native missing-value support.

    The output stays a pandas DataFrame with the original column names, so that gradient boosting can be
    told which columns are categorical.
    """
    selected = select_columns(columns, drop_calc=drop_calc)
    kept_columns = selected["numeric"] + selected["categorical"] + selected["binary"]
    return ColumnTransformer([("keep", "passthrough", kept_columns)],
                             remainder="drop", verbose_feature_names_out=False
                             ).set_output(transform="pandas")


def categorical_columns(columns, drop_calc=False):
    """Names of the categorical columns that a tree preprocessor keeps (for native categorical support)."""
    return select_columns(columns, drop_calc=drop_calc)["categorical"]
