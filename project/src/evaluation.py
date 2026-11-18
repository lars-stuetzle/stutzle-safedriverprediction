"""Evaluation protocol shared by notebooks 02 and 03: stratified 5-fold CV, Gini and PR-AUC."""

from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data_utils import RANDOM_STATE

# the same 5 folds for every model, so that all results are directly comparable
CV_FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}


def gini_from_auc(auc):
    """Normalised Gini coefficient (the competition metric) from the ROC-AUC."""
    return 2 * auc - 1


def evaluate(model, X, y, name, n_jobs=-1):
    """Cross-validate a model and return Gini and PR-AUC (mean and std over the folds),
    the train Gini (to detect overfitting) and the mean fit time per fold.

    n_jobs: number of folds run in parallel; use 1 for models that are parallel themselves
    (e.g. a random forest with n_jobs=-1), to avoid oversubscribing the CPU and the memory.
    """
    scores = cross_validate(model, X, y, cv=CV_FOLDS, scoring=SCORING, n_jobs=n_jobs,
                            return_train_score=True)
    gini = gini_from_auc(scores["test_roc_auc"])
    return {
        "model": name,
        "gini_mean": gini.mean(),
        "gini_std": gini.std(),
        "pr_auc_mean": scores["test_pr_auc"].mean(),
        "pr_auc_std": scores["test_pr_auc"].std(),
        "train_gini": gini_from_auc(scores["train_roc_auc"]).mean(),
        "fit_time_s": scores["fit_time"].mean(),
    }


def search_summary(search, name):
    """Same summary as evaluate() for the best configuration of a fitted grid/randomized search.

    The search must have been run with scoring=SCORING, cv=CV_FOLDS, return_train_score=True and
    refit=False; the best configuration is the one with the highest mean ROC-AUC (= highest Gini).
    """
    results = search.cv_results_
    best = results["mean_test_roc_auc"].argmax()
    return {
        "model": name,
        "gini_mean": gini_from_auc(results["mean_test_roc_auc"][best]),
        "gini_std": 2 * results["std_test_roc_auc"][best],
        "pr_auc_mean": results["mean_test_pr_auc"][best],
        "pr_auc_std": results["std_test_pr_auc"][best],
        "train_gini": gini_from_auc(results["mean_train_roc_auc"][best]),
        "fit_time_s": results["mean_fit_time"][best],
    }


def best_params(search):
    """Hyperparameters of the configuration with the highest mean ROC-AUC."""
    return search.cv_results_["params"][search.cv_results_["mean_test_roc_auc"].argmax()]
