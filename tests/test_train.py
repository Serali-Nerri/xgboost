import pytest
from sklearn.model_selection import KFold

from train import build_cv_splitter, get_cv_n_splits, select_final_n_estimators


def test_build_cv_splitter_uses_config_values():
    splitter = build_cv_splitter(
        {"n_splits": 4, "shuffle": True, "random_state": 123},
        "random",
    )

    assert isinstance(splitter, KFold)
    assert splitter.n_splits == 4
    assert splitter.shuffle is True
    assert splitter.random_state == 123


def test_build_cv_splitter_ignores_random_state_when_shuffle_disabled():
    splitter = build_cv_splitter(
        {"n_splits": 3, "shuffle": False, "random_state": 999},
        "random",
    )

    assert splitter.n_splits == 3
    assert splitter.shuffle is False
    assert splitter.random_state is None


def test_build_cv_splitter_rejects_non_boolean_shuffle():
    with pytest.raises(ValueError, match="config.cv.shuffle must be a boolean"):
        build_cv_splitter({"n_splits": 5, "shuffle": "yes"}, "random")


def test_get_cv_n_splits_rejects_deprecated_n_folds():
    with pytest.raises(ValueError, match="config.cv.n_folds is deprecated"):
        get_cv_n_splits({"n_folds": 5})


def test_get_cv_n_splits_returns_integer_value():
    assert get_cv_n_splits({"n_splits": 7}) == 7


def test_select_final_n_estimators_uses_median_best_iteration_plus_one():
    final_n_estimators, fold_best_iterations = select_final_n_estimators(
        {
            "fold_details": [
                {"best_iteration": 9},
                {"best_iteration": 11},
                {"best_iteration": None},
                {"best_iteration": 13},
            ]
        },
        fallback=200,
    )

    assert final_n_estimators == 12
    assert fold_best_iterations == [10, 12, 14]
