import pandas as pd
import pytest
from sklearn.model_selection import KFold

from train import (
    build_cv_splitter,
    build_sample_weight_config,
    build_tail_residual_config,
    compute_sample_weight_series,
    get_cv_n_splits,
    select_final_n_estimators,
)


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


def test_build_sample_weight_config_and_compute_weights():
    config = build_sample_weight_config(
        {
            "sample_weighting": {
                "enabled": True,
                "feature_column": "Npl (kN)",
                "threshold_mode": "train_quantile",
                "quantile": 0.5,
                "below_weight": 1.0,
                "above_weight": 2.0,
            }
        }
    )
    features = pd.DataFrame({"Npl (kN)": [100.0, 200.0, 300.0, 400.0]})

    weights = compute_sample_weight_series(features, config)

    assert weights.tolist() == [1.0, 1.0, 2.0, 2.0]


def test_build_tail_residual_config_supports_logratio_and_soft_gating():
    config = build_tail_residual_config(
        {
            "tail_residual_correction": {
                "enabled": True,
                "feature_column": "Npl (kN)",
                "threshold_mode": "train_quantile",
                "quantile": 0.85,
                "correction_mode": "logratio",
                "append_global_prediction_features": True,
                "gating": {
                    "mode": "soft_linear",
                    "upper_threshold_mode": "train_quantile",
                    "upper_quantile": 0.95,
                },
                "params": {
                    "objective": "reg:squarederror",
                    "max_depth": 4,
                    "learning_rate": 0.1,
                    "n_estimators": 100,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 1,
                    "reg_alpha": 0.0,
                    "reg_lambda": 1.0,
                    "gamma": 0.0,
                    "random_state": 42,
                    "tree_method": "hist",
                    "device": "cpu",
                    "n_jobs": -1,
                },
            }
        }
    )

    assert config["correction_mode"] == "logratio"
    assert config["append_global_prediction_features"] is True
    assert config["gating"]["mode"] == "soft_linear"
    assert config["gating"]["quantile"] == pytest.approx(0.95)
