import sys
import types

import numpy as np
import pandas as pd
import pytest

import src.model_trainer as model_trainer_module
from src.model_trainer import (
    ModelTrainer,
    _build_selection_objective_config,
    _calculate_selection_objective,
)


class RecordingSplitter:
    def __init__(self):
        self.split_called_with = None

    def split(self, X, y=None, groups=None):
        self.split_called_with = (X.copy(), None if y is None else y.copy(), groups)
        indices = np.arange(len(X))
        yield indices[:2], indices[2:]
        yield indices[2:], indices[:2]

    def get_n_splits(self, X=None, y=None, groups=None):
        return 2


class DummyTrial:
    number = 7

    def suggest_int(self, name, low, high):
        return low

    def suggest_float(self, name, low, high, log=False):
        return low


class DummyStudy:
    def __init__(self):
        self.trials = []
        self.best_params = {"max_depth": 3}
        self.best_value = 0.25
        self.best_trial = DummyTrial()

    def enqueue_trial(self, params):
        return None

    def optimize(self, objective, n_trials, timeout):
        trial = DummyTrial()
        self.best_value = objective(trial)
        self.trials = [trial]


class ConstantPsiRegressor:
    def __init__(self, **kwargs):
        self.best_iteration = 7
        self.constant_prediction = 1.5

    def fit(self, X_train, y_train, verbose=False, eval_set=None):
        return self

    def predict(self, X_val):
        return np.full(len(X_val), self.constant_prediction, dtype=float)


def test_cross_validate_restores_report_space_metrics_for_psi_target(monkeypatch):
    trainer = ModelTrainer(
        params={"device": "cpu", "n_jobs": -1, "random_state": 42},
        target_mode="psi_over_npl",
        target_transform_type=None,
        validation_size=0.0,
    )
    splitter = RecordingSplitter()
    X = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0],
            "Npl (kN)": [100.0, 200.0, 100.0, 200.0],
        }
    )
    y = pd.Series([1.0, 1.0, 2.0, 2.0])
    y_report = pd.Series([100.0, 200.0, 200.0, 400.0])

    monkeypatch.setattr(model_trainer_module.xgb, "XGBRegressor", ConstantPsiRegressor)

    results = trainer.cross_validate(X, y, y_report=y_report, cv=splitter)

    expected_rmse = float(np.sqrt(((50.0**2) + (100.0**2)) / 2.0))
    assert splitter.split_called_with is not None
    assert np.isclose(results["mean_cv_rmse"], expected_rmse)
    assert np.isclose(results["mean_cv_cov"], 0.0)
    assert all(detail["best_iteration"] == 7 for detail in results["fold_details"])


def test_optimize_hyperparameters_uses_report_target_and_splitter(monkeypatch, tmp_path):
    trainer = ModelTrainer(
        params={"device": "cpu", "n_jobs": -1, "random_state": 42},
        use_optuna=True,
        n_trials=1,
        optuna_timeout=1,
        target_mode="psi_over_npl",
        target_transform_type=None,
        validation_size=0.0,
    )
    X = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0],
            "Npl (kN)": [100.0, 200.0, 100.0, 200.0],
        }
    )
    y = pd.Series([1.0, 1.0, 2.0, 2.0])
    y_report = pd.Series([100.0, 200.0, 200.0, 400.0])
    splitter = RecordingSplitter()

    fake_optuna = types.ModuleType("optuna")
    setattr(fake_optuna, "create_study", lambda **kwargs: DummyStudy())
    fake_optuna.samplers = types.SimpleNamespace(
        TPESampler=lambda **kwargs: object()
    )
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)
    monkeypatch.setattr(model_trainer_module.xgb, "XGBRegressor", ConstantPsiRegressor)
    monkeypatch.setattr(model_trainer_module, "save_best_params", lambda **kwargs: None)

    results = trainer.optimize_hyperparameters(
        X,
        y,
        y_report=y_report,
        cv=splitter,
        n_trials=1,
        study_name="test-study",
        storage_url=f"sqlite:///{tmp_path / 'optuna.db'}",
        best_params_output_path=str(tmp_path / "best_params.json"),
    )

    assert splitter.split_called_with is not None
    split_X, split_y, split_groups = splitter.split_called_with
    assert split_X.equals(X)
    assert split_y is not None
    assert split_y.equals(y)
    assert split_groups is None
    assert results["best_params"] == {"max_depth": 3}
    assert results["target_mode"] == "psi_over_npl"


def test_selection_objective_matches_planned_formula():
    selection_objective = _build_selection_objective_config(
        {
            "metric_space": "original_nexp",
            "rmse_normalizer": "mean_actual",
            "cov_threshold": 0.10,
            "r2_threshold": 0.99,
            "cov_weight": 2.0,
            "r2_weight": 2.0,
        }
    )
    score = _calculate_selection_objective(
        {
            "rmse": 10.0,
            "r2": 0.98,
            "cov": 0.15,
            "mean_actual": 100.0,
        },
        selection_objective,
    )

    assert score == pytest.approx(3.1)


def test_selection_objective_rejects_unsupported_metric_space():
    with pytest.raises(ValueError, match="selection_objective.metric_space"):
        _build_selection_objective_config({"metric_space": "transformed"})
