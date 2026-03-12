"""
Model Trainer Module for CFST XGBoost Pipeline

This module handles XGBoost model training, cross-validation, and hyperparameter optimization.
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any, Protocol, Union, cast
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from optuna.trial import Trial
import time
import json
import os
from pathlib import Path

from src.domain_features import (
    inverse_target_transform,
    normalize_target_mode,
    restore_report_target,
)
from src.utils.logger import get_logger
from src.utils.model_utils import load_best_params, save_best_params
from src.preprocessor import Preprocessor

logger = get_logger(__name__)

TUNABLE_PARAM_KEYS = (
    "max_depth",
    "learning_rate",
    "n_estimators",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "reg_alpha",
    "reg_lambda",
    "gamma",
)

OPTUNA_SEARCH_SPACE_VERSION = "centered_v4_stratified_consistent_cv"
VALID_METRIC_SPACES = {"transformed", "original"}
VALID_SELECTION_METRIC_SPACES = {"original_nexp", "report_target"}
VALID_RMSE_NORMALIZERS = {"mean_actual"}
DEFAULT_SELECTION_OBJECTIVE = {
    "metric_space": "original_nexp",
    "rmse_normalizer": "mean_actual",
    "cov_threshold": 0.10,
    "r2_threshold": 0.99,
    "cov_weight": 2.0,
    "r2_weight": 2.0,
}


def _normalize_metric_space(metric_space: Optional[str]) -> str:
    """Normalize metric space names used by CV and Optuna scoring."""
    normalized = (metric_space or "transformed").strip().lower()
    if normalized not in VALID_METRIC_SPACES:
        raise ValueError(
            f"Unsupported metric space '{metric_space}'. "
            f"Expected one of {sorted(VALID_METRIC_SPACES)}."
        )
    return normalized


def _negative_rmse_score(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    metric_space: str = "transformed",
    target_transform_type: Optional[str] = None,
) -> float:
    """Return negative RMSE so sklearn/Optuna can maximize the scorer."""
    normalized_space = _normalize_metric_space(metric_space)
    y_true_array = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_array = np.asarray(y_pred, dtype=float).reshape(-1)

    if normalized_space == "original":
        y_true_array = inverse_target_transform(y_true_array, target_transform_type)
        y_pred_array = inverse_target_transform(y_pred_array, target_transform_type)

    return -float(np.sqrt(mean_squared_error(y_true_array, y_pred_array)))


def _build_selection_objective_config(
    raw_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    config = DEFAULT_SELECTION_OBJECTIVE.copy()
    if raw_config:
        config.update(raw_config)
    metric_space = str(config.get("metric_space", "original_nexp")).strip().lower()
    if metric_space not in VALID_SELECTION_METRIC_SPACES:
        raise ValueError(
            "selection_objective.metric_space must be one of "
            f"{sorted(VALID_SELECTION_METRIC_SPACES)}"
        )
    rmse_normalizer = str(config.get("rmse_normalizer", "mean_actual")).strip().lower()
    if rmse_normalizer not in VALID_RMSE_NORMALIZERS:
        raise ValueError(
            "selection_objective.rmse_normalizer must be one of "
            f"{sorted(VALID_RMSE_NORMALIZERS)}"
        )
    config["metric_space"] = metric_space
    config["rmse_normalizer"] = rmse_normalizer
    return config


def _calculate_cov_statistics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, Optional[float]]:
    y_true_array = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_array = np.asarray(y_pred, dtype=float).reshape(-1)
    ratios = y_pred_array / y_true_array
    valid_mask = np.isfinite(ratios) & (y_true_array != 0)
    valid_ratios = ratios[valid_mask]
    if len(valid_ratios) == 0:
        return {"cov": None, "mean_ratio": None, "std_ratio": None}

    mean_ratio = float(np.mean(valid_ratios))
    std_ratio = float(np.std(valid_ratios, ddof=1)) if len(valid_ratios) > 1 else 0.0
    cov = std_ratio / mean_ratio if mean_ratio != 0 else None
    return {
        "cov": float(cov) if cov is not None else None,
        "mean_ratio": mean_ratio,
        "std_ratio": std_ratio,
    }


def _calculate_regression_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
) -> Dict[str, Optional[float]]:
    y_true_array = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_array = np.asarray(y_pred, dtype=float).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_true_array, y_pred_array)))
    r2 = float(r2_score(y_true_array, y_pred_array))
    mae = float(np.mean(np.abs(y_true_array - y_pred_array)))
    mean_actual = float(np.mean(y_true_array))
    cov_stats = _calculate_cov_statistics(y_true_array, y_pred_array)
    return {
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
        "mean_actual": mean_actual,
        "cov": cov_stats["cov"],
        "mean_ratio": cov_stats["mean_ratio"],
        "std_ratio": cov_stats["std_ratio"],
    }


def _calculate_selection_objective(
    metrics: Dict[str, Optional[float]],
    selection_objective: Dict[str, Any],
) -> float:
    rmse = metrics.get("rmse")
    r2 = metrics.get("r2")
    cov = metrics.get("cov")
    mean_actual = metrics.get("mean_actual")

    if rmse is None or r2 is None or mean_actual is None or cov is None:
        return 1e9

    if selection_objective.get("metric_space") not in VALID_SELECTION_METRIC_SPACES:
        raise ValueError(
            "selection_objective.metric_space must be set to a supported reported-target space"
        )

    rmse_normalizer = str(selection_objective.get("rmse_normalizer", "mean_actual")).strip().lower()
    if rmse_normalizer == "mean_actual":
        scale = max(abs(float(mean_actual)), 1e-9)
    else:
        raise ValueError(
            "selection_objective.rmse_normalizer must be set to a supported normalizer"
        )

    nrmse = float(rmse) / scale
    cov_threshold = float(selection_objective["cov_threshold"])
    r2_threshold = float(selection_objective["r2_threshold"])
    cov_weight = float(selection_objective["cov_weight"])
    r2_weight = float(selection_objective["r2_weight"])

    cov_penalty = cov_weight * max(0.0, float(cov) - cov_threshold) / max(cov_threshold, 1e-9)
    r2_penalty = r2_weight * max(0.0, r2_threshold - float(r2)) / max(1.0 - r2_threshold, 1e-9)
    return float(nrmse + cov_penalty + r2_penalty)


class CrossValidatorProtocol(Protocol):
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups: Any = None) -> Any:
        ...

    def get_n_splits(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        groups: Any = None,
    ) -> int:
        ...


CrossValidatorLike = Union[int, CrossValidatorProtocol]


class ModelTrainer:
    """
    Model trainer for CFST XGBoost pipeline.

    Handles XGBoost model training, cross-validation, and optional hyperparameter optimization.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        use_optuna: bool = False,
        n_trials: int = 100,
        optuna_timeout: int = 3600,
        best_params_path: str = "logs/best_params.json",
        expected_context_hash: Optional[str] = None,
        optuna_metric_space: str = "transformed",
        target_transform_type: Optional[str] = None,
        target_mode: str = "raw",
        columns_to_drop: Optional[List[str]] = None,
        validation_size: float = 0.0,
        early_stopping_rounds: Optional[int] = None,
        eval_metric: Optional[str] = None,
        selection_objective: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ModelTrainer.

        Args:
            params: XGBoost parameters (defaults to reasonable values if None)
            use_optuna: Whether to use Optuna for hyperparameter optimization
            n_trials: Number of Optuna trials for hyperparameter optimization
            optuna_timeout: Timeout for Optuna optimization in seconds
            best_params_path: Path to persisted best parameters JSON
            expected_context_hash: Expected training context hash for safe parameter reuse
            optuna_metric_space: RMSE scoring space for Optuna ('transformed' or 'original')
            target_transform_type: Target transform applied before model fitting
            target_mode: Modeled target definition ('raw' or 'psi_over_npl')
            columns_to_drop: Columns removed by the preprocessor
            validation_size: Fold-internal validation share used for early stopping
            early_stopping_rounds: Early stopping rounds for fold/internal validation
            eval_metric: XGBoost eval metric used when validation is available
            selection_objective: Composite CV/Optuna selection objective in report space
        """
        self.params = params or self._get_default_params()
        self.use_optuna = use_optuna
        self.n_trials = n_trials
        self.optuna_timeout = optuna_timeout
        self.best_params_path = best_params_path
        self.expected_context_hash = expected_context_hash
        self.optuna_metric_space = _normalize_metric_space(optuna_metric_space)
        self.target_transform_type = target_transform_type
        self.target_mode = normalize_target_mode(target_mode)
        self.columns_to_drop = list(columns_to_drop or [])
        self.validation_size = float(validation_size or 0.0)
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.selection_objective = _build_selection_objective_config(selection_objective)
        self.loaded_best_params = False
        self.model: Optional[xgb.XGBRegressor] = None
        self.training_history = []

        # Auto-load best parameters if use_optuna is False
        if not self.use_optuna:
            loaded_params = load_best_params(
                input_path=self.best_params_path,
                expected_context_hash=self.expected_context_hash,
            )
            if loaded_params is not None:
                self.params.update(loaded_params)
                self.loaded_best_params = True
                logger.info("Using loaded best parameters for training")

        logger.info(f"ModelTrainer initialized with use_optuna={use_optuna}")
        logger.info(
            "ModelTrainer metric setup: "
            f"optuna_metric_space={self.optuna_metric_space}, "
            f"target_transform_type={self.target_transform_type or 'none'}, "
            f"target_mode={self.target_mode}, "
            f"validation_size={self.validation_size}, "
            f"early_stopping_rounds={self.early_stopping_rounds}, "
            f"eval_metric={self.eval_metric}, "
            f"selection_objective={json.dumps(self.selection_objective, sort_keys=True)}"
        )
        logger.debug(f"Initial parameters: {json.dumps(self.params, indent=2)}")

    @staticmethod
    def _get_default_params() -> Dict[str, Any]:
        """
        Get default XGBoost parameters.

        Returns:
            Dictionary of default parameters
        """
        return {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "tree_method": "hist",
            "device": "cpu",
            "n_jobs": -1,  # Use all CPU cores
        }

    def get_optuna_center_point(self) -> Dict[str, Any]:
        """
        Return the current Optuna center point derived from model params.
        """
        return {
            "max_depth": int(self.params.get("max_depth", 5)),
            "learning_rate": float(self.params.get("learning_rate", 0.05)),
            "n_estimators": int(self.params.get("n_estimators", 1200)),
            "subsample": float(self.params.get("subsample", 0.8)),
            "colsample_bytree": float(self.params.get("colsample_bytree", 0.75)),
            "min_child_weight": int(self.params.get("min_child_weight", 10)),
            "reg_alpha": float(self.params.get("reg_alpha", 0.5)),
            "reg_lambda": float(self.params.get("reg_lambda", 2.0)),
            "gamma": float(self.params.get("gamma", 0.05)),
        }

    def get_optuna_search_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Build an Optuna search space centered on config.model.params.
        """
        center = self.get_optuna_center_point()

        search_space: Dict[str, Dict[str, Any]] = {
            "max_depth": {
                "kind": "int",
                "low": max(3, center["max_depth"] - 1),
                "high": min(8, center["max_depth"] + 2),
            },
            "learning_rate": {
                "kind": "float",
                "low": max(0.01, center["learning_rate"] / 2.5),
                "high": min(0.12, max(center["learning_rate"] * 1.8, 0.02)),
                "log": True,
            },
            "n_estimators": {
                "kind": "int",
                "low": max(800, int(round(center["n_estimators"] * 0.6))),
                "high": min(4500, int(round(center["n_estimators"] * 2.2))),
            },
            "subsample": {
                "kind": "float",
                "low": max(0.55, center["subsample"] - 0.18),
                "high": min(0.98, center["subsample"] + 0.08),
            },
            "colsample_bytree": {
                "kind": "float",
                "low": max(0.4, center["colsample_bytree"] - 0.18),
                "high": min(0.95, center["colsample_bytree"] + 0.12),
            },
            "min_child_weight": {
                "kind": "int",
                "low": max(4, int(round(center["min_child_weight"] * 0.6))),
                "high": min(40, int(round(center["min_child_weight"] * 2.0))),
            },
            "reg_alpha": {
                "kind": "float",
                "low": max(1e-3, center["reg_alpha"] / 20.0),
                "high": min(20.0, max(center["reg_alpha"] * 20.0, 1e-2)),
                "log": True,
            },
            "reg_lambda": {
                "kind": "float",
                "low": max(1e-2, center["reg_lambda"] / 15.0),
                "high": min(50.0, max(center["reg_lambda"] * 15.0, 0.1)),
                "log": True,
            },
            "gamma": {
                "kind": "float",
                "low": max(1e-4, center["gamma"] / 30.0),
                "high": min(5.0, max(center["gamma"] * 20.0, 1e-3)),
                "log": True,
            },
        }

        return search_space

    def build_optuna_trial_params(self, trial: Trial) -> Dict[str, Any]:
        """
        Sample one Optuna trial from the centered search space.
        """
        search_space = self.get_optuna_search_space()
        tuned_params: Dict[str, Any] = {}

        for name, spec in search_space.items():
            if spec["kind"] == "int":
                tuned_params[name] = trial.suggest_int(
                    name, int(spec["low"]), int(spec["high"])
                )
            else:
                tuned_params[name] = trial.suggest_float(
                    name,
                    float(spec["low"]),
                    float(spec["high"]),
                    log=bool(spec.get("log", False)),
                )

        return {
            "objective": self.params.get("objective", "reg:squarederror"),
            **tuned_params,
            "random_state": self.params.get("random_state", 42),
            "tree_method": self.params.get("tree_method", "hist"),
            "device": self.params.get("device", "cpu"),
            "n_jobs": self.params.get("n_jobs", -1),
        }

    def _fit_model(
        self,
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight_train: Optional[Union[pd.Series, np.ndarray]] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight_val: Optional[Union[pd.Series, np.ndarray]] = None,
        early_stopping_rounds: Optional[int] = None,
        eval_metric: Optional[str] = None,
    ) -> xgb.XGBRegressor:
        """
        Fit one XGBoost model using the same validation/early-stopping path as final training.
        """
        model_params = params.copy()
        fit_kwargs: Dict[str, Any] = {"verbose": False}
        active_early_stopping_rounds = (
            early_stopping_rounds
            if early_stopping_rounds is not None
            else self.early_stopping_rounds
        )
        active_eval_metric = eval_metric if eval_metric is not None else self.eval_metric

        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            if sample_weight_val is not None:
                fit_kwargs["sample_weight_eval_set"] = [
                    np.asarray(sample_weight_val, dtype=float).reshape(-1)
                ]
            if active_early_stopping_rounds is not None:
                model_params["early_stopping_rounds"] = active_early_stopping_rounds
        else:
            model_params.pop("early_stopping_rounds", None)

        if active_eval_metric is not None:
            model_params["eval_metric"] = active_eval_metric
        else:
            model_params.pop("eval_metric", None)

        model = xgb.XGBRegressor(**model_params)
        if sample_weight_train is not None:
            fit_kwargs["sample_weight"] = np.asarray(
                sample_weight_train, dtype=float
            ).reshape(-1)
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    def _split_train_validation(
        self,
        X_train_fold: pd.DataFrame,
        y_train_fold: pd.Series,
        fold_index: int,
        stratify_labels: Optional[pd.Series] = None,
        sample_weight_fold: Optional[pd.Series] = None,
    ) -> Tuple[
        pd.DataFrame,
        Optional[pd.DataFrame],
        pd.Series,
        Optional[pd.Series],
        Optional[pd.Series],
        Optional[pd.Series],
    ]:
        """
        Split a fold into fit/validation subsets using the same strategy as final training.
        """
        if self.validation_size <= 0:
            return X_train_fold, None, y_train_fold, None, sample_weight_fold, None

        split_kwargs: Dict[str, Any] = {
            "test_size": self.validation_size,
            "random_state": int(self.params.get("random_state", 42)) + int(fold_index),
        }
        if stratify_labels is not None and stratify_labels.nunique() > 1:
            split_kwargs["stratify"] = stratify_labels

        try:
            if sample_weight_fold is not None:
                split_result = cast(
                    Tuple[
                        pd.DataFrame,
                        pd.DataFrame,
                        pd.Series,
                        pd.Series,
                        pd.Series,
                        pd.Series,
                    ],
                    train_test_split(
                        X_train_fold,
                        y_train_fold,
                        sample_weight_fold,
                        **split_kwargs,
                    ),
                )
            else:
                split_result = cast(
                    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
                    train_test_split(
                        X_train_fold,
                        y_train_fold,
                        **split_kwargs,
                    ),
                )
        except ValueError as exc:
            logger.warning(
                "Fold %s stratified validation split failed (%s); retrying without stratification",
                fold_index + 1,
                exc,
            )
            split_kwargs.pop("stratify", None)
            if sample_weight_fold is not None:
                split_result = cast(
                    Tuple[
                        pd.DataFrame,
                        pd.DataFrame,
                        pd.Series,
                        pd.Series,
                        pd.Series,
                        pd.Series,
                    ],
                    train_test_split(
                        X_train_fold,
                        y_train_fold,
                        sample_weight_fold,
                        **split_kwargs,
                    ),
                )
            else:
                split_result = cast(
                    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
                    train_test_split(
                        X_train_fold,
                        y_train_fold,
                        **split_kwargs,
                    ),
                )

        if sample_weight_fold is not None:
            return cast(
                Tuple[
                    pd.DataFrame,
                    Optional[pd.DataFrame],
                    pd.Series,
                    Optional[pd.Series],
                    Optional[pd.Series],
                    Optional[pd.Series],
                ],
                split_result,
            )

        X_fit_raw, X_val_raw, y_fit, y_val = cast(
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
            split_result,
        )
        return X_fit_raw, X_val_raw, y_fit, y_val, None, None

    def _score_with_consistent_cv(
        self,
        params: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        cv: CrossValidatorLike,
        metric_space: Optional[str] = None,
        target_transform_type: Optional[str] = None,
        stratify_labels: Optional[pd.Series] = None,
        y_report: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Run fold-by-fold evaluation using the same fold-internal validation path as final training.
        """
        if isinstance(cv, int) and not isinstance(cv, bool):
            splitter = KFold(
                n_splits=cv,
                shuffle=True,
                random_state=self.params.get("random_state", 42),
            )
        else:
            splitter = cv

        n_folds = splitter.get_n_splits(X, y, None)
        scoring_space = _normalize_metric_space(metric_space or self.optuna_metric_space)
        active_target_transform = (
            target_transform_type
            if target_transform_type is not None
            else self.target_transform_type
        )
        cv_target = stratify_labels if stratify_labels is not None else y

        fold_scores: List[float] = []
        fold_rmse: List[float] = []
        fold_r2: List[float] = []
        fold_cov: List[float] = []
        fold_details: List[Dict[str, Any]] = []
        start_time = time.time()

        for fold_index, (train_idx, test_idx) in enumerate(splitter.split(X, cv_target)):
            X_train_fold_raw = X.iloc[train_idx].copy()
            X_test_fold_raw = X.iloc[test_idx].copy()
            y_train_fold = y.iloc[train_idx].copy()
            y_test_fold = y.iloc[test_idx].copy()
            y_test_report = (
                y_report.iloc[test_idx].copy()
                if y_report is not None
                else None
            )
            fold_strata = (
                stratify_labels.iloc[train_idx].copy()
                if stratify_labels is not None
                else None
            )
            fold_sample_weight = (
                sample_weight.iloc[train_idx].copy()
                if sample_weight is not None
                else None
            )

            (
                X_fit_raw,
                X_val_raw,
                y_fit,
                y_val,
                sample_weight_fit,
                sample_weight_val,
            ) = self._split_train_validation(
                X_train_fold_raw,
                y_train_fold,
                fold_index,
                stratify_labels=fold_strata,
                sample_weight_fold=fold_sample_weight,
            )

            preprocessor = Preprocessor(columns_to_drop=self.columns_to_drop)
            X_fit = preprocessor.fit_transform(X_fit_raw)
            X_test = preprocessor.transform(X_test_fold_raw)
            X_val = preprocessor.transform(X_val_raw) if X_val_raw is not None else None

            fold_model = self._fit_model(
                params=params,
                X_train=X_fit,
                y_train=y_fit,
                sample_weight_train=sample_weight_fit,
                X_val=X_val,
                y_val=y_val,
                sample_weight_val=sample_weight_val,
            )
            y_pred = fold_model.predict(X_test)
            y_pred_report = restore_report_target(
                y_pred,
                target_mode=self.target_mode,
                target_transform_type=active_target_transform,
                reference_features=X_test_fold_raw,
            )
            if y_test_report is None:
                y_true_report = restore_report_target(
                    y_test_fold,
                    target_mode=self.target_mode,
                    target_transform_type=active_target_transform,
                    reference_features=X_test_fold_raw,
                )
            else:
                y_true_report = y_test_report.to_numpy(dtype=float)

            report_metrics = _calculate_regression_metrics(
                y_true_report,
                y_pred_report,
            )
            fold_score = _calculate_selection_objective(
                report_metrics,
                self.selection_objective,
            )
            transformed_rmse = float(np.sqrt(mean_squared_error(y_test_fold, y_pred)))
            fold_scores.append(float(fold_score))
            rmse_value = report_metrics["rmse"]
            r2_value = report_metrics["r2"]
            fold_rmse.append(float(rmse_value) if rmse_value is not None else np.nan)
            fold_r2.append(float(r2_value) if r2_value is not None else np.nan)
            if report_metrics["cov"] is not None:
                fold_cov.append(float(report_metrics["cov"]))
            fold_details.append(
                {
                    "fold": fold_index + 1,
                    "selection_score": float(fold_score),
                    "rmse_original_space": float(rmse_value) if rmse_value is not None else None,
                    "r2_original_space": float(r2_value) if r2_value is not None else None,
                    "cov_original_space": report_metrics["cov"],
                    "rmse_training_space": transformed_rmse,
                    "n_train_outer": int(len(train_idx)),
                    "n_test_outer": int(len(test_idx)),
                    "n_train_inner": int(len(X_fit)),
                    "n_validation_inner": int(len(X_val)) if X_val is not None else 0,
                    "best_iteration": getattr(fold_model, "best_iteration", None),
                }
            )

        cv_time = time.time() - start_time
        cv_scores = np.asarray(fold_scores, dtype=float)
        return {
            "cv_scores": cv_scores,
            "mean_cv_score": float(np.mean(cv_scores)),
            "std_cv_score": float(np.std(cv_scores)),
            "max_cv_score": float(np.max(cv_scores)),
            "min_cv_score": float(np.min(cv_scores)),
            "mean_cv_rmse": float(np.mean(fold_rmse)) if fold_rmse else None,
            "std_cv_rmse": float(np.std(fold_rmse)) if fold_rmse else None,
            "mean_cv_r2": float(np.mean(fold_r2)) if fold_r2 else None,
            "std_cv_r2": float(np.std(fold_r2)) if fold_r2 else None,
            "mean_cv_cov": float(np.mean(fold_cov)) if fold_cov else None,
            "std_cv_cov": float(np.std(fold_cov)) if fold_cov else None,
            "cv_time": cv_time,
            "n_folds": n_folds,
            "fold_details": fold_details,
            "metric_space": scoring_space,
            "target_mode": self.target_mode,
            "selection_objective": self.selection_objective,
            "validation_size": self.validation_size,
            "early_stopping_rounds": self.early_stopping_rounds,
            "eval_metric": self.eval_metric,
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight_val: Optional[Union[pd.Series, np.ndarray]] = None,
        eval_set: Optional[List[Tuple[pd.DataFrame, pd.Series]]] = None,
        early_stopping_rounds: Optional[int] = None,
        eval_metric: Optional[str] = None,
    ) -> xgb.XGBRegressor:
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training target values
            X_val: Validation features (optional)
            y_val: Validation target values (optional)
            eval_set: Evaluation set list for XGBoost (optional)
            early_stopping_rounds: Early stopping rounds for XGBoost (optional)
            eval_metric: Evaluation metric for XGBoost (optional)

        Returns:
            Trained XGBRegressor model

        Raises:
            Exception: If training fails
        """
        logger.info("Starting model training")
        logger.info(f"Training data shape: {X_train.shape}")
        if eval_set:
            logger.info(f"Evaluation set provided with {len(eval_set)} dataset(s)")
        elif X_val is not None:
            logger.info(f"Validation data shape: {X_val.shape}")

        try:
            # Prepare evaluation set if validation data provided
            eval_set_to_use = eval_set
            if eval_set_to_use is None and X_val is not None and y_val is not None:
                eval_set_to_use = [(X_val, y_val)]

            if early_stopping_rounds is not None and not eval_set_to_use:
                logger.warning(
                    "early_stopping_rounds provided without eval_set; early stopping will be ignored"
                )

            # Train model
            start_time = time.time()
            if eval_set_to_use and y_val is None and X_val is None:
                eval_X, eval_y = eval_set_to_use[0]
            else:
                eval_X, eval_y = X_val, y_val
            self.model = self._fit_model(
                params=self.params,
                X_train=X_train,
                y_train=y_train,
                sample_weight_train=sample_weight,
                X_val=eval_X,
                y_val=eval_y,
                sample_weight_val=sample_weight_val,
                early_stopping_rounds=early_stopping_rounds,
                eval_metric=eval_metric,
            )

            training_time = time.time() - start_time
            logger.info(f"Model training completed in {training_time:.2f} seconds")

            # Log training completion
            self.training_history.append(
                {
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "n_samples": len(X_train),
                    "n_features": X_train.shape[1],
                    "training_time": training_time,
                    "has_validation": bool(eval_set_to_use),
                }
            )

            # Log feature importance summary
            if hasattr(self.model, "feature_importances_"):
                importance_dict = dict(
                    zip(X_train.columns, self.model.feature_importances_)
                )
                top_features = sorted(
                    importance_dict.items(), key=lambda x: x[1], reverse=True
                )[:10]
                logger.info(f"Top 5 most important features:")
                for feat, imp in top_features[:5]:
                    logger.info(f"  {feat}: {imp:.4f}")

            return self.model

        except Exception as e:
            error_msg = f"Model training failed: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        y_report: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None,
        cv: CrossValidatorLike = 5,
        scoring: str = "neg_root_mean_squared_error",
        metric_space: Optional[str] = None,
        target_transform_type: Optional[str] = None,
        stratify_labels: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.

        Args:
            X: Features DataFrame
            y: Target Series
            y_report: Original reported target used for model selection and final metrics
            cv: Number of folds or cross-validator splitter
            scoring: Scoring metric (default: negative RMSE)
            metric_space: RMSE scoring space ('transformed' or 'original')
            target_transform_type: Target transform used before fitting
            stratify_labels: Optional labels used for stratified outer/inner splits

        Returns:
            Dictionary with cross-validation results
        """
        if scoring != "neg_root_mean_squared_error":
            raise ValueError(
                "Only neg_root_mean_squared_error is supported in the aligned CV path"
            )

        n_folds = cv if isinstance(cv, int) and not isinstance(cv, bool) else cv.get_n_splits(X, y)
        logger.info(f"Starting {n_folds}-fold cross-validation")
        results = self._score_with_consistent_cv(
            params=self.params,
            X=X,
            y=y,
            cv=cv,
            metric_space=metric_space,
            target_transform_type=target_transform_type,
            stratify_labels=stratify_labels,
            y_report=y_report,
            sample_weight=sample_weight,
        )
        scoring_space = results["metric_space"]

        logger.info(f"Cross-validation completed in {results['cv_time']:.2f} seconds")
        logger.info(f"CV scores: {results['cv_scores']}")
        logger.info(
            f"Mean CV composite score ({scoring_space} space): "
            f"{results['mean_cv_score']:.4f} (+/- {results['std_cv_score']:.4f})"
        )
        if results.get("mean_cv_rmse") is not None:
            mean_cv_cov = results["mean_cv_cov"]
            mean_cv_r2 = results["mean_cv_r2"]
            cov_text = f"{mean_cv_cov:.4f}" if mean_cv_cov is not None else "N/A"
            r2_text = f"{mean_cv_r2:.4f}" if mean_cv_r2 is not None else "N/A"
            logger.info(
                "Mean CV report-space metrics: "
                f"RMSE={results['mean_cv_rmse']:.4f}, "
                f"R²={r2_text}, "
                f"COV={cov_text}"
            )

        return results

    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        y_report: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None,
        cv: CrossValidatorLike = 5,
        n_trials: Optional[int] = None,
        study_name: str = "xgboost_optimization",
        storage_url: str = "sqlite:///logs/optuna_study.db",
        best_params_output_path: str = "logs/best_params.json",
        run_context: Optional[Dict[str, Any]] = None,
        stratify_labels: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            X: Features DataFrame
            y: Target Series
            y_report: Original reported target used for model selection
            cv: Number of folds or cross-validator splitter
            n_trials: Number of optimization trials (uses instance default if None)
            study_name: Optuna study name
            storage_url: Optuna storage URL
            best_params_output_path: Output path for best parameters snapshot
            run_context: Optional context metadata (context_hash, data_file, etc.)
            stratify_labels: Optional labels used for stratified outer/inner splits

        Returns:
            Dictionary with optimization results

        Raises:
            ImportError: If optuna is not installed
        """
        if not self.use_optuna:
            logger.warning("Optuna optimization not enabled during initialization")
            return {}

        try:
            import optuna
        except ImportError:
            error_msg = "Optuna is not installed. Install with: pip install optuna"
            logger.error(error_msg)
            raise ImportError(error_msg)

        n_trials = n_trials or self.n_trials

        logger.info(
            f"Starting Optuna hyperparameter optimization with {n_trials} trials"
        )
        logger.info(f"Optimization timeout: {self.optuna_timeout} seconds")
        logger.info(f"Optuna strategy version: {OPTUNA_SEARCH_SPACE_VERSION}")
        logger.info(
            "Optuna selection metric: composite objective in reported target space "
            f"(target_mode={self.target_mode}, target_transform={self.target_transform_type or 'none'})"
        )

        center_point = self.get_optuna_center_point()
        search_space = self.get_optuna_search_space()
        logger.info(f"Optuna center point: {json.dumps(center_point, sort_keys=True)}")
        logger.info(f"Optuna search space: {json.dumps(search_space, sort_keys=True)}")

        # Create logs directory if it doesn't exist
        Path("logs").mkdir(parents=True, exist_ok=True)

        context_hash = run_context.get("context_hash") if run_context else None
        data_file = run_context.get("data_file") if run_context else None
        cv_splitter = (
            KFold(
                n_splits=cv,
                shuffle=True,
                random_state=self.params.get("random_state", 42),
            )
            if isinstance(cv, int) and not isinstance(cv, bool)
            else cv
        )
        n_folds = (
            cv if isinstance(cv, int) and not isinstance(cv, bool) else cv_splitter.get_n_splits(X, y)
        )

        cpu_count = max(1, os.cpu_count() or 1)
        if self.params.get("device") == "cuda":
            objective_model_n_jobs = 1
        else:
            objective_model_n_jobs = max(1, cpu_count // max(1, n_folds))
        logger.info(
            "Optuna CV execution: "
            f"{n_folds} folds, model_n_jobs={objective_model_n_jobs}, "
            f"validation_size={self.validation_size}, "
            f"early_stopping_rounds={self.early_stopping_rounds}"
        )

        # Define objective function for Optuna
        def objective(trial: Trial) -> float:
            params = self.build_optuna_trial_params(trial)
            params["n_jobs"] = objective_model_n_jobs
            cv_results = self._score_with_consistent_cv(
                params=params,
                X=X,
                y=y,
                cv=cv_splitter,
                metric_space=self.optuna_metric_space,
                target_transform_type=self.target_transform_type,
                stratify_labels=stratify_labels,
                y_report=y_report,
                sample_weight=sample_weight,
            )

            return float(cv_results["mean_cv_score"])

        # Create Optuna study with persistent storage
        sampler = optuna.samplers.TPESampler(
            seed=self.params.get("random_state", 42),
            multivariate=True,
        )
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            sampler=sampler,
        )
        n_trials_before = len(study.trials)
        if n_trials_before == 0:
            study.enqueue_trial(center_point)
            logger.info("Enqueued config center point as the first Optuna trial")

        # Run optimization
        start_time = time.time()
        objective_fn = cast(Any, objective)
        study.optimize(objective_fn, n_trials=n_trials, timeout=self.optuna_timeout)
        opt_time = time.time() - start_time
        n_trials_after = len(study.trials)

        # Get best parameters
        best_params = study.best_params
        best_score = study.best_value
        best_trial = study.best_trial

        logger.info(f"Optuna optimization completed in {opt_time:.2f} seconds")
        logger.info(f"Best composite objective: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")

        # Save best parameters to file
        save_best_params(
            best_params=best_params,
            best_score=best_score,
            trial_number=best_trial.number,
            n_trials=n_trials_after,
            output_path=best_params_output_path,
            context_hash=context_hash,
            data_file=data_file,
            study_name=study_name,
            storage_url=storage_url,
            score_label="best_composite_objective",
        )

        # Update model parameters with best found parameters
        self.params.update(best_params)
        logger.info(f"Updated model parameters with best parameters")

        # Return optimization results
        results = {
            "best_params": best_params,
            "best_score": best_score,
            "metric_space": self.optuna_metric_space,
            "target_transform_type": self.target_transform_type,
            "target_mode": self.target_mode,
            "selection_objective": self.selection_objective,
            "n_trials_before": n_trials_before,
            "n_trials_after": n_trials_after,
            "n_trials": n_trials_after,
            "optimization_time": opt_time,
            "study_name": study_name,
            "storage_url": storage_url,
            "study": study,
        }

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            logger.warning("No model trained yet")
            return {}

        info = {
            "model_type": "XGBRegressor",
            "trained": self.model is not None,
            "parameters": self.params,
            "n_features": getattr(self.model, "n_features_in_", None),
            "feature_names": getattr(self.model, "feature_names_in_", None),
            "training_history": self.training_history,
        }

        return info

    def save_training_history(self, output_path: str) -> None:
        """
        Save training history to JSON file.

        Args:
            output_path: Path to save training history
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(self.training_history, f, indent=2)

            logger.info(f"Training history saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save training history: {str(e)}")

    def __str__(self) -> str:
        """
        String representation of ModelTrainer.
        """
        info = self.get_model_info()
        if not info:
            return "ModelTrainer (no model trained)"

        return f"ModelTrainer(XGBRegressor, n_features={info.get('n_features', 'N/A')})"
