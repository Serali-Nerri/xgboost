#!/usr/bin/env python3
"""
Training Script for CFST XGBoost Pipeline

This script executes the complete training pipeline:
1. Load configuration
2. Load data
3. Preprocess data
4. Train XGBoost model
5. Evaluate model
6. Save model and results

Usage:
    python train.py --config config/config.yaml
    python train.py --config config/config.yaml --output models/my_model
"""

import argparse
import hashlib
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from src.domain_features import (
    get_training_target_name,
    normalize_target_mode,
    restore_report_target,
)
from src.feature_sets import resolve_feature_selection
from src.utils.logger import setup_logger
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model_trainer import ModelTrainer, OPTUNA_SEARCH_SPACE_VERSION
from src.evaluator import Evaluator
from src.utils.model_utils import save_model
from src.splitting import (
    build_regression_stratification_labels,
    get_split_strategy,
    required_stratum_count,
)
from src.visualizer import (
    create_evaluation_dashboard,
)

logger = setup_logger(__name__)


XGB_PARAM_KEYS = {
    "objective",
    "max_depth",
    "learning_rate",
    "n_estimators",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "reg_alpha",
    "reg_lambda",
    "gamma",
    "random_state",
    "tree_method",
    "device",
    "n_jobs",
}

REQUIRED_MODEL_PARAM_KEYS = XGB_PARAM_KEYS.copy()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping at the top level")

    return cast(Dict[str, Any], config)


def _file_sha256(file_path: str) -> str:
    """Compute SHA256 for a file path."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_training_context(
    data_file_path: str,
    target_column: str,
    target_mode: str,
    target_transform_type: Optional[str],
    columns_to_drop: List[str],
    feature_selection: Dict[str, Any],
    optuna_metric_space: str,
    selection_objective: Dict[str, Any],
    split_strategy: str,
    split_config: Dict[str, Any],
    validation_size: float,
    early_stopping_rounds: Optional[int],
    eval_metric: Optional[str],
) -> Dict[str, Any]:
    """
    Build deterministic training context for artifact compatibility checks.
    """
    data_sha256 = _file_sha256(data_file_path)
    context_payload = {
        "data_file": str(Path(data_file_path).resolve()),
        "data_sha256": data_sha256,
        "target_column": target_column,
        "target_mode": target_mode,
        "target_transform_type": target_transform_type or "none",
        "columns_to_drop": sorted(columns_to_drop),
        "feature_selection": feature_selection,
        "optuna_metric_space": optuna_metric_space,
        "selection_objective": selection_objective,
        "split_strategy": split_strategy,
        "split_config": split_config,
        "validation_size": validation_size,
        "early_stopping_rounds": early_stopping_rounds,
        "eval_metric": eval_metric,
    }
    context_json = json.dumps(
        context_payload, sort_keys=True, ensure_ascii=True
    ).encode("utf-8")
    context_hash = hashlib.sha256(context_json).hexdigest()[:12]
    context_payload["context_hash"] = context_hash
    return context_payload


def build_study_name(data_file_path: str, context_hash: str) -> str:
    """Build an Optuna study name isolated by dataset fingerprint."""
    dataset_stem = Path(data_file_path).stem
    sanitized_stem = re.sub(r"[^A-Za-z0-9_]+", "_", dataset_stem).strip("_")
    return f"xgboost_optimization__{sanitized_stem}__{context_hash}"


def build_optuna_tuning_fingerprint(
    model_params: Dict[str, Any],
    cv_config: Dict[str, Any],
    optuna_metric_space: str,
    target_mode: str,
    feature_selection: Dict[str, Any],
    selection_objective: Dict[str, Any],
    split_strategy: str,
    split_config: Dict[str, Any],
    validation_size: float,
    early_stopping_rounds: Optional[int],
    eval_metric: Optional[str],
) -> str:
    """Build a deterministic fingerprint for the Optuna tuning strategy."""
    fingerprint_payload = {
        "search_space_version": OPTUNA_SEARCH_SPACE_VERSION,
        "optuna_metric_space": optuna_metric_space,
        "target_mode": target_mode,
        "feature_selection": feature_selection,
        "selection_objective": selection_objective,
        "split_strategy": split_strategy,
        "split_config": split_config,
        "validation_size": validation_size,
        "early_stopping_rounds": early_stopping_rounds,
        "eval_metric": eval_metric,
        "model_params": model_params,
        "cv": {
            "n_splits": get_cv_n_splits(cv_config),
            "shuffle": cv_config.get("shuffle", False),
            "random_state": cv_config.get("random_state"),
        },
    }
    fingerprint_json = json.dumps(
        fingerprint_payload, sort_keys=True, ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(fingerprint_json).hexdigest()[:10]


def build_versioned_study_name(
    data_file_path: str, context_hash: str, tuning_fingerprint: str
) -> str:
    """Build an Optuna study name isolated by data context and tuning strategy."""
    base_study_name = build_study_name(data_file_path, context_hash)
    return f"{base_study_name}__{tuning_fingerprint}"


def build_xgb_params(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read model.params as the only valid hyperparameter source.
    """
    legacy_top_level_keys = sorted(XGB_PARAM_KEYS.intersection(model_config.keys()))
    if legacy_top_level_keys:
        raise ValueError(
            "Legacy model hyperparameter keys at config.model top-level are not allowed: "
            f"{legacy_top_level_keys}. Move them under config.model.params."
        )

    params = model_config.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError(
            "config.model.params must be a non-empty dictionary and is the only valid "
            "location for XGBoost hyperparameters."
        )

    missing_required = sorted(
        key for key in REQUIRED_MODEL_PARAM_KEYS if key not in params
    )
    if missing_required:
        raise ValueError(
            f"config.model.params is missing required keys: {missing_required}"
        )

    return params.copy()


def get_cv_n_splits(cv_config: Dict[str, Any]) -> int:
    """Read cross-validation folds from cv.n_splits only."""
    if "n_folds" in cv_config:
        raise ValueError(
            "config.cv.n_folds is deprecated. Use config.cv.n_splits instead."
        )
    n_splits = cv_config.get("n_splits", 5)
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("config.cv.n_splits must be an integer >= 2")
    return n_splits


def format_target_space_description(
    report_target_column: str,
    target_mode: str,
    target_transform_type: Optional[str],
) -> str:
    """Describe the target space used by fitting and reporting."""
    target_name = get_training_target_name(report_target_column, target_mode)
    if target_transform_type == "log":
        return f"ln({target_name}) -> inverse -> {report_target_column}"
    if target_transform_type == "sqrt":
        return f"sqrt({target_name}) -> inverse -> {report_target_column}"
    if target_name != report_target_column:
        return f"{target_name} -> {report_target_column}"
    return report_target_column


def format_training_space_label(
    report_target_column: str,
    target_mode: str,
    target_transform_type: Optional[str],
) -> str:
    """Human-readable label for the model output space."""
    target_name = get_training_target_name(report_target_column, target_mode)
    if target_transform_type == "log":
        return f"ln({target_name}) space"
    if target_transform_type == "sqrt":
        return f"sqrt({target_name}) space"
    return target_name


def build_target_metadata(
    report_target_column: str,
    target_mode: str,
    target_transform_type: Optional[str],
    derived_columns: List[str],
) -> Dict[str, Any]:
    """Build a consistent target metadata payload for saved artifacts."""
    modeled_target_column = get_training_target_name(report_target_column, target_mode)
    return {
        "target_mode": target_mode,
        "report_target_column": report_target_column,
        "modeled_target_column": modeled_target_column,
        "derived_columns": list(derived_columns),
        "target_transform": {
            "enabled": target_transform_type is not None,
            "type": target_transform_type,
            "mode": target_mode,
            "original_column": report_target_column,
            "modeled_column": modeled_target_column,
            "derived_columns": list(derived_columns),
        },
    }


def select_final_n_estimators(cv_results: Dict[str, Any], fallback: int) -> Tuple[int, List[int]]:
    """Select a final tree count from CV best_iteration values."""
    fold_details = cast(List[Dict[str, Any]], cv_results.get("fold_details", []))
    best_iterations = [
        int(detail["best_iteration"]) + 1
        for detail in fold_details
        if detail.get("best_iteration") is not None
    ]
    if not best_iterations:
        return int(fallback), []
    return int(np.median(np.asarray(best_iterations, dtype=int))), best_iterations


def build_cv_splitter(
    cv_config: Dict[str, Any], split_strategy: str
) -> KFold:
    """Build a CV splitter from config.cv settings."""
    n_splits = get_cv_n_splits(cv_config)
    shuffle = cv_config.get("shuffle", False)
    random_state = cv_config.get("random_state")

    if not isinstance(shuffle, bool):
        raise ValueError("config.cv.shuffle must be a boolean")

    if isinstance(random_state, bool) or (
        random_state is not None and not isinstance(random_state, int)
    ):
        raise ValueError("config.cv.random_state must be an integer or null")

    splitter_random_state = random_state if shuffle else None
    if split_strategy == "regression_stratified":
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=splitter_random_state,
        )

    return KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=splitter_random_state,
    )


def train_model(config_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute complete training pipeline.

    Args:
        config_path: Path to configuration YAML file
        output_dir: Output directory for saving model and results (optional)

    Returns:
        Dictionary with training results

    Raises:
        Exception: If any step in the pipeline fails
    """
    logger.info("=" * 80)
    logger.info("CFST XGBOOST PIPELINE - TRAINING STARTED")
    logger.info("=" * 80)

    try:
        # Step 1: Load configuration
        logger.info("Step 1: Loading configuration...")
        config = load_config(config_path)
        logger.info(f"Configuration loaded from {config_path}")

        # Extract paths and parameters
        data_config = cast(Dict[str, Any], config.get("data", {}))
        model_config = cast(Dict[str, Any], config.get("model", {}))
        cv_config = cast(Dict[str, Any], config.get("cv", {}))
        evaluation_config = cast(Dict[str, Any], config.get("evaluation", {}))
        output_config = cast(Dict[str, Any], config.get("paths", {}))

        data_path = data_config.get("file_path")
        target_column = data_config.get("target_column", "K")
        columns_to_drop = data_config.get("columns_to_drop", [])
        feature_selection_config = cast(
            Dict[str, Any],
            data_config.get("feature_selection", {}),
        )
        split_config = cast(Dict[str, Any], data_config.get("split", {}))
        if not data_path:
            raise ValueError("config.data.file_path is required")

        # Set output directory
        if output_dir is None:
            output_dir = output_config.get("output_dir", "output")

        output_dir = cast(str, output_dir)

        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Data file: {data_path}")
        logger.info(f"Target column: {target_column}")
        logger.info(f"Columns to drop: {columns_to_drop}")

        target_mode = normalize_target_mode(data_config.get("target_mode", "raw"))

        # Read target transform configuration
        target_transform_config = data_config.get("target_transform", {})
        target_transform_type = (
            target_transform_config.get("type", None)
            if target_transform_config.get("enabled", False)
            else None
        )

        if target_transform_type:
            logger.info(f"Target transform enabled: {target_transform_type}")

        optuna_metric_space = str(
            model_config.get("optuna_metric_space", "transformed")
        )
        cv_metric_space = str(model_config.get("cv_metric_space", optuna_metric_space))
        validation_size = float(model_config.get("validation_size", 0.1) or 0.0)
        early_stopping_rounds = model_config.get("early_stopping_rounds")
        eval_metric = model_config.get("eval_metric")
        selection_objective_config = cast(
            Dict[str, Any],
            model_config.get("selection_objective", {}),
        )
        split_strategy = get_split_strategy(split_config)
        regime_config = cast(
            Dict[str, Any], evaluation_config.get("regime_analysis", {})
        )
        logger.info(f"Optuna RMSE metric space: {optuna_metric_space}")
        logger.info(f"Cross-validation RMSE metric space: {cv_metric_space}")
        logger.info(f"Data split strategy: {split_strategy}")
        logger.info(f"Target mode: {target_mode}")

        # Validate XGBoost params early so study naming matches the real tuning config.
        xgb_params = build_xgb_params(model_config)

        # Step 2: Load data
        logger.info("\nStep 2: Loading data...")
        data_loader = DataLoader(required_columns=[target_column])

        features, target_transformed = data_loader.load_data(
            data_path,
            target_column,
            target_transform=target_transform_type,
            target_mode=target_mode,
        )

        # Save report target and modeled target
        report_target_raw = data_loader.target_raw
        training_target_raw = data_loader.training_target_raw
        if report_target_raw is None:
            raise ValueError("DataLoader did not preserve raw target values")
        if training_target_raw is None:
            raise ValueError("DataLoader did not preserve raw modeled target values")

        logger.info(
            f"Data loaded: {len(features)} samples, {len(features.columns)} features"
        )

        feature_selection = resolve_feature_selection(
            features.columns.tolist(),
            columns_to_drop=columns_to_drop,
            feature_selection_config=feature_selection_config,
            feature_frame=features,
        )
        selected_features = cast(List[str], feature_selection["selected_features"])
        logger.info(
            "Feature selection resolved: source=%s, preset=%s, selected=%s",
            feature_selection["selection_source"],
            feature_selection.get("preset"),
            feature_selection["selected_feature_count"],
        )
        logger.info("Selected feature list: %s", selected_features)

        training_context = build_training_context(
            data_file_path=data_path,
            target_column=target_column,
            target_mode=target_mode,
            target_transform_type=target_transform_type,
            columns_to_drop=columns_to_drop,
            feature_selection=feature_selection,
            optuna_metric_space=optuna_metric_space,
            selection_objective=selection_objective_config,
            split_strategy=split_strategy,
            split_config=split_config,
            validation_size=validation_size,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric=eval_metric,
        )
        context_hash = training_context["context_hash"]
        tuning_fingerprint = build_optuna_tuning_fingerprint(
            xgb_params,
            cv_config,
            optuna_metric_space,
            target_mode,
            feature_selection,
            selection_objective_config,
            split_strategy,
            split_config,
            validation_size,
            early_stopping_rounds,
            eval_metric,
        )
        study_name = build_versioned_study_name(
            data_path, context_hash, tuning_fingerprint
        )
        logger.info(f"Training context hash: {context_hash}")
        logger.info(f"Optuna strategy version: {OPTUNA_SEARCH_SPACE_VERSION}")
        logger.info(f"Optuna tuning fingerprint: {tuning_fingerprint}")
        logger.info(f"Optuna study name: {study_name}")

        # Step 2.5: Split data into train/test sets (FIXES DATA LEAKAGE)
        logger.info("\nStep 2.5: Splitting data into train/test sets...")
        test_size = data_config.get("test_size", 0.2)
        random_state = data_config.get("random_state", 42)
        n_cv_splits = get_cv_n_splits(cv_config)
        stratify_labels_full: Optional[pd.Series] = None
        stratification_metadata: Dict[str, Any] = {"strategy": split_strategy}
        minimum_stratum_size = required_stratum_count(
            test_size=float(test_size),
            validation_size=validation_size,
            n_splits=n_cv_splits,
            configured_minimum=cast(Optional[int], split_config.get("min_stratum_size")),
        )
        if split_strategy == "regression_stratified":
            stratify_labels_candidate, stratification_metadata = (
                build_regression_stratification_labels(
                    features=features,
                    target_raw=training_target_raw,
                    split_config=split_config,
                    minimum_count=minimum_stratum_size,
                )
            )
            if stratify_labels_candidate.nunique() > 1:
                stratify_labels_full = stratify_labels_candidate
                logger.info(
                    "Regression stratification enabled: "
                    f"{stratification_metadata['n_strata']} strata, "
                    f"min_count={stratification_metadata['minimum_count_observed']}, "
                    f"required>={stratification_metadata['minimum_count_required']}"
                )
            else:
                logger.warning(
                    "Regression stratification requested but only one stable stratum was produced; "
                    "falling back to random split"
                )

        split_kwargs: Dict[str, Any] = {
            "test_size": test_size,
            "random_state": random_state,
        }
        if stratify_labels_full is not None:
            split_kwargs["stratify"] = stratify_labels_full
            split_result = cast(
                Tuple[
                    pd.DataFrame,
                    pd.DataFrame,
                    pd.Series,
                    pd.Series,
                    pd.Series,
                    pd.Series,
                    pd.Series,
                    pd.Series,
                    pd.Series,
                    pd.Series,
                ],
                train_test_split(
                    features,
                    target_transformed,
                    report_target_raw,
                    training_target_raw,
                    stratify_labels_full,
                    **split_kwargs,
                ),
            )
            (
                X_train_full,
                X_test,
                y_train_trans_full,
                y_test_trans,
                y_train_report_full,
                y_test_report,
                y_train_model_raw_full,
                y_test_model_raw,
                train_strata_full,
                test_strata,
            ) = split_result
        else:
            split_result = cast(
                Tuple[
                    pd.DataFrame,
                    pd.DataFrame,
                    pd.Series,
                    pd.Series,
                    pd.Series,
                    pd.Series,
                    pd.Series,
                    pd.Series,
                ],
                train_test_split(
                    features,
                    target_transformed,
                    report_target_raw,
                    training_target_raw,
                    **split_kwargs,
                ),
            )
            (
                X_train_full,
                X_test,
                y_train_trans_full,
                y_test_trans,
                y_train_report_full,
                y_test_report,
                y_train_model_raw_full,
                y_test_model_raw,
            ) = split_result
            train_strata_full = None
            test_strata = None

        effective_split_strategy = (
            "regression_stratified" if train_strata_full is not None else "random"
        )
        if effective_split_strategy != split_strategy:
            logger.warning(
                "Effective split strategy downgraded to random because stable strata were unavailable"
            )

        logger.info(
            f"Training set: {len(X_train_full)} samples ({(1 - test_size) * 100:.0f}%)"
        )
        logger.info(f"Test set: {len(X_test)} samples ({test_size * 100:.0f}%)")

        # Step 3: Prepare trainer for model selection and final refit
        logger.info("\nStep 3: Preparing model selection and final retraining...")
        use_optuna = model_config.get("use_optuna", False)
        n_trials = model_config.get("n_trials", 100)
        optuna_timeout = model_config.get("optuna_timeout", 3600)
        optuna_storage_url = model_config.get(
            "optuna_storage_url", "sqlite:///logs/optuna_study.db"
        )
        best_params_path = model_config.get("best_params_path", "logs/best_params.json")
        cv_splitter = build_cv_splitter(cv_config, effective_split_strategy)

        trainer = ModelTrainer(
            params=xgb_params,
            use_optuna=use_optuna,
            n_trials=n_trials,
            optuna_timeout=optuna_timeout,
            best_params_path=best_params_path,
            expected_context_hash=context_hash,
            optuna_metric_space=optuna_metric_space,
            target_transform_type=target_transform_type,
            target_mode=target_mode,
            columns_to_drop=columns_to_drop,
            include_features=selected_features,
            validation_size=validation_size,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric=eval_metric,
            selection_objective=selection_objective_config,
        )

        params_source = (
            "best_params_file" if trainer.loaded_best_params else "config_model_params"
        )
        optuna_run_info = None

        # Step 4: Optional hyperparameter optimization in CV
        if use_optuna:
            logger.info("Starting Optuna hyperparameter optimization...")
            opt_results = trainer.optimize_hyperparameters(
                X_train_full,
                y_train_trans_full,
                y_report=y_train_report_full,
                cv=cv_splitter,
                study_name=study_name,
                storage_url=optuna_storage_url,
                best_params_output_path=best_params_path,
                run_context=training_context,
                stratify_labels=train_strata_full,
            )
            logger.info(
                "Optuna optimization completed: "
                f"{opt_results['n_trials_before']} -> {opt_results['n_trials_after']} trials"
            )
            params_source = "optuna_best"
            optuna_run_info = {
                "study_name": opt_results["study_name"],
                "storage_url": opt_results["storage_url"],
                "n_trials_before": opt_results["n_trials_before"],
                "n_trials_after": opt_results["n_trials_after"],
                "best_score": opt_results["best_score"],
                "metric_space": opt_results["metric_space"],
                "target_transform_type": opt_results["target_transform_type"],
                "target_mode": opt_results["target_mode"],
                "selection_objective": opt_results["selection_objective"],
                "best_params": opt_results["best_params"],
            }

        logger.info("\nStep 4.5: Performing cross-validation on training data...")
        cv_results = trainer.cross_validate(
            X_train_full,
            y_train_trans_full,
            y_report=y_train_report_full,
            cv=cv_splitter,
            metric_space=cv_metric_space,
            target_transform_type=target_transform_type,
            stratify_labels=train_strata_full,
        )
        logger.info(
            "Cross-validation composite score "
            f"({cv_metric_space} space): {cv_results['mean_cv_score']:.4f} "
            f"(+/- {cv_results['std_cv_score']:.4f})"
        )

        final_n_estimators, fold_best_iterations = select_final_n_estimators(
            cv_results,
            fallback=int(trainer.params.get("n_estimators", xgb_params["n_estimators"])),
        )
        trainer.params["n_estimators"] = final_n_estimators
        logger.info(
            "Selected final n_estimators=%s from CV fold best_iteration values=%s",
            final_n_estimators,
            fold_best_iterations,
        )

        logger.info("\nStep 5: Fitting final model on full training split...")
        preprocessor = Preprocessor(
            columns_to_drop=columns_to_drop,
            include_features=selected_features,
        )
        X_train_processed = preprocessor.fit_transform(X_train_full)
        X_test_processed = preprocessor.transform(X_test)
        logger.info(
            f"Preprocessing completed: {len(X_train_processed.columns)} features remaining"
        )

        feature_names = preprocessor.get_remaining_features()
        logger.info(f"Remaining features: {feature_names}")

        missing_info_train = preprocessor.check_missing_values(X_train_processed)
        missing_info_test = preprocessor.check_missing_values(X_test_processed)
        if missing_info_train or missing_info_test:
            logger.warning(
                f"Found missing values - Train: {missing_info_train}, Test: {missing_info_test}"
            )
        else:
            logger.info("No missing values found in train or test sets")

        model = trainer.train(
            X_train_processed,
            y_train_trans_full,
            eval_set=None,
            early_stopping_rounds=None,
            eval_metric=None,
        )
        logger.info(
            "Final model training completed on target space: "
            f"{format_target_space_description(target_column, target_mode, target_transform_type)}"
        )

        # Step 6: Evaluate model on BOTH training and test sets
        logger.info("\nStep 6: Evaluating model...")
        evaluator = Evaluator()
        regime_schema = evaluator.fit_regime_schema(
            y_true=y_train_report_full,
            features=X_train_full,
            regime_config=regime_config,
        )

        # Make predictions on BOTH sets (in transformed space)
        from src.predictor import Predictor

        predictor = Predictor(model, preprocessor, feature_names)
        y_train_pred_trans = predictor.predict(X_train_full)
        y_test_pred_trans = predictor.predict(X_test)

        # Apply inverse transform + target-mode restoration to get back to report space
        y_train_pred_orig = restore_report_target(
            y_train_pred_trans,
            target_mode=target_mode,
            target_transform_type=target_transform_type,
            reference_features=X_train_full,
        )
        y_test_pred_orig = restore_report_target(
            y_test_pred_trans,
            target_mode=target_mode,
            target_transform_type=target_transform_type,
            reference_features=X_test,
        )
        logger.info("Mapped model outputs back to reported target space")

        # Calculate metrics in ORIGINAL space (recommended - true application scenario)
        train_metrics = evaluator.calculate_metrics(
            y_train_report_full, y_train_pred_orig
        )
        test_metrics = evaluator.calculate_metrics(y_test_report, y_test_pred_orig)

        # Also calculate metrics in transformed space (for reference)
        train_metrics_trans = evaluator.calculate_metrics(
            y_train_trans_full, y_train_pred_trans
        )
        test_metrics_trans = evaluator.calculate_metrics(
            y_test_trans, y_test_pred_trans
        )
        train_regime_metrics = evaluator.calculate_regime_metrics(
            y_true=y_train_report_full,
            y_pred=y_train_pred_orig,
            features=X_train_full,
            regime_schema=regime_schema,
        )
        test_regime_metrics = evaluator.calculate_regime_metrics(
            y_true=y_test_report,
            y_pred=y_test_pred_orig,
            features=X_test,
            regime_schema=regime_schema,
        )

        # Log original space metrics (PRIMARY)
        logger.info(f"Training set evaluation (original space):")
        logger.info(f"  RMSE: {train_metrics['rmse']:.4f} kN")
        logger.info(f"  MAE: {train_metrics['mae']:.4f} kN")
        logger.info(f"  R²: {train_metrics['r2']:.4f}")
        if train_metrics.get("mape"):
            logger.info(f"  MAPE: {train_metrics['mape']:.2f}%")
        if train_metrics.get("cov"):
            logger.info(f"  COV: {train_metrics['cov']:.4f}")

        logger.info(f"Test set evaluation (original space - TRUE GENERALIZATION):")
        logger.info(f"  RMSE: {test_metrics['rmse']:.4f} kN")
        logger.info(f"  MAE: {test_metrics['mae']:.4f} kN")
        logger.info(f"  R²: {test_metrics['r2']:.4f}")
        if test_metrics.get("mape"):
            logger.info(f"  MAPE: {test_metrics['mape']:.2f}%")
        if test_metrics.get("cov"):
            logger.info(f"  COV: {test_metrics['cov']:.4f}")
        if test_regime_metrics:
            logger.info("Test regime summary:")
            for regime_name, regime_result in test_regime_metrics.items():
                worst_group = regime_result.get("worst_rmse_group")
                if worst_group:
                    logger.info(
                        f"  {regime_name}: worst={worst_group['label']} "
                        f"(RMSE={worst_group['metrics']['rmse']:.4f} kN, "
                        f"n={worst_group['n_samples']})"
                    )

        # Log transformed space metrics (REFERENCE)
        training_space_label = format_training_space_label(
            target_column,
            target_mode,
            target_transform_type,
        )
        logger.info(f"\n--- Training Space Metrics (Reference) ---")
        logger.info(
            f"Training RMSE ({training_space_label}): {train_metrics_trans['rmse']:.4f}"
        )
        logger.info(
            f"Test RMSE ({training_space_label}): {test_metrics_trans['rmse']:.4f}"
        )
        logger.info(
            f"Train/Test ratio ({training_space_label}): "
            f"{test_metrics_trans['rmse'] / train_metrics_trans['rmse']:.2f}"
        )

        # Check for overfitting (using original space metrics)
        rmse_ratio = test_metrics["rmse"] / train_metrics["rmse"]
        if rmse_ratio > 1.2:
            logger.warning(
                f"Potential overfitting detected! Test RMSE is {rmse_ratio:.2f}x training RMSE"
            )
        elif rmse_ratio < 0.8:
            logger.warning(
                f"Unusual: Test RMSE is lower than training RMSE (ratio: {rmse_ratio:.2f})"
            )
        else:
            logger.info(
                f"Model generalization appears healthy (train/test RMSE ratio: {rmse_ratio:.2f})"
            )

        # Step 7: Create visualizations for BOTH training and test sets
        logger.info("\nStep 7: Creating visualizations...")
        output_path = Path(output_dir)
        plots_dir = output_path / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Training set plots (using original space for interpretability)
        create_evaluation_dashboard(
            y_train_report_full,
            y_train_pred_orig,
            model,
            feature_names,
            str(plots_dir),
            "xgboost_model_train",
        )

        # Test set plots (PRIMARY - shows true generalization)
        create_evaluation_dashboard(
            y_test_report,
            y_test_pred_orig,
            model,
            feature_names,
            str(plots_dir),
            "xgboost_model_test",
        )

        logger.info(f"Visualizations saved to {plots_dir}/")
        logger.info(f"  Training: xgboost_model_train_*.png")
        logger.info(f"  Test:     xgboost_model_test_*.png")

        # Step 8: Save model and artifacts
        logger.info("\nStep 8: Saving model and artifacts...")
        target_metadata = build_target_metadata(
            report_target_column=target_column,
            target_mode=target_mode,
            target_transform_type=target_transform_type,
            derived_columns=data_loader.derived_columns,
        )

        # Save model with metadata
        metadata = {
            "config": config,
            "context_hash": context_hash,
            "training_context": training_context,
            "params_source": params_source,
            "final_model_params": trainer.params.copy(),
            "optuna_run_info": optuna_run_info,
            "optuna_metric_space": optuna_metric_space,
            "cv_metric_space": cv_metric_space,
            "selection_objective": trainer.selection_objective,
            **target_metadata,
            "split_strategy_requested": split_strategy,
            "split_strategy_effective": effective_split_strategy,
            "stratification_metadata": stratification_metadata,
            "selection_metrics_cv": {
                "composite_objective": cv_results["mean_cv_score"],
                "rmse": cv_results.get("mean_cv_rmse"),
                "r2": cv_results.get("mean_cv_r2"),
                "cov": cv_results.get("mean_cv_cov"),
            },
            "feature_selection": feature_selection,
            "train_metrics_original_space": train_metrics,  # LEGACY
            "train_full_apparent_metrics_original_space": train_metrics,  # PRIMARY
            "test_metrics_original_space": test_metrics,  # PRIMARY
            "train_metrics_transformed_space": train_metrics_trans,  # REFERENCE
            "test_metrics_transformed_space": test_metrics_trans,  # REFERENCE
            "regime_schema": regime_schema,
            "train_regime_metrics_original_space": train_regime_metrics,
            "test_regime_metrics_original_space": test_regime_metrics,
            "cross_validation_results": cv_results,
            "feature_names": feature_names,
            "n_train_samples": len(X_train_full),
            "n_test_samples": len(X_test),
            "n_features": len(feature_names),
            "test_size": test_size,
            "n_train_fit_samples": len(X_train_full),
            "n_validation_samples": 0,
            "final_n_estimators_from_cv": final_n_estimators,
            "fold_best_iterations": fold_best_iterations,
            "training_successful": True,
            "overfitting_check": {
                "rmse_ratio_original": test_metrics["rmse"] / train_metrics["rmse"],
                "rmse_ratio_transformed": test_metrics_trans["rmse"]
                / train_metrics_trans["rmse"],
                "detected": test_metrics["rmse"] / train_metrics["rmse"] > 1.2,
            },
        }

        save_model(
            model=model,
            preprocessor=preprocessor,
            feature_names=feature_names,
            output_dir=output_dir,
            metadata=metadata,
        )

        # Convert cv_results to be JSON serializable
        serializable_cv_results = {}
        if isinstance(cv_results, dict):
            for key, value in cv_results.items():
                if hasattr(value, "tolist"):  # numpy arrays
                    serializable_cv_results[key] = value.tolist()
                elif isinstance(value, dict) and "cv_scores" in str(value):
                    # Handle nested CV results
                    serializable_cv_results[key] = {}
                    for k2, v2 in value.items():
                        if hasattr(v2, "tolist"):
                            serializable_cv_results[key][k2] = v2.tolist()
                        else:
                            serializable_cv_results[key][k2] = v2
                else:
                    serializable_cv_results[key] = value

        eval_report_path = output_path / "evaluation_report.json"
        evaluator.save_evaluation_report(
            {
                "model_name": "xgboost_model",
                "timestamp": pd.Timestamp.now().isoformat(),
                "context_hash": context_hash,
                "params_source": params_source,
                "final_model_params": trainer.params.copy(),
                "optuna_run_info": optuna_run_info,
                "optuna_metric_space": optuna_metric_space,
                "cv_metric_space": cv_metric_space,
                "selection_objective": trainer.selection_objective,
                **target_metadata,
                "split_strategy_requested": split_strategy,
                "split_strategy_effective": effective_split_strategy,
                "stratification_metadata": stratification_metadata,
                "selection_metrics_cv": {
                    "composite_objective": cv_results["mean_cv_score"],
                    "rmse": cv_results.get("mean_cv_rmse"),
                    "r2": cv_results.get("mean_cv_r2"),
                    "cov": cv_results.get("mean_cv_cov"),
                },
                "feature_selection": feature_selection,
                "train_metrics_original_space": train_metrics,  # LEGACY
                "train_full_apparent_metrics_original_space": train_metrics,  # PRIMARY
                "test_metrics_original_space": test_metrics,  # PRIMARY
                "train_metrics_transformed_space": train_metrics_trans,  # REFERENCE
                "test_metrics_transformed_space": test_metrics_trans,  # REFERENCE
                "regime_schema": regime_schema,
                "train_regime_metrics_original_space": train_regime_metrics,
                "test_regime_metrics_original_space": test_regime_metrics,
                "cv_results": serializable_cv_results,
                "feature_names": feature_names,
                "n_features": len(feature_names),
                "data_split": {
                    "n_train": len(X_train_full),
                    "n_test": len(X_test),
                    "test_size": test_size,
                    "n_train_fit": len(X_train_full),
                    "n_validation": 0,
                    "n_strata_train": int(train_strata_full.nunique())
                    if train_strata_full is not None
                    else None,
                    "n_strata_test": int(test_strata.nunique())
                    if test_strata is not None
                    else None,
                },
                "final_n_estimators_from_cv": final_n_estimators,
                "fold_best_iterations": fold_best_iterations,
                "overfitting_analysis": {
                    "rmse_ratio_original": test_metrics["rmse"] / train_metrics["rmse"],
                    "rmse_ratio_transformed": test_metrics_trans["rmse"]
                    / train_metrics_trans["rmse"],
                    "status": "overfitting"
                    if test_metrics["rmse"] / train_metrics["rmse"] > 1.2
                    else "healthy",
                },
            },
            str(eval_report_path),
        )

        logger.info(f"Model and artifacts saved to {output_dir}")

        # Step 9: Final summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Model saved to: {output_dir}/xgboost_model.pkl")
        logger.info(f"Preprocessor saved to: {output_dir}/preprocessor.pkl")
        logger.info(f"Evaluation report: {eval_report_path}")
        logger.info(f"Plots saved to: {plots_dir}/")
        logger.info("\n" + "=" * 80)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        logger.info(
            f"Target: {format_target_space_description(target_column, target_mode, target_transform_type)}"
        )
        logger.info(f"")
        logger.info(f"Original Space (Primary):")
        logger.info(
            f"  Training: RMSE={train_metrics['rmse']:.4f} kN, R²={train_metrics['r2']:.4f}, COV={train_metrics.get('cov', 'N/A')}"
        )
        logger.info(
            f"  Test:     RMSE={test_metrics['rmse']:.4f} kN, R²={test_metrics['r2']:.4f}, COV={test_metrics.get('cov', 'N/A')}"
        )
        logger.info(
            "  CV: "
            f"J={cv_results['mean_cv_score']:.4f}, "
            f"RMSE={(cv_results.get('mean_cv_rmse') if cv_results.get('mean_cv_rmse') is not None else float('nan')):.4f}, "
            f"R²={(cv_results.get('mean_cv_r2') if cv_results.get('mean_cv_r2') is not None else float('nan')):.4f}, "
            f"COV={(cv_results.get('mean_cv_cov') if cv_results.get('mean_cv_cov') is not None else float('nan')):.4f}"
        )
        logger.info(f"")
        logger.info(f"Training Space (Reference):")
        logger.info(
            f"  Training RMSE ({training_space_label}): {train_metrics_trans['rmse']:.4f}"
        )
        logger.info(
            f"  Test RMSE ({training_space_label}):     {test_metrics_trans['rmse']:.4f}"
        )
        logger.info(f"")
        logger.info(f"Overfitting Analysis:")
        rmse_ratio_final = test_metrics["rmse"] / train_metrics["rmse"]
        status = "OVERFIT" if rmse_ratio_final > 1.2 else "OK"
        logger.info(f"  Ratio: {rmse_ratio_final:.2f} ({status})")
        if test_regime_metrics:
            logger.info("Regime Worst Cases:")
            for regime_name, regime_result in test_regime_metrics.items():
                worst_group = regime_result.get("worst_rmse_group")
                if worst_group:
                    logger.info(
                        f"  {regime_name}: {worst_group['label']} "
                        f"(RMSE={worst_group['metrics']['rmse']:.4f} kN)"
                    )
        logger.info("=" * 80)

        return {
            "model": model,
            "preprocessor": preprocessor,
            "params_source": params_source,
            "final_model_params": trainer.params.copy(),
            "context_hash": context_hash,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_metrics_trans": train_metrics_trans,
            "test_metrics_trans": test_metrics_trans,
            "train_regime_metrics": train_regime_metrics,
            "test_regime_metrics": test_regime_metrics,
            "target_transform_type": target_transform_type,
            "target_mode": target_mode,
            "cv_results": cv_results,
            "feature_names": feature_names,
            "output_dir": output_dir,
        }

    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train CFST XGBoost Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default configuration
  python train.py

  # Custom configuration file
  python train.py --config config/config.yaml

  # Custom output directory
  python train.py --output my_model_output

  # Custom config and output
  python train.py --config config/config.yaml --output models/cfst_model
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML file (default: config/config.yaml)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory for model and results (default: from config)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Check if config file exists
    if not Path(args.config).exists():
        logger.error(f"Configuration file not found: {args.config}")
        logger.error(
            "Please check the file path or create a config file using config/config.example.yaml"
        )
        sys.exit(1)

    try:
        # Run training pipeline
        train_model(args.config, args.output)
        logger.info("Training completed successfully!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error("Use --verbose for more details")
        if args.verbose:
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
