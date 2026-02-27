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
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.logger import setup_logger
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model_trainer import ModelTrainer
from src.evaluator import Evaluator
from src.utils.model_utils import save_model
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


def load_config(config_path: str):
    """Load configuration from YAML file."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


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
    target_transform_type: Optional[str],
    columns_to_drop: List[str],
) -> Dict[str, Any]:
    """
    Build deterministic training context for artifact compatibility checks.
    """
    data_sha256 = _file_sha256(data_file_path)
    context_payload = {
        "data_file": str(Path(data_file_path).resolve()),
        "data_sha256": data_sha256,
        "target_column": target_column,
        "target_transform_type": target_transform_type or "none",
        "columns_to_drop": sorted(columns_to_drop),
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


def train_model(config_path: str, output_dir: str = None) -> dict:
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
        data_config = config.get("data", {})
        model_config = config.get("model", {})
        cv_config = config.get("cv", {})
        output_config = config.get("paths", {})

        data_path = data_config.get("file_path")
        target_column = data_config.get("target_column", "K")
        columns_to_drop = data_config.get("columns_to_drop", [])
        if not data_path:
            raise ValueError("config.data.file_path is required")

        # Set output directory
        if output_dir is None:
            output_dir = output_config.get("output_dir", "output")

        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Data file: {data_path}")
        logger.info(f"Target column: {target_column}")
        logger.info(f"Columns to drop: {columns_to_drop}")

        # Read target transform configuration
        target_transform_config = data_config.get("target_transform", {})
        target_transform_type = (
            target_transform_config.get("type", None)
            if target_transform_config.get("enabled", False)
            else None
        )

        if target_transform_type:
            logger.info(f"Target transform enabled: {target_transform_type}")

        training_context = build_training_context(
            data_file_path=data_path,
            target_column=target_column,
            target_transform_type=target_transform_type,
            columns_to_drop=columns_to_drop,
        )
        context_hash = training_context["context_hash"]
        study_name = build_study_name(data_path, context_hash)
        logger.info(f"Training context hash: {context_hash}")
        logger.info(f"Optuna study name: {study_name}")

        # Step 2: Load data
        logger.info("\nStep 2: Loading data...")
        data_loader = DataLoader(required_columns=[target_column])

        features, target_transformed = data_loader.load_data(
            data_path, target_column, target_transform=target_transform_type
        )

        # Save original target values (for inverse transform evaluation)
        target_raw = data_loader.target_raw

        logger.info(
            f"Data loaded: {len(features)} samples, {len(features.columns)} features"
        )

        # Step 2.5: Split data into train/test sets (FIXES DATA LEAKAGE)
        logger.info("\nStep 2.5: Splitting data into train/test sets...")
        test_size = data_config.get("test_size", 0.2)
        random_state = data_config.get("random_state", 42)

        # Split transformed target (for model training)
        X_train_full, X_test, y_train_trans_full, y_test_trans = train_test_split(
            features, target_transformed, test_size=test_size, random_state=random_state
        )

        # Also split original target values (for original space evaluation)
        _, _, y_train_orig_full, y_test_orig = train_test_split(
            features, target_raw, test_size=test_size, random_state=random_state
        )

        logger.info(
            f"Training set: {len(X_train_full)} samples ({(1 - test_size) * 100:.0f}%)"
        )
        logger.info(f"Test set: {len(X_test)} samples ({test_size * 100:.0f}%)")

        # Step 2.6: Split training data into train/validation sets for early stopping
        validation_size = model_config.get("validation_size", 0.1)
        X_val = y_val_trans = y_val_orig = None
        if validation_size and validation_size > 0:
            logger.info(
                "\nStep 2.6: Splitting training data into train/validation sets..."
            )
            X_train, X_val, y_train_trans, y_val_trans, y_train_orig, y_val_orig = (
                train_test_split(
                    X_train_full,
                    y_train_trans_full,
                    y_train_orig_full,
                    test_size=validation_size,
                    random_state=random_state,
                )
            )
            logger.info(
                f"Training subset: {len(X_train)} samples ({(1 - validation_size) * 100:.0f}%)"
            )
            logger.info(
                f"Validation set: {len(X_val)} samples ({validation_size * 100:.0f}%)"
            )
        else:
            X_train = X_train_full
            y_train_trans = y_train_trans_full
            y_train_orig = y_train_orig_full

        # Step 3: Preprocess data (FIT ON TRAINING DATA ONLY - CRITICAL FIX)
        logger.info("\nStep 3: Preprocessing data...")
        preprocessor = Preprocessor(columns_to_drop=columns_to_drop)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        X_val_processed = preprocessor.transform(X_val) if X_val is not None else None
        logger.info(
            f"Preprocessing completed: {len(X_train_processed.columns)} features remaining"
        )

        # Get remaining feature names
        feature_names = preprocessor.get_remaining_features()
        logger.info(f"Remaining features: {feature_names}")

        # Check for missing values
        missing_info_train = preprocessor.check_missing_values(X_train_processed)
        missing_info_test = preprocessor.check_missing_values(X_test_processed)
        if missing_info_train or missing_info_test:
            logger.warning(
                f"Found missing values - Train: {missing_info_train}, Test: {missing_info_test}"
            )
        else:
            logger.info("No missing values found in train or test sets")

        # Step 4: Train model
        logger.info("\nStep 4: Training XGBoost model...")

        # Initialize trainer with parameters
        use_optuna = model_config.get("use_optuna", False)
        n_trials = model_config.get("n_trials", 100)
        optuna_timeout = model_config.get("optuna_timeout", 3600)
        optuna_storage_url = model_config.get(
            "optuna_storage_url", "sqlite:///logs/optuna_study.db"
        )
        best_params_path = model_config.get("best_params_path", "logs/best_params.json")
        cv_n_splits = get_cv_n_splits(cv_config)

        # Prepare XGBoost parameters (strictly from model.params)
        xgb_params = build_xgb_params(model_config)

        early_stopping_rounds = model_config.get("early_stopping_rounds")
        eval_metric = model_config.get("eval_metric")

        trainer = ModelTrainer(
            params=xgb_params,
            use_optuna=use_optuna,
            n_trials=n_trials,
            optuna_timeout=optuna_timeout,
            best_params_path=best_params_path,
            expected_context_hash=context_hash,
        )

        eval_set = (
            [(X_val_processed, y_val_trans)] if X_val_processed is not None else None
        )
        params_source = (
            "best_params_file" if trainer.loaded_best_params else "config_model_params"
        )
        optuna_run_info = None

        # Optional: Hyperparameter optimization, then retrain final model with best params
        if use_optuna:
            logger.info("Starting Optuna hyperparameter optimization...")
            opt_results = trainer.optimize_hyperparameters(
                X_train_processed,
                y_train_trans,  # Training data only
                cv=cv_n_splits,
                study_name=study_name,
                storage_url=optuna_storage_url,
                best_params_output_path=best_params_path,
                run_context=training_context,
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
                "best_params": opt_results["best_params"],
            }

        model = trainer.train(
            X_train_processed,
            y_train_trans,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            eval_metric=eval_metric,
        )
        logger.info(f"Final model training completed on ln({target_column}) target")

        # Step 5: Cross-validation on training data only
        logger.info("\nStep 5: Performing cross-validation on training data...")
        cv_results = trainer.cross_validate(
            X_train_processed,
            y_train_trans,  # Training data only
            cv=cv_n_splits,
        )
        logger.info(
            f"Cross-validation RMSE: {cv_results['mean_cv_score']:.4f} (+/- {cv_results['std_cv_score']:.4f})"
        )

        # Step 6: Evaluate model on BOTH training and test sets
        logger.info("\nStep 6: Evaluating model...")
        evaluator = Evaluator()

        # Make predictions on BOTH sets (in transformed space)
        from src.predictor import Predictor

        predictor = Predictor(model, preprocessor, feature_names)
        y_train_pred_trans = predictor.predict(X_train_full)
        y_test_pred_trans = predictor.predict(X_test)

        # Apply inverse transform to get back to original space
        if target_transform_type == "log":
            y_train_pred_orig = np.exp(y_train_pred_trans)
            y_test_pred_orig = np.exp(y_test_pred_trans)
            logger.info("Applied exp() inverse transform to predictions")
        else:
            y_train_pred_orig = y_train_pred_trans
            y_test_pred_orig = y_test_pred_trans

        # Calculate metrics in ORIGINAL space (recommended - true application scenario)
        train_metrics = evaluator.calculate_metrics(
            y_train_orig_full, y_train_pred_orig
        )
        test_metrics = evaluator.calculate_metrics(y_test_orig, y_test_pred_orig)

        # Also calculate metrics in transformed space (for reference)
        train_metrics_trans = evaluator.calculate_metrics(
            y_train_trans_full, y_train_pred_trans
        )
        test_metrics_trans = evaluator.calculate_metrics(
            y_test_trans, y_test_pred_trans
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

        # Log transformed space metrics (REFERENCE)
        logger.info(f"\n--- Transformed Space Metrics (Reference) ---")
        logger.info(f"Training RMSE (ln space): {train_metrics_trans['rmse']:.4f}")
        logger.info(f"Test RMSE (ln space): {test_metrics_trans['rmse']:.4f}")
        logger.info(
            f"Train/Test ratio (ln space): {test_metrics_trans['rmse'] / train_metrics_trans['rmse']:.2f}"
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
            y_train_orig_full,
            y_train_pred_orig,
            model,
            feature_names,
            str(plots_dir),
            "xgboost_model_train",
        )

        # Test set plots (PRIMARY - shows true generalization)
        create_evaluation_dashboard(
            y_test_orig,
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

        # Save model with metadata
        metadata = {
            "config": config,
            "context_hash": context_hash,
            "training_context": training_context,
            "params_source": params_source,
            "final_model_params": trainer.params.copy(),
            "optuna_run_info": optuna_run_info,
            "target_transform": {
                "enabled": target_transform_type is not None,
                "type": target_transform_type,
                "original_column": target_column,
            },
            "train_metrics_original_space": train_metrics,  # PRIMARY
            "test_metrics_original_space": test_metrics,  # PRIMARY
            "train_metrics_transformed_space": train_metrics_trans,  # REFERENCE
            "test_metrics_transformed_space": test_metrics_trans,  # REFERENCE
            "cross_validation_results": cv_results,
            "feature_names": feature_names,
            "n_train_samples": len(X_train_full),
            "n_test_samples": len(X_test),
            "n_features": len(feature_names),
            "test_size": test_size,
            "n_train_fit_samples": len(X_train),
            "n_validation_samples": len(X_val) if X_val is not None else 0,
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
                "target_transform": {
                    "enabled": target_transform_type is not None,
                    "type": target_transform_type,
                    "original_column": target_column,
                },
                "train_metrics_original_space": train_metrics,  # PRIMARY
                "test_metrics_original_space": test_metrics,  # PRIMARY
                "train_metrics_transformed_space": train_metrics_trans,  # REFERENCE
                "test_metrics_transformed_space": test_metrics_trans,  # REFERENCE
                "cv_results": serializable_cv_results,
                "data_split": {
                    "n_train": len(X_train_full),
                    "n_test": len(X_test),
                    "test_size": test_size,
                    "n_train_fit": len(X_train),
                    "n_validation": len(X_val) if X_val is not None else 0,
                },
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
        logger.info(f"Target: ln({target_column}) → exp() → {target_column}")
        logger.info(f"")
        logger.info(f"Original Space (Primary):")
        logger.info(
            f"  Training: RMSE={train_metrics['rmse']:.4f} kN, R²={train_metrics['r2']:.4f}, COV={train_metrics.get('cov', 'N/A')}"
        )
        logger.info(
            f"  Test:     RMSE={test_metrics['rmse']:.4f} kN, R²={test_metrics['r2']:.4f}, COV={test_metrics.get('cov', 'N/A')}"
        )
        logger.info(f"  CV:       RMSE={-cv_results['mean_cv_score']:.4f}")
        logger.info(f"")
        logger.info(f"Transformed Space (Reference):")
        logger.info(f"  Training RMSE (ln): {train_metrics_trans['rmse']:.4f}")
        logger.info(f"  Test RMSE (ln):     {test_metrics_trans['rmse']:.4f}")
        logger.info(f"")
        logger.info(f"Overfitting Analysis:")
        rmse_ratio_final = test_metrics["rmse"] / train_metrics["rmse"]
        status = "OVERFIT" if rmse_ratio_final > 1.2 else "OK"
        logger.info(f"  Ratio: {rmse_ratio_final:.2f} ({status})")
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
            "target_transform_type": target_transform_type,
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
