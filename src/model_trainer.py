"""
Model Trainer Module for CFST XGBoost Pipeline

This module handles XGBoost model training, cross-validation, and hyperparameter optimization.
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any, cast
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from optuna.trial import Trial
import time
import json
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.model_utils import load_best_params, save_best_params

logger = get_logger(__name__)


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
        """
        self.params = params or self._get_default_params()
        self.use_optuna = use_optuna
        self.n_trials = n_trials
        self.optuna_timeout = optuna_timeout
        self.best_params_path = best_params_path
        self.expected_context_hash = expected_context_hash
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

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
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

            # Create model with current parameters
            # Note: XGBoost >= 2.0 requires early_stopping_rounds and eval_metric in constructor, not fit()
            model_params = self.params.copy()
            if early_stopping_rounds is not None and eval_set_to_use:
                model_params["early_stopping_rounds"] = early_stopping_rounds
            if eval_metric is not None:
                model_params["eval_metric"] = eval_metric
            self.model = xgb.XGBRegressor(**model_params)

            # Train model
            start_time = time.time()
            fit_kwargs: Dict[str, Any] = {"verbose": False}
            if eval_set_to_use:
                fit_kwargs = {"verbose": False, "eval_set": eval_set_to_use}

            self.model.fit(X_train, y_train, **fit_kwargs)

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
        cv: int = 5,
        scoring: str = "neg_root_mean_squared_error",
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.

        Args:
            X: Features DataFrame
            y: Target Series
            cv: Number of folds
            scoring: Scoring metric (default: negative RMSE)

        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Starting {cv}-fold cross-validation")

        # Create a fresh model for cross-validation WITHOUT early_stopping_rounds
        # sklearn's cross_val_score doesn't pass eval_set, which XGBoost requires
        # when early_stopping_rounds is set
        cv_params = self.params.copy()
        # Remove early stopping related parameters that require eval_set
        cv_params.pop("early_stopping_rounds", None)
        cv_params.pop("eval_metric", None)
        cv_model = xgb.XGBRegressor(**cv_params)

        # Define scoring function
        if scoring == "neg_root_mean_squared_error":
            scorer = make_scorer(
                lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred))
            )
        else:
            scorer = scoring

        # Perform cross-validation
        # Use n_jobs=1 for GPU to avoid resource conflicts, n_jobs=-1 for CPU
        cv_n_jobs = 1 if self.params.get("device") == "cuda" else -1
        start_time = time.time()
        cv_scores = cross_val_score(
            cv_model, X, y, cv=cv, scoring=scorer, n_jobs=cv_n_jobs, verbose=0
        )
        cv_time = time.time() - start_time

        # Calculate metrics
        results = {
            "cv_scores": cv_scores,
            "mean_cv_score": np.mean(cv_scores),
            "std_cv_score": np.std(cv_scores),
            "max_cv_score": np.max(cv_scores),
            "min_cv_score": np.min(cv_scores),
            "cv_time": cv_time,
            "n_folds": cv,
        }

        logger.info(f"Cross-validation completed in {cv_time:.2f} seconds")
        logger.info(f"CV scores: {cv_scores}")
        logger.info(
            f"Mean CV score: {results['mean_cv_score']:.4f} (+/- {results['std_cv_score']:.4f})"
        )

        return results

    def optimize_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        n_trials: Optional[int] = None,
        study_name: str = "xgboost_optimization",
        storage_url: str = "sqlite:///logs/optuna_study.db",
        best_params_output_path: str = "logs/best_params.json",
        run_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            X: Features DataFrame
            y: Target Series
            cv: Number of folds for cross-validation
            n_trials: Number of optimization trials (uses instance default if None)
            study_name: Optuna study name
            storage_url: Optuna storage URL
            best_params_output_path: Output path for best parameters snapshot
            run_context: Optional context metadata (context_hash, data_file, etc.)

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

        # Create logs directory if it doesn't exist
        Path("logs").mkdir(parents=True, exist_ok=True)

        context_hash = run_context.get("context_hash") if run_context else None
        data_file = run_context.get("data_file") if run_context else None

        # Define objective function for Optuna
        def objective(trial: Trial) -> float:
            # Search space tuned for larger dataset with long-tail target distribution.
            params = {
                "objective": self.params.get("objective", "reg:squarederror"),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.015, 0.12, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 800, 4000),
                "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.95),
                "min_child_weight": trial.suggest_int("min_child_weight", 4, 30),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
                # Fixed parameters
                "random_state": self.params.get("random_state", 42),
                "tree_method": self.params.get("tree_method", "hist"),
                "device": self.params.get("device", "cpu"),
                "n_jobs": self.params.get("n_jobs", -1),
            }

            # Use cross-validation for evaluation
            kf = KFold(
                n_splits=cv,
                shuffle=True,
                random_state=self.params.get("random_state", 42),
            )
            scores: List[float] = []

            for train_idx, val_idx in kf.split(X):
                X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

                model_fit = xgb.XGBRegressor(**params)
                model_fit.fit(X_train_cv, y_train_cv, verbose=False)

                # Predict and calculate RMSE
                y_pred = model_fit.predict(X_val_cv)
                rmse = float(np.sqrt(mean_squared_error(y_val_cv, y_pred)))
                scores.append(rmse)

            return float(np.mean(scores))

        # Create Optuna study with persistent storage
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
        )
        n_trials_before = len(study.trials)

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
        logger.info(f"Best RMSE: {best_score:.4f}")
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
        )

        # Update model parameters with best found parameters
        self.params.update(best_params)
        logger.info(f"Updated model parameters with best parameters")

        # Return optimization results
        results = {
            "best_params": best_params,
            "best_score": best_score,
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
