"""
Two-stage model wrapper for high-Npl tail residual correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.domain_features import restore_report_target

VALID_TAIL_CORRECTION_MODES = {"additive", "ratio", "logratio"}
VALID_TAIL_GATING_MODES = {"hard", "soft_linear"}

GLOBAL_PREDICTION_FEATURE_NAME = "__global_pred_report__"
LOG_GLOBAL_PREDICTION_FEATURE_NAME = "__log_global_pred_report__"


def augment_tail_features(
    X: pd.DataFrame,
    *,
    global_prediction_report: Optional[np.ndarray] = None,
    append_global_prediction_features: bool = False,
) -> pd.DataFrame:
    """Optionally append first-stage report-space predictions to tail-model features."""
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Tail feature augmentation expects a pandas DataFrame")

    if not append_global_prediction_features:
        return X.copy()

    if global_prediction_report is None:
        raise ValueError(
            "global_prediction_report is required when append_global_prediction_features=True"
        )

    global_prediction_array = np.asarray(global_prediction_report, dtype=float).reshape(-1)
    if len(global_prediction_array) != len(X):
        raise ValueError(
            "global_prediction_report length must match the feature frame length"
        )

    augmented = X.copy()
    augmented[GLOBAL_PREDICTION_FEATURE_NAME] = global_prediction_array
    augmented[LOG_GLOBAL_PREDICTION_FEATURE_NAME] = np.log(
        np.clip(global_prediction_array, 1e-9, None)
    )
    return augmented


def build_tail_training_target(
    y_true_report: np.ndarray,
    global_prediction_report: np.ndarray,
    *,
    correction_mode: str,
) -> np.ndarray:
    """Build the second-stage learning target from report-space predictions."""
    normalized_mode = str(correction_mode).strip().lower()
    if normalized_mode not in VALID_TAIL_CORRECTION_MODES:
        raise ValueError(
            f"Unsupported tail correction mode '{correction_mode}'. "
            f"Expected one of {sorted(VALID_TAIL_CORRECTION_MODES)}."
        )

    y_true_array = np.asarray(y_true_report, dtype=float).reshape(-1)
    global_prediction_array = np.asarray(global_prediction_report, dtype=float).reshape(-1)
    if len(y_true_array) != len(global_prediction_array):
        raise ValueError("Tail training target inputs must have matching lengths")

    if normalized_mode == "additive":
        return y_true_array - global_prediction_array
    if normalized_mode == "ratio":
        return y_true_array / np.clip(global_prediction_array, 1e-9, None) - 1.0
    return np.log(
        np.clip(y_true_array, 1e-9, None) / np.clip(global_prediction_array, 1e-9, None)
    )


def compute_tail_gate_weights(
    feature_values: np.ndarray,
    *,
    threshold: float,
    gating_mode: str,
    upper_threshold: Optional[float] = None,
) -> np.ndarray:
    """Compute hard or smooth gate weights for tail correction."""
    normalized_mode = str(gating_mode).strip().lower()
    if normalized_mode not in VALID_TAIL_GATING_MODES:
        raise ValueError(
            f"Unsupported tail gating mode '{gating_mode}'. "
            f"Expected one of {sorted(VALID_TAIL_GATING_MODES)}."
        )

    values = np.asarray(feature_values, dtype=float).reshape(-1)
    if normalized_mode == "hard":
        return np.ones(len(values), dtype=float)

    if upper_threshold is None:
        raise ValueError("upper_threshold is required when gating_mode='soft_linear'")
    if float(upper_threshold) <= float(threshold):
        raise ValueError("upper_threshold must be greater than threshold for soft gating")

    denominator = float(upper_threshold) - float(threshold)
    return np.clip((values - float(threshold)) / denominator, 0.0, 1.0)


def apply_tail_correction(
    base_prediction_report: np.ndarray,
    tail_output: np.ndarray,
    *,
    correction_mode: str,
    correction_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply tail-model output to report-space base predictions."""
    normalized_mode = str(correction_mode).strip().lower()
    if normalized_mode not in VALID_TAIL_CORRECTION_MODES:
        raise ValueError(
            f"Unsupported tail correction mode '{correction_mode}'. "
            f"Expected one of {sorted(VALID_TAIL_CORRECTION_MODES)}."
        )

    base_prediction_array = np.asarray(base_prediction_report, dtype=float).reshape(-1)
    tail_output_array = np.asarray(tail_output, dtype=float).reshape(-1)
    if len(base_prediction_array) != len(tail_output_array):
        raise ValueError("Tail correction inputs must have matching lengths")

    if correction_weight is None:
        correction_weight_array = np.ones(len(base_prediction_array), dtype=float)
    else:
        correction_weight_array = np.asarray(correction_weight, dtype=float).reshape(-1)
        if len(correction_weight_array) != len(base_prediction_array):
            raise ValueError("correction_weight length must match prediction length")

    if normalized_mode == "additive":
        return base_prediction_array + correction_weight_array * tail_output_array
    if normalized_mode == "ratio":
        return base_prediction_array * (1.0 + correction_weight_array * tail_output_array)
    return base_prediction_array * np.exp(correction_weight_array * tail_output_array)


@dataclass
class TailResidualEnsemble:
    """
    Combine a global model with a high-tail residual correction model.

    The global model predicts in the configured training space. The tail model predicts
    additive residuals in the reported Nexp-space for samples above the configured tail
    threshold, using the same preprocessed feature frame as the global model.
    """

    global_model: Any
    tail_model: Any
    tail_feature_name: str
    tail_threshold: float
    target_mode: str
    target_transform_type: Optional[str]
    correction_mode: str = "additive"
    append_global_prediction_features: bool = False
    gating_mode: str = "hard"
    tail_gate_upper_threshold: Optional[float] = None
    report_prediction_min: Optional[float] = 0.0
    metadata: Optional[Dict[str, Any]] = None

    predicts_in_report_space: bool = True

    def __post_init__(self) -> None:
        if not hasattr(self.global_model, "predict"):
            raise AttributeError("global_model must implement predict")
        if not hasattr(self.tail_model, "predict"):
            raise AttributeError("tail_model must implement predict")
        self.correction_mode = str(self.correction_mode).strip().lower()
        if self.correction_mode not in VALID_TAIL_CORRECTION_MODES:
            raise ValueError(
                f"Unsupported tail correction mode '{self.correction_mode}'. "
                f"Expected one of {sorted(VALID_TAIL_CORRECTION_MODES)}."
            )
        self.gating_mode = str(self.gating_mode).strip().lower()
        if self.gating_mode not in VALID_TAIL_GATING_MODES:
            raise ValueError(
                f"Unsupported tail gating mode '{self.gating_mode}'. "
                f"Expected one of {sorted(VALID_TAIL_GATING_MODES)}."
            )
        if (
            self.gating_mode == "soft_linear"
            and self.tail_gate_upper_threshold is None
        ):
            raise ValueError(
                "tail_gate_upper_threshold is required when gating_mode='soft_linear'"
            )

    @property
    def feature_importances_(self) -> np.ndarray:
        if not hasattr(self.global_model, "feature_importances_"):
            raise AttributeError("global_model does not expose feature_importances_")
        return np.asarray(self.global_model.feature_importances_, dtype=float)

    def _tail_mask(self, X: pd.DataFrame) -> np.ndarray:
        if self.tail_feature_name not in X.columns:
            raise ValueError(
                f"Tail feature '{self.tail_feature_name}' is missing from prediction input"
            )
        values = X[self.tail_feature_name].to_numpy(dtype=float)
        return values >= float(self.tail_threshold)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("TailResidualEnsemble expects a pandas DataFrame")

        global_prediction_training_space = np.asarray(
            self.global_model.predict(X),
            dtype=float,
        ).reshape(-1)
        global_prediction_report_space = restore_report_target(
            global_prediction_training_space,
            target_mode=self.target_mode,
            target_transform_type=self.target_transform_type,
            reference_features=X,
        )

        final_prediction = global_prediction_report_space.copy()
        tail_mask = self._tail_mask(X)
        if np.any(tail_mask):
            tail_features = augment_tail_features(
                X.loc[tail_mask],
                global_prediction_report=global_prediction_report_space[tail_mask],
                append_global_prediction_features=self.append_global_prediction_features,
            )
            tail_output = np.asarray(
                self.tail_model.predict(tail_features),
                dtype=float,
            ).reshape(-1)
            tail_weights = compute_tail_gate_weights(
                X.loc[tail_mask, self.tail_feature_name].to_numpy(dtype=float),
                threshold=self.tail_threshold,
                gating_mode=self.gating_mode,
                upper_threshold=self.tail_gate_upper_threshold,
            )
            final_prediction[tail_mask] = apply_tail_correction(
                final_prediction[tail_mask],
                tail_output,
                correction_mode=self.correction_mode,
                correction_weight=tail_weights,
            )

        if self.report_prediction_min is not None:
            final_prediction = np.maximum(
                final_prediction,
                float(self.report_prediction_min),
            )

        return final_prediction
