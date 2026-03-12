"""
Two-stage model wrapper for high-Npl tail residual correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.domain_features import restore_report_target


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
    report_prediction_min: Optional[float] = 0.0
    metadata: Optional[Dict[str, Any]] = None

    predicts_in_report_space: bool = True

    def __post_init__(self) -> None:
        if not hasattr(self.global_model, "predict"):
            raise AttributeError("global_model must implement predict")
        if not hasattr(self.tail_model, "predict"):
            raise AttributeError("tail_model must implement predict")

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
            tail_residual = np.asarray(
                self.tail_model.predict(X.loc[tail_mask]),
                dtype=float,
            ).reshape(-1)
            final_prediction[tail_mask] = final_prediction[tail_mask] + tail_residual

        if self.report_prediction_min is not None:
            final_prediction = np.maximum(
                final_prediction,
                float(self.report_prediction_min),
            )

        return final_prediction
