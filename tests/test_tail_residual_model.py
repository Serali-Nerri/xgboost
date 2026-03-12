import numpy as np
import pandas as pd

from src.tail_residual_model import (
    GLOBAL_PREDICTION_FEATURE_NAME,
    LOG_GLOBAL_PREDICTION_FEATURE_NAME,
    TailResidualEnsemble,
    augment_tail_features,
    build_tail_training_target,
)


class ConstantModel:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def test_tail_residual_ensemble_adds_residual_only_above_threshold():
    ensemble = TailResidualEnsemble(
        global_model=ConstantModel(np.log(100.0)),
        tail_model=ConstantModel(25.0),
        tail_feature_name="Npl (kN)",
        tail_threshold=500.0,
        target_mode="raw",
        target_transform_type="log",
        report_prediction_min=0.0,
    )
    X = pd.DataFrame(
        {
            "Npl (kN)": [300.0, 700.0],
            "feat": [1.0, 2.0],
        }
    )

    predictions = ensemble.predict(X)

    assert np.allclose(predictions, np.array([100.0, 125.0]))


def test_tail_residual_ensemble_exposes_global_feature_importances():
    class GlobalModel(ConstantModel):
        feature_importances_ = np.array([0.7, 0.3], dtype=float)

    ensemble = TailResidualEnsemble(
        global_model=GlobalModel(np.log(50.0)),
        tail_model=ConstantModel(5.0),
        tail_feature_name="Npl (kN)",
        tail_threshold=100.0,
        target_mode="raw",
        target_transform_type="log",
    )

    assert np.allclose(ensemble.feature_importances_, np.array([0.7, 0.3]))


def test_augment_tail_features_appends_global_prediction_columns():
    X = pd.DataFrame({"Npl (kN)": [100.0, 200.0], "feat": [1.0, 2.0]})

    augmented = augment_tail_features(
        X,
        global_prediction_report=np.array([50.0, 80.0]),
        append_global_prediction_features=True,
    )

    assert np.allclose(
        augmented[GLOBAL_PREDICTION_FEATURE_NAME].to_numpy(dtype=float),
        np.array([50.0, 80.0]),
    )
    assert np.allclose(
        augmented[LOG_GLOBAL_PREDICTION_FEATURE_NAME].to_numpy(dtype=float),
        np.log(np.array([50.0, 80.0])),
    )


def test_build_tail_training_target_supports_logratio():
    target = build_tail_training_target(
        np.array([120.0, 80.0]),
        np.array([100.0, 100.0]),
        correction_mode="logratio",
    )

    assert np.allclose(target, np.log(np.array([1.2, 0.8])))


def test_tail_residual_ensemble_supports_logratio_soft_gating_and_global_pred_features():
    ensemble = TailResidualEnsemble(
        global_model=ConstantModel(np.log(100.0)),
        tail_model=ConstantModel(np.log(1.21)),
        tail_feature_name="Npl (kN)",
        tail_threshold=500.0,
        tail_gate_upper_threshold=900.0,
        target_mode="raw",
        target_transform_type="log",
        correction_mode="logratio",
        append_global_prediction_features=True,
        gating_mode="soft_linear",
        report_prediction_min=0.0,
    )
    X = pd.DataFrame(
        {
            "Npl (kN)": [300.0, 700.0, 900.0],
            "feat": [1.0, 2.0, 3.0],
        }
    )

    predictions = ensemble.predict(X)

    assert np.allclose(predictions, np.array([100.0, 110.0, 121.0]))
