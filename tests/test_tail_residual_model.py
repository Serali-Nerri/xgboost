import numpy as np
import pandas as pd

from src.tail_residual_model import TailResidualEnsemble


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
