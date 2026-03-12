import numpy as np
import pandas as pd
import pytest

from src.predictor import Predictor


class StubModel:
    def predict(self, X):
        return X.sum(axis=1).to_numpy(dtype=float)


class ConstantModel:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


class ReportSpaceModel:
    predicts_in_report_space = True

    def predict(self, X):
        return X["a"].to_numpy(dtype=float) + 1.0


class StubPreprocessor:
    def transform(self, X):
        transformed = X.copy()
        transformed["a"] = transformed["a"] * 10
        return transformed


def test_predict_ignores_extra_features_and_applies_preprocessor():
    predictor = Predictor(
        model=StubModel(),
        preprocessor=StubPreprocessor(),
        feature_names=["a", "b"],
    )
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "extra": [99.0, 100.0]})

    predictions = predictor.predict(X)

    assert np.allclose(predictions, np.array([13.0, 24.0]))


def test_predict_raises_when_required_feature_missing():
    predictor = Predictor(model=StubModel(), feature_names=["a", "b"])
    X = pd.DataFrame({"a": [1.0]})

    with pytest.raises(Exception, match="Missing required features"):
        predictor.predict(X)


def test_predict_single_raises_for_multiple_rows():
    predictor = Predictor(model=StubModel(), feature_names=["a", "b"])
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    with pytest.raises(Exception, match="Single prediction expects 1 row"):
        predictor.predict_single(X)


def test_predict_batch_matches_direct_predict():
    predictor = Predictor(model=StubModel(), feature_names=["a", "b"])
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

    direct = predictor.predict(X)
    batched = predictor.predict_batch(X, batch_size=2)

    assert np.allclose(direct, batched)


def test_predict_restores_nexp_for_psi_target_without_requiring_nexp_input():
    predictor = Predictor(
        model=ConstantModel(np.log(0.8)),
        feature_names=["As (mm^2)", "Ac (mm^2)", "fy (MPa)", "fc (MPa)", "Npl (kN)"],
        metadata={
            "target_mode": "psi_over_npl",
            "report_target_column": "Nexp (kN)",
            "target_transform": {
                "enabled": True,
                "type": "log",
                "mode": "psi_over_npl",
                "original_column": "Nexp (kN)",
            },
        },
    )
    X = pd.DataFrame(
        {
            "As (mm^2)": [1000.0, 1200.0],
            "Ac (mm^2)": [2000.0, 1800.0],
            "fy (MPa)": [300.0, 320.0],
            "fc (MPa)": [40.0, 50.0],
        }
    )

    predictions = predictor.predict(X)

    expected_npl = np.array([(1000.0 * 300.0 + 2000.0 * 40.0) / 1000.0, (1200.0 * 320.0 + 1800.0 * 50.0) / 1000.0])
    assert np.allclose(predictions, 0.8 * expected_npl)


def test_predict_skips_restoration_when_model_outputs_report_space():
    predictor = Predictor(
        model=ReportSpaceModel(),
        feature_names=["a", "b"],
        metadata={
            "target_mode": "raw",
            "target_transform": {
                "enabled": True,
                "type": "log",
                "mode": "raw",
                "original_column": "Nexp (kN)",
            },
            "prediction_output_in_report_space": True,
        },
    )
    X = pd.DataFrame({"a": [2.0, 5.0], "b": [10.0, 20.0]})

    predictions = predictor.predict(X)

    assert np.allclose(predictions, np.array([3.0, 6.0]))
