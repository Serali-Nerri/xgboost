import numpy as np
import pandas as pd
import pytest

from src.evaluator import Evaluator


def test_calculate_metrics_raises_on_length_mismatch():
    evaluator = Evaluator()

    with pytest.raises(ValueError, match="Length mismatch"):
        evaluator.calculate_metrics(pd.Series([1.0, 2.0]), np.array([1.0]))


def test_calculate_metrics_returns_none_mape_for_all_zero_targets():
    evaluator = Evaluator()
    metrics = evaluator.calculate_metrics(pd.Series([0.0, 0.0]), np.array([0.0, 1.0]))

    assert metrics["mape"] is None
    assert metrics["n_samples"] == 2


def test_calculate_metrics_returns_expected_keys_for_normal_input():
    evaluator = Evaluator()
    metrics = evaluator.calculate_metrics(pd.Series([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 33.0]))

    for key in ["rmse", "mae", "r2", "mse", "max_error", "cov", "n_samples"]:
        assert key in metrics
    assert metrics["n_samples"] == 3


def test_regime_schema_reuses_train_quantile_edges_on_test_split():
    evaluator = Evaluator()
    regime_config = {
        "enabled": True,
        "sort_metric": "cov",
        "regimes": [
            {
                "name": "scale_npl",
                "mode": "train_quantile",
                "source": "feature",
                "column": "Npl (kN)",
                "bins": 3,
            },
            {
                "name": "axiality",
                "mode": "categorical",
                "source": "feature",
                "column": "axial_flag",
            },
        ],
    }
    X_train = pd.DataFrame(
        {
            "Npl (kN)": [100.0, 120.0, 200.0, 240.0, 320.0, 360.0],
            "axial_flag": ["axial", "axial", "eccentric", "eccentric", "axial", "eccentric"],
        }
    )
    y_train = pd.Series([100.0, 118.0, 210.0, 250.0, 330.0, 350.0])
    y_pred_train = np.array([105.0, 120.0, 205.0, 255.0, 325.0, 345.0])
    schema = evaluator.fit_regime_schema(y_train, X_train, regime_config)

    X_test = pd.DataFrame(
        {
            "Npl (kN)": [110.0, 230.0, 340.0],
            "axial_flag": ["axial", "eccentric", "axial"],
        }
    )
    y_test = pd.Series([108.0, 225.0, 335.0])
    y_pred_test = np.array([109.0, 228.0, 330.0])
    results = evaluator.calculate_regime_metrics(y_test, y_pred_test, X_test, schema)

    scale_ranges = results["scale_npl"]["ranges"]
    assert len(scale_ranges) == 3
    assert results["axiality"]["groups"][0]["metrics"]["mean_ratio"] is not None
    assert "worst_rmse_group" in results["scale_npl"]
    assert "worst_cov_group" in results["scale_npl"]
    assert "worst_r2_group" in results["scale_npl"]


def test_regime_schema_supports_fixed_bins_and_sorts_groups_by_cov():
    evaluator = Evaluator()
    regime_config = {
        "enabled": True,
        "reference_split": "train",
        "sort_metric": "cov",
        "regimes": [
            {
                "name": "eccentricity_severity",
                "mode": "fixed_bins",
                "source": "feature",
                "column": "e/h",
                "edges": [0.0, 0.0, 0.1, 0.3, float("inf")],
                "labels": ["axial", "small_ecc", "moderate_ecc", "large_ecc"],
            }
        ],
    }
    X_train = pd.DataFrame({"e/h": [0.0, 0.0, 0.05, 0.06, 0.2, 0.22, 0.4, 0.45]})
    y_train = pd.Series([100.0, 100.0, 110.0, 112.0, 130.0, 128.0, 160.0, 162.0])
    schema = evaluator.fit_regime_schema(y_train, X_train, regime_config)

    y_pred = np.array([100.0, 100.0, 120.0, 100.0, 160.0, 110.0, 170.0, 120.0])
    results = evaluator.calculate_regime_metrics(y_train, y_pred, X_train, schema)

    groups = results["eccentricity_severity"]["groups"]
    cov_values = [group["metrics"]["cov"] for group in groups]

    assert [item["label"] for item in results["eccentricity_severity"]["ranges"]] == [
        "axial",
        "small_ecc",
        "moderate_ecc",
        "large_ecc",
    ]
    assert groups[0]["label"] == "large_ecc"
    assert cov_values[0] >= cov_values[-1]
    assert results["eccentricity_severity"]["worst_rmse_group"]["label"] == "large_ecc"
