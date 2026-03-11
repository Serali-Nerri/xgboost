import re

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader


def test_load_data_applies_log_transform_and_preserves_raw_target(tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({
        "feat": [1.0, 2.0, 3.0],
        "Nexp (kN)": [10.0, 20.0, 40.0],
    })
    df.to_csv(csv_path, index=False)

    loader = DataLoader(required_columns=["Nexp (kN)"])
    features, target = loader.load_data(str(csv_path), "Nexp (kN)", target_transform="log")

    expected = np.log(df["Nexp (kN)"].to_numpy(dtype=float))

    assert list(features.columns) == ["feat"]
    assert np.allclose(target.to_numpy(dtype=float), expected)
    assert loader.target_raw is not None
    assert np.allclose(loader.target_raw.to_numpy(dtype=float), df["Nexp (kN)"].to_numpy(dtype=float))


def test_load_data_applies_sqrt_transform(tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({
        "feat": [1.0, 2.0],
        "Nexp (kN)": [9.0, 16.0],
    })
    df.to_csv(csv_path, index=False)

    loader = DataLoader(required_columns=["Nexp (kN)"])
    _, target = loader.load_data(str(csv_path), "Nexp (kN)", target_transform="sqrt")

    assert np.allclose(
        target.to_numpy(dtype=float),
        np.sqrt(df["Nexp (kN)"].to_numpy(dtype=float)),
    )


def test_load_data_without_transform_returns_original_target(tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({
        "feat": [1.0, 2.0],
        "Nexp (kN)": [11.0, 13.0],
    })
    df.to_csv(csv_path, index=False)

    loader = DataLoader(required_columns=["Nexp (kN)"])
    _, target = loader.load_data(str(csv_path), "Nexp (kN)", target_transform=None)

    assert np.allclose(
        target.to_numpy(dtype=float),
        df["Nexp (kN)"].to_numpy(dtype=float),
    )


def test_load_data_raises_when_target_column_missing(tmp_path):
    csv_path = tmp_path / "data.csv"
    pd.DataFrame({"feat": [1.0, 2.0]}).to_csv(csv_path, index=False)

    loader = DataLoader()
    with pytest.raises(ValueError, match=re.escape("Target column 'Nexp (kN)' not found in data")):
        loader.load_data(str(csv_path), "Nexp (kN)")


def test_load_data_builds_psi_target_and_derived_columns(tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "As (mm^2)": [1000.0, 1000.0, 1000.0],
            "Ac (mm^2)": [2000.0, 2000.0, 2000.0],
            "fy (MPa)": [300.0, 300.0, 300.0],
            "fc (MPa)": [40.0, 40.0, 40.0],
            "b (mm)": [100.0, 120.0, 120.0],
            "h (mm)": [100.0, 120.0, 120.0],
            "r0 (mm)": [50.0, 0.0, 0.0],
            "L (mm)": [300.0, 600.0, 600.0],
            "e1 (mm)": [0.0, 10.0, 10.0],
            "e2 (mm)": [0.0, 0.0, -5.0],
            "Nexp (kN)": [380.0, 304.0, 304.0],
        }
    )
    df.to_csv(csv_path, index=False)

    loader = DataLoader(required_columns=["Nexp (kN)"])
    features, target = loader.load_data(
        str(csv_path),
        "Nexp (kN)",
        target_transform=None,
        target_mode="psi_over_npl",
    )

    assert "Npl (kN)" in features.columns
    assert "b/h" in features.columns
    assert "L/h" in features.columns
    assert "axial_indicator" in features.columns
    assert "steel_area_ratio" in features.columns
    assert "strength_ratio" in features.columns
    assert "steel_capacity_share" in features.columns
    assert "e_min/h" in features.columns
    assert "end_asymmetry_ratio" in features.columns
    assert "single_curvature_e/h" in features.columns
    assert "double_curvature_e/h" in features.columns
    assert "reverse_curvature_flag" in features.columns
    assert "axial_flag" in features.columns
    assert "section_family" in features.columns
    assert np.allclose(target.to_numpy(dtype=float), np.array([1.0, 0.8, 0.8]))
    assert np.allclose(
        features["axial_indicator"].to_numpy(dtype=float),
        np.array([1.0, 0.0, 0.0]),
    )
    assert np.allclose(
        features["strength_ratio"].to_numpy(dtype=float),
        np.array([7.5, 7.5, 7.5]),
    )
    assert np.allclose(
        features["e_min/h"].to_numpy(dtype=float),
        np.array([0.0, 0.0, 5.0 / 120.0]),
    )
    assert np.allclose(
        features["end_asymmetry_ratio"].to_numpy(dtype=float),
        np.array([0.0, 0.0, 0.5]),
    )
    assert np.allclose(
        features["single_curvature_e/h"].to_numpy(dtype=float),
        np.array([0.0, 5.0 / 120.0, 5.0 / 240.0]),
    )
    assert np.allclose(
        features["double_curvature_e/h"].to_numpy(dtype=float),
        np.array([0.0, 5.0 / 120.0, 15.0 / 240.0]),
    )
    assert np.allclose(
        features["reverse_curvature_flag"].to_numpy(dtype=float),
        np.array([0.0, 0.0, 1.0]),
    )
    assert loader.training_target_name == "psi"
