import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compute_feature_parameters.py"
)
SPEC = importlib.util.spec_from_file_location("compute_feature_parameters_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _base_source(include_nexp: bool = True):
    source = {
        "b (mm)": 100.0,
        "h (mm)": 100.0,
        "r0 (mm)": 50.0,
        "t (mm)": 5.0,
        "R (%)": 0.0,
        "fy (MPa)": 300.0,
        "fc (MPa)": 40.0,
        "L (mm)": 300.0,
        "e1 (mm)": 0.0,
        "e2 (mm)": 0.0,
    }
    if include_nexp:
        source["Nexp (kN)"] = 1.0
    return source


def test_compute_feature_parameters_outputs_psi_when_nexp_is_available():
    row = MODULE.compute_feature_row(_base_source(include_nexp=True), row_number=2)
    psi_index = MODULE.OUTPUT_COLUMNS.index("psi")
    npl_index = MODULE.OUTPUT_COLUMNS.index("Npl (kN)")
    nexp_index = MODULE.OUTPUT_COLUMNS.index("Nexp (kN)")
    axial_indicator_index = MODULE.OUTPUT_COLUMNS.index("axial_indicator")
    strength_ratio_index = MODULE.OUTPUT_COLUMNS.index("strength_ratio")
    steel_capacity_share_index = MODULE.OUTPUT_COLUMNS.index("steel_capacity_share")
    e_min_over_h_index = MODULE.OUTPUT_COLUMNS.index("e_min/h")
    end_asymmetry_ratio_index = MODULE.OUTPUT_COLUMNS.index("end_asymmetry_ratio")
    single_curvature_index = MODULE.OUTPUT_COLUMNS.index("single_curvature_e/h")
    double_curvature_index = MODULE.OUTPUT_COLUMNS.index("double_curvature_e/h")
    reverse_curvature_index = MODULE.OUTPUT_COLUMNS.index("reverse_curvature_flag")

    assert "psi" in MODULE.OUTPUT_COLUMNS
    assert "axial_indicator" in MODULE.OUTPUT_COLUMNS
    assert "strength_ratio" in MODULE.OUTPUT_COLUMNS
    assert "steel_capacity_share" in MODULE.OUTPUT_COLUMNS
    assert "e_min/h" in MODULE.OUTPUT_COLUMNS
    assert "end_asymmetry_ratio" in MODULE.OUTPUT_COLUMNS
    assert "single_curvature_e/h" in MODULE.OUTPUT_COLUMNS
    assert "double_curvature_e/h" in MODULE.OUTPUT_COLUMNS
    assert "reverse_curvature_flag" in MODULE.OUTPUT_COLUMNS
    assert row[nexp_index] == pytest.approx(1.0)
    assert row[psi_index] == pytest.approx(row[nexp_index] / row[npl_index])
    assert row[axial_indicator_index] == pytest.approx(1.0)
    assert row[strength_ratio_index] == pytest.approx(7.5)
    assert 0.0 < row[steel_capacity_share_index] < 1.0
    assert row[e_min_over_h_index] == pytest.approx(0.0)
    assert row[end_asymmetry_ratio_index] == pytest.approx(0.0)
    assert row[single_curvature_index] == pytest.approx(0.0)
    assert row[double_curvature_index] == pytest.approx(0.0)
    assert row[reverse_curvature_index] == pytest.approx(0.0)


def test_compute_feature_parameters_allows_missing_nexp_and_leaves_psi_blank():
    mapping = MODULE.resolve_columns(
        [
            "b (mm)",
            "h (mm)",
            "r0 (mm)",
            "t (mm)",
            "R (%)",
            "fy (MPa)",
            "fc (MPa)",
            "L (mm)",
            "e1 (mm)",
            "e2 (mm)",
        ]
    )
    row = MODULE.compute_feature_row(_base_source(include_nexp=False), row_number=2)
    psi_index = MODULE.OUTPUT_COLUMNS.index("psi")
    nexp_index = MODULE.OUTPUT_COLUMNS.index("Nexp (kN)")

    assert "Nexp (kN)" not in mapping
    assert row[nexp_index] is None
    assert row[psi_index] is None


def test_compute_feature_parameters_encodes_reverse_curvature_components():
    source = _base_source(include_nexp=True)
    source["e1 (mm)"] = 10.0
    source["e2 (mm)"] = -20.0
    row = MODULE.compute_feature_row(source, row_number=2)

    e_min_over_h_index = MODULE.OUTPUT_COLUMNS.index("e_min/h")
    end_asymmetry_ratio_index = MODULE.OUTPUT_COLUMNS.index("end_asymmetry_ratio")
    single_curvature_index = MODULE.OUTPUT_COLUMNS.index("single_curvature_e/h")
    double_curvature_index = MODULE.OUTPUT_COLUMNS.index("double_curvature_e/h")
    reverse_curvature_index = MODULE.OUTPUT_COLUMNS.index("reverse_curvature_flag")

    assert row[e_min_over_h_index] == pytest.approx(0.1)
    assert row[end_asymmetry_ratio_index] == pytest.approx(0.5)
    assert row[single_curvature_index] == pytest.approx(0.05)
    assert row[double_curvature_index] == pytest.approx(0.15)
    assert row[reverse_curvature_index] == pytest.approx(1.0)
