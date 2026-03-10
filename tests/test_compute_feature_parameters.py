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

    assert "psi" in MODULE.OUTPUT_COLUMNS
    assert row[nexp_index] == pytest.approx(1.0)
    assert row[psi_index] == pytest.approx(row[nexp_index] / row[npl_index])


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
