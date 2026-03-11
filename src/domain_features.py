"""
Shared CFST domain-derived feature and target-space helpers.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


REPORT_TARGET_COLUMN_DEFAULT = "Nexp (kN)"
TARGET_MODE_RAW = "raw"
TARGET_MODE_PSI_OVER_NPL = "psi_over_npl"
VALID_TARGET_MODES = {TARGET_MODE_RAW, TARGET_MODE_PSI_OVER_NPL}

NPL_COLUMN = "Npl (kN)"
PSI_COLUMN = "psi"
B_OVER_H_COLUMN = "b/h"
L_OVER_H_COLUMN = "L/h"
AXIAL_FLAG_COLUMN = "axial_flag"
AXIAL_INDICATOR_COLUMN = "axial_indicator"
SECTION_FAMILY_COLUMN = "section_family"
STEEL_AREA_RATIO_COLUMN = "steel_area_ratio"
STRENGTH_RATIO_COLUMN = "strength_ratio"
STEEL_CAPACITY_SHARE_COLUMN = "steel_capacity_share"
E_MIN_OVER_H_COLUMN = "e_min/h"
END_ASYMMETRY_RATIO_COLUMN = "end_asymmetry_ratio"
SINGLE_CURVATURE_E_OVER_H_COLUMN = "single_curvature_e/h"
DOUBLE_CURVATURE_E_OVER_H_COLUMN = "double_curvature_e/h"
REVERSE_CURVATURE_FLAG_COLUMN = "reverse_curvature_flag"
E_MIN_OVER_H_COLUMN = "e_min/h"
END_ASYMMETRY_RATIO_COLUMN = "end_asymmetry_ratio"
SINGLE_CURVATURE_E_OVER_H_COLUMN = "single_curvature_e/h"
DOUBLE_CURVATURE_E_OVER_H_COLUMN = "double_curvature_e/h"
REVERSE_CURVATURE_FLAG_COLUMN = "reverse_curvature_flag"

_ATOL_ASPECT = 1e-6
_ATOL_RADIUS = 1e-3


def normalize_target_mode(target_mode: Optional[str]) -> str:
    """Normalize the configured training target mode."""
    normalized = (target_mode or TARGET_MODE_RAW).strip().lower()
    if normalized not in VALID_TARGET_MODES:
        raise ValueError(
            f"Unsupported target_mode '{target_mode}'. "
            f"Expected one of {sorted(VALID_TARGET_MODES)}."
        )
    return normalized


def get_training_target_name(
    report_target_column: str,
    target_mode: Optional[str],
) -> str:
    """Return the modeled target name for the configured target mode."""
    normalized_mode = normalize_target_mode(target_mode)
    if normalized_mode == TARGET_MODE_PSI_OVER_NPL:
        return PSI_COLUMN
    return report_target_column


def apply_target_transform(
    values: Union[pd.Series, np.ndarray],
    target_transform_type: Optional[str],
) -> pd.Series:
    """Apply the configured target transform in model space."""
    series = pd.Series(np.asarray(values, dtype=float).reshape(-1))
    if target_transform_type == "log":
        return np.log(series)
    if target_transform_type == "sqrt":
        return np.sqrt(series)
    return series


def inverse_target_transform(
    values: Union[pd.Series, np.ndarray],
    target_transform_type: Optional[str],
) -> np.ndarray:
    """Invert the configured target transform."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if target_transform_type == "log":
        return np.exp(array)
    if target_transform_type == "sqrt":
        return np.square(array)
    return array


def _safe_divide(
    numerator: Union[pd.Series, np.ndarray],
    denominator: Union[pd.Series, np.ndarray],
    *,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Vectorized division with deterministic zero-denominator handling."""
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    result = np.full_like(numerator_array, fill_value=fill_value, dtype=float)
    valid = denominator_array != 0
    np.divide(numerator_array, denominator_array, out=result, where=valid)
    return result


def _require_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for {context}: {missing}"
        )


def _infer_axial_mask(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Infer whether each sample is axial (True) or eccentric (False)."""
    if "e_bar" in df.columns:
        return np.isclose(
            df["e_bar"].astype(float).to_numpy(dtype=float),
            0.0,
            atol=1e-12,
        )

    if {"e1 (mm)", "e2 (mm)"}.issubset(df.columns):
        eccentricity = np.maximum(
            np.abs(df["e1 (mm)"].astype(float).to_numpy(dtype=float)),
            np.abs(df["e2 (mm)"].astype(float).to_numpy(dtype=float)),
        )
        return np.isclose(eccentricity, 0.0, atol=1e-12)

    return None


def _can_derive_eccentricity_shape(df: pd.DataFrame) -> bool:
    return {"e1 (mm)", "e2 (mm)", "h (mm)"}.issubset(df.columns)


def _get_eccentricity_components(
    df: pd.DataFrame,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return signed end eccentricities and reference section depth when available.
    """
    required = {"e1 (mm)", "e2 (mm)", "h (mm)"}
    if not required.issubset(df.columns):
        return None

    e1 = df["e1 (mm)"].astype(float).to_numpy(dtype=float)
    e2 = df["e2 (mm)"].astype(float).to_numpy(dtype=float)
    h = df["h (mm)"].astype(float).to_numpy(dtype=float)
    return e1, e2, h


def infer_section_family(df: pd.DataFrame) -> pd.Series:
    """Infer CFST section family labels from geometric ratios."""
    if B_OVER_H_COLUMN in df.columns:
        aspect_ratio = pd.Series(
            np.asarray(df[B_OVER_H_COLUMN], dtype=float).reshape(-1),
            index=df.index,
        )
    else:
        _require_columns(df, ["b (mm)", "h (mm)"], "section_family derivation")
        aspect_ratio = pd.Series(
            _safe_divide(df["b (mm)"], df["h (mm)"]),
            index=df.index,
        )

    if "r0/h" in df.columns:
        radius_ratio = pd.Series(
            np.asarray(df["r0/h"], dtype=float).reshape(-1),
            index=df.index,
        )
    else:
        _require_columns(df, ["r0 (mm)", "h (mm)"], "section_family derivation")
        radius_ratio = pd.Series(
            _safe_divide(df["r0 (mm)"], df["h (mm)"]),
            index=df.index,
        )

    families = pd.Series("rectangular", index=df.index, dtype="object")
    is_circular = (
        np.isclose(aspect_ratio.to_numpy(dtype=float), 1.0, atol=_ATOL_ASPECT)
        & np.isclose(radius_ratio.to_numpy(dtype=float), 0.5, atol=_ATOL_RADIUS)
    )
    is_square = (
        np.isclose(aspect_ratio.to_numpy(dtype=float), 1.0, atol=_ATOL_ASPECT)
        & ~is_circular
    )
    is_obround = (~is_circular) & (~is_square) & (
        radius_ratio.to_numpy(dtype=float) > _ATOL_RADIUS
    )

    families.loc[is_circular] = "circular"
    families.loc[is_square] = "square"
    families.loc[is_obround] = "obround"
    return families


def ensure_domain_feature_columns(
    df: pd.DataFrame,
    *,
    required_columns: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Backfill domain-derived feature columns when their source columns are available.
    """
    result = df.copy()
    derived_columns: List[str] = []

    if (
        NPL_COLUMN not in result.columns
        and {"As (mm^2)", "Ac (mm^2)", "fy (MPa)", "fc (MPa)"}.issubset(result.columns)
    ):
        result[NPL_COLUMN] = (
            result["As (mm^2)"].astype(float) * result["fy (MPa)"].astype(float)
            + result["Ac (mm^2)"].astype(float) * result["fc (MPa)"].astype(float)
        ) / 1000.0
        derived_columns.append(NPL_COLUMN)

    if (
        B_OVER_H_COLUMN not in result.columns
        and {"b (mm)", "h (mm)"}.issubset(result.columns)
    ):
        result[B_OVER_H_COLUMN] = _safe_divide(
            result["b (mm)"], result["h (mm)"]
        )
        derived_columns.append(B_OVER_H_COLUMN)

    if (
        L_OVER_H_COLUMN not in result.columns
        and {"L (mm)", "h (mm)"}.issubset(result.columns)
    ):
        result[L_OVER_H_COLUMN] = _safe_divide(
            result["L (mm)"], result["h (mm)"]
        )
        derived_columns.append(L_OVER_H_COLUMN)

    if _can_derive_eccentricity_shape(result):
        e1_abs = np.abs(result["e1 (mm)"].astype(float).to_numpy(dtype=float))
        e2_abs = np.abs(result["e2 (mm)"].astype(float).to_numpy(dtype=float))
        e_abs_max = np.maximum(e1_abs, e2_abs)
        e_abs_min = np.minimum(e1_abs, e2_abs)
        h_values = result["h (mm)"]
        signed_product = (
            result["e1 (mm)"].astype(float).to_numpy(dtype=float)
            * result["e2 (mm)"].astype(float).to_numpy(dtype=float)
        )

        if E_MIN_OVER_H_COLUMN not in result.columns:
            result[E_MIN_OVER_H_COLUMN] = _safe_divide(
                e_abs_min,
                h_values,
            )
            derived_columns.append(E_MIN_OVER_H_COLUMN)

        if END_ASYMMETRY_RATIO_COLUMN not in result.columns:
            result[END_ASYMMETRY_RATIO_COLUMN] = _safe_divide(
                e_abs_min,
                e_abs_max,
            )
            derived_columns.append(END_ASYMMETRY_RATIO_COLUMN)

        if SINGLE_CURVATURE_E_OVER_H_COLUMN not in result.columns:
            result[SINGLE_CURVATURE_E_OVER_H_COLUMN] = _safe_divide(
                0.5 * np.abs(
                    result["e1 (mm)"].astype(float).to_numpy(dtype=float)
                    + result["e2 (mm)"].astype(float).to_numpy(dtype=float)
                ),
                h_values,
            )
            derived_columns.append(SINGLE_CURVATURE_E_OVER_H_COLUMN)

        if DOUBLE_CURVATURE_E_OVER_H_COLUMN not in result.columns:
            result[DOUBLE_CURVATURE_E_OVER_H_COLUMN] = _safe_divide(
                0.5 * np.abs(
                    result["e1 (mm)"].astype(float).to_numpy(dtype=float)
                    - result["e2 (mm)"].astype(float).to_numpy(dtype=float)
                ),
                h_values,
            )
            derived_columns.append(DOUBLE_CURVATURE_E_OVER_H_COLUMN)

        if REVERSE_CURVATURE_FLAG_COLUMN not in result.columns:
            result[REVERSE_CURVATURE_FLAG_COLUMN] = np.where(
                signed_product < 0.0,
                1.0,
                0.0,
            )
            derived_columns.append(REVERSE_CURVATURE_FLAG_COLUMN)

    if (
        STEEL_AREA_RATIO_COLUMN not in result.columns
        and {"As (mm^2)", "Ac (mm^2)"}.issubset(result.columns)
    ):
        result[STEEL_AREA_RATIO_COLUMN] = _safe_divide(
            result["As (mm^2)"], result["Ac (mm^2)"]
        )
        derived_columns.append(STEEL_AREA_RATIO_COLUMN)

    if (
        STRENGTH_RATIO_COLUMN not in result.columns
        and {"fy (MPa)", "fc (MPa)"}.issubset(result.columns)
    ):
        result[STRENGTH_RATIO_COLUMN] = _safe_divide(
            result["fy (MPa)"], result["fc (MPa)"]
        )
        derived_columns.append(STRENGTH_RATIO_COLUMN)

    if (
        STEEL_CAPACITY_SHARE_COLUMN not in result.columns
        and {"As (mm^2)", "Ac (mm^2)", "fy (MPa)", "fc (MPa)"}.issubset(result.columns)
    ):
        steel_capacity = (
            result["As (mm^2)"].astype(float) * result["fy (MPa)"].astype(float)
        )
        total_capacity = steel_capacity + (
            result["Ac (mm^2)"].astype(float) * result["fc (MPa)"].astype(float)
        )
        result[STEEL_CAPACITY_SHARE_COLUMN] = _safe_divide(
            steel_capacity,
            total_capacity,
        )
        derived_columns.append(STEEL_CAPACITY_SHARE_COLUMN)

    eccentricity_components = _get_eccentricity_components(result)
    if eccentricity_components is not None:
        e1, e2, h = eccentricity_components
        abs_e1 = np.abs(e1)
        abs_e2 = np.abs(e2)
        e_min = np.minimum(abs_e1, abs_e2)
        e_max = np.maximum(abs_e1, abs_e2)

        if E_MIN_OVER_H_COLUMN not in result.columns:
            result[E_MIN_OVER_H_COLUMN] = _safe_divide(e_min, h)
            derived_columns.append(E_MIN_OVER_H_COLUMN)

        if END_ASYMMETRY_RATIO_COLUMN not in result.columns:
            result[END_ASYMMETRY_RATIO_COLUMN] = _safe_divide(e_min, e_max)
            derived_columns.append(END_ASYMMETRY_RATIO_COLUMN)

        if SINGLE_CURVATURE_E_OVER_H_COLUMN not in result.columns:
            result[SINGLE_CURVATURE_E_OVER_H_COLUMN] = _safe_divide(
                np.abs(e1 + e2) / 2.0,
                h,
            )
            derived_columns.append(SINGLE_CURVATURE_E_OVER_H_COLUMN)

        if DOUBLE_CURVATURE_E_OVER_H_COLUMN not in result.columns:
            result[DOUBLE_CURVATURE_E_OVER_H_COLUMN] = _safe_divide(
                np.abs(e1 - e2) / 2.0,
                h,
            )
            derived_columns.append(DOUBLE_CURVATURE_E_OVER_H_COLUMN)

        if REVERSE_CURVATURE_FLAG_COLUMN not in result.columns:
            result[REVERSE_CURVATURE_FLAG_COLUMN] = np.where(
                e1 * e2 < 0.0,
                1.0,
                0.0,
            )
            derived_columns.append(REVERSE_CURVATURE_FLAG_COLUMN)

    axial_mask = _infer_axial_mask(result)
    if AXIAL_FLAG_COLUMN not in result.columns:
        if axial_mask is not None:
            result[AXIAL_FLAG_COLUMN] = np.where(axial_mask, "axial", "eccentric")
            derived_columns.append(AXIAL_FLAG_COLUMN)

    if AXIAL_INDICATOR_COLUMN not in result.columns:
        if axial_mask is not None:
            result[AXIAL_INDICATOR_COLUMN] = np.where(axial_mask, 1.0, 0.0)
            derived_columns.append(AXIAL_INDICATOR_COLUMN)

    if SECTION_FAMILY_COLUMN not in result.columns:
        can_build_section_family = (
            B_OVER_H_COLUMN in result.columns
            or {"b (mm)", "h (mm)"}.issubset(result.columns)
        ) and (
            "r0/h" in result.columns
            or {"r0 (mm)", "h (mm)"}.issubset(result.columns)
        )
        if can_build_section_family:
            result[SECTION_FAMILY_COLUMN] = infer_section_family(result)
            derived_columns.append(SECTION_FAMILY_COLUMN)

    if required_columns:
        _require_columns(result, list(required_columns), "domain-derived feature setup")

    return result, derived_columns


def ensure_target_mode_columns(
    df: pd.DataFrame,
    *,
    report_target_column: str = REPORT_TARGET_COLUMN_DEFAULT,
    target_mode: Optional[str] = None,
    include_training_target: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Ensure target-mode-specific helper columns are available for training/evaluation."""
    normalized_mode = normalize_target_mode(target_mode)
    required_columns: List[str] = []
    if normalized_mode == TARGET_MODE_PSI_OVER_NPL:
        required_columns.append(NPL_COLUMN)

    result, derived_columns = ensure_domain_feature_columns(
        df,
        required_columns=required_columns or None,
    )

    if (
        normalized_mode == TARGET_MODE_PSI_OVER_NPL
        and include_training_target
        and PSI_COLUMN not in result.columns
    ):
        _require_columns(
            result,
            [report_target_column, NPL_COLUMN],
            "psi target derivation",
        )
        result[PSI_COLUMN] = _safe_divide(
            result[report_target_column],
            result[NPL_COLUMN],
        )
        derived_columns.append(PSI_COLUMN)

    return result, derived_columns


def ensure_prediction_feature_columns(
    df: pd.DataFrame,
    *,
    target_mode: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Ensure only prediction-time derived features are available."""
    normalized_mode = normalize_target_mode(target_mode)
    required_columns = [NPL_COLUMN] if normalized_mode == TARGET_MODE_PSI_OVER_NPL else None
    return ensure_domain_feature_columns(
        df,
        required_columns=required_columns,
    )


def compute_training_target(
    df: pd.DataFrame,
    *,
    report_target_column: str = REPORT_TARGET_COLUMN_DEFAULT,
    target_mode: Optional[str] = None,
) -> pd.Series:
    """Read or derive the modeled target in its untransformed training space."""
    normalized_mode = normalize_target_mode(target_mode)
    if normalized_mode == TARGET_MODE_RAW:
        _require_columns(df, [report_target_column], "raw target derivation")
        return df[report_target_column].astype(float).copy()

    target_ready_df, _ = ensure_target_mode_columns(
        df,
        report_target_column=report_target_column,
        target_mode=normalized_mode,
    )
    return target_ready_df[PSI_COLUMN].astype(float).copy()


def restore_report_target(
    values: Union[pd.Series, np.ndarray],
    *,
    target_mode: Optional[str] = None,
    target_transform_type: Optional[str] = None,
    reference_features: Optional[pd.DataFrame] = None,
    reference_scale: Optional[Union[pd.Series, np.ndarray]] = None,
) -> np.ndarray:
    """Map modeled outputs back into the reported Nexp-space."""
    normalized_mode = normalize_target_mode(target_mode)
    modeled_values = inverse_target_transform(values, target_transform_type)

    if normalized_mode == TARGET_MODE_RAW:
        return modeled_values

    if reference_scale is not None:
        scale = np.asarray(reference_scale, dtype=float).reshape(-1)
    else:
        if reference_features is None:
            raise ValueError(
                "reference_features or reference_scale is required to restore psi_over_npl targets"
            )
        feature_frame, _ = ensure_domain_feature_columns(
            reference_features,
            required_columns=[NPL_COLUMN],
        )
        scale = feature_frame[NPL_COLUMN].to_numpy(dtype=float)

    return modeled_values * scale
