"""
Feature-set presets and selection helpers for CFST experiments.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.domain_features import (
    AXIAL_INDICATOR_COLUMN,
    B_OVER_H_COLUMN,
    L_OVER_H_COLUMN,
    NPL_COLUMN,
    STEEL_CAPACITY_SHARE_COLUMN,
    STRENGTH_RATIO_COLUMN,
)


HISTORICAL_18_FEATURES: List[str] = [
    "R (%)",
    "fy (MPa)",
    "fc (MPa)",
    "e1 (mm)",
    "e2 (mm)",
    "r0/h",
    "b/t",
    "Ac (mm^2)",
    "As (mm^2)",
    "Re (mm)",
    "te (mm)",
    "ke",
    "xi",
    "sigma_re (MPa)",
    "lambda_bar",
    "e/h",
    "e1/e2",
    "e_bar",
]

CURRENT_21_FEATURES: List[str] = [
    *HISTORICAL_18_FEATURES,
    NPL_COLUMN,
    B_OVER_H_COLUMN,
    L_OVER_H_COLUMN,
]

HISTORICAL_18_PLUS_BH_LH_FEATURES: List[str] = [
    *HISTORICAL_18_FEATURES,
    B_OVER_H_COLUMN,
    L_OVER_H_COLUMN,
]

COMPACT_REPARAM_V1_FEATURES: List[str] = [
    "R (%)",
    STRENGTH_RATIO_COLUMN,
    STEEL_CAPACITY_SHARE_COLUMN,
    "ke",
    "xi",
    "sigma_re (MPa)",
    "lambda_bar",
    L_OVER_H_COLUMN,
    "r0/h",
    B_OVER_H_COLUMN,
    "b/t",
    NPL_COLUMN,
    "e/h",
    "e1/e2",
    AXIAL_INDICATOR_COLUMN,
]

COMPACT_REPARAM_V1_NO_NPL_FEATURES: List[str] = [
    feature
    for feature in COMPACT_REPARAM_V1_FEATURES
    if feature != NPL_COLUMN
]

ECCENTRICITY_COMPACT_V1_FEATURES: List[str] = [
    feature
    for feature in CURRENT_21_FEATURES
    if feature not in {"e1 (mm)", "e2 (mm)"}
] + [AXIAL_INDICATOR_COLUMN]

FEATURE_SET_PRESETS: Dict[str, List[str]] = {
    "historical_18": HISTORICAL_18_FEATURES,
    "current_21": CURRENT_21_FEATURES,
    "historical_18_plus_bh_lh": HISTORICAL_18_PLUS_BH_LH_FEATURES,
    "compact_reparam_v1": COMPACT_REPARAM_V1_FEATURES,
    "compact_reparam_v1_no_npl": COMPACT_REPARAM_V1_NO_NPL_FEATURES,
    "eccentricity_compact_v1": ECCENTRICITY_COMPACT_V1_FEATURES,
}


def get_feature_preset(preset_name: str) -> List[str]:
    """Return a copy of the named feature-set preset."""
    normalized_name = str(preset_name).strip()
    if normalized_name not in FEATURE_SET_PRESETS:
        raise ValueError(
            f"Unsupported feature_selection.preset '{preset_name}'. "
            f"Expected one of {sorted(FEATURE_SET_PRESETS)}."
        )
    return list(FEATURE_SET_PRESETS[normalized_name])


def resolve_feature_selection(
    available_columns: Sequence[str],
    *,
    columns_to_drop: Optional[Sequence[str]] = None,
    feature_selection_config: Optional[Dict[str, Any]] = None,
    feature_frame: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Resolve the effective training feature list from config and available columns.
    """
    available = list(available_columns)
    available_set = set(available)
    drop_list = list(columns_to_drop or [])
    drop_set = set(drop_list)
    config = dict(feature_selection_config or {})

    preset = config.get("preset")
    include_features_raw = config.get("include_features")
    label = config.get("name")

    if preset is not None and include_features_raw is not None:
        raise ValueError(
            "config.data.feature_selection cannot define both 'preset' and "
            "'include_features' at the same time."
        )

    if include_features_raw is not None and not isinstance(include_features_raw, list):
        raise ValueError("config.data.feature_selection.include_features must be a list")

    if preset is not None:
        selected_features = get_feature_preset(str(preset))
        selection_source = "preset"
    elif include_features_raw is not None:
        selected_features = [str(feature) for feature in include_features_raw]
        selection_source = "include_features"
    else:
        if feature_frame is not None:
            selected_features = [
                column
                for column in available
                if column not in drop_set
                and (
                    pd.api.types.is_numeric_dtype(feature_frame[column])
                    or pd.api.types.is_bool_dtype(feature_frame[column])
                )
            ]
        else:
            selected_features = [column for column in available if column not in drop_set]
        selection_source = "auto"

    duplicates = [
        feature
        for index, feature in enumerate(selected_features)
        if feature in selected_features[:index]
    ]
    if duplicates:
        raise ValueError(
            "Duplicate feature names are not allowed in the selected feature list: "
            f"{duplicates}"
        )

    conflicts_with_drop = [
        feature for feature in selected_features if feature in drop_set
    ]
    if conflicts_with_drop:
        raise ValueError(
            "Selected features conflict with config.data.columns_to_drop: "
            f"{conflicts_with_drop}"
        )

    missing_features = [
        feature for feature in selected_features if feature not in available_set
    ]
    if missing_features:
        raise ValueError(
            "Selected features are not available after domain derivation: "
            f"{missing_features}"
        )

    return {
        "name": label or str(preset or "auto"),
        "preset": str(preset) if preset is not None else None,
        "selection_source": selection_source,
        "requested_include_features": (
            [str(feature) for feature in include_features_raw]
            if include_features_raw is not None
            else None
        ),
        "selected_features": list(selected_features),
        "selected_feature_count": len(selected_features),
        "available_feature_count": len(available),
        "columns_to_drop": drop_list,
    }
