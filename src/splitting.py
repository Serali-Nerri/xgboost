"""
Utilities for regression-aware stratified splitting and regime binning.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)
VALID_REGIME_MODES = {"categorical", "fixed_bins", "train_quantile"}


def _coerce_auxiliary_specs(raw_specs: Any) -> List[Dict[str, Any]]:
    """Normalize auxiliary stratification feature specs from config."""
    if raw_specs is None:
        return []
    if not isinstance(raw_specs, list):
        raise ValueError("split.auxiliary_features must be a list of mappings")

    normalized: List[Dict[str, Any]] = []
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, dict):
            raise ValueError("Each split.auxiliary_features item must be a mapping")
        column = raw_spec.get("column")
        bins = raw_spec.get("bins", 3)
        if not isinstance(column, str) or not column.strip():
            raise ValueError("Each auxiliary stratification feature requires a column")
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
            raise ValueError(
                f"Invalid bins for auxiliary stratification feature '{column}': {bins}"
            )
        normalized.append({"column": column, "bins": bins})

    return normalized


def _quantile_codes(series: pd.Series, n_bins: int) -> Tuple[pd.Series, int]:
    """
    Create stable quantile bin codes for splitting.

    Uses raw-value qcut first. If duplicated quantile edges collapse bins, it falls back
    to rank-based qcut while still preserving the requested ordering.
    """
    series_no_na = cast(pd.Series, series.astype(float))
    unique_count = int(series_no_na.nunique(dropna=True))
    effective_bins = min(max(2, int(n_bins)), unique_count)
    if effective_bins < 2:
        return pd.Series(["bin0"] * len(series_no_na), index=series_no_na.index), 1

    try:
        raw_codes = pd.qcut(
            series_no_na,
            q=effective_bins,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        raw_codes = None

    if raw_codes is not None and not raw_codes.isna().any():
        int_codes = cast(pd.Series, raw_codes.astype(int))
        return int_codes.map(lambda value: f"bin{value}"), int(int_codes.nunique())

    ranked = series_no_na.rank(method="first")
    rank_codes = pd.qcut(
        ranked,
        q=effective_bins,
        labels=False,
        duplicates="drop",
    )
    int_codes = cast(pd.Series, rank_codes.astype(int))
    return int_codes.map(lambda value: f"bin{value}"), int(int_codes.nunique())


def build_regression_stratification_labels(
    features: pd.DataFrame,
    target_raw: pd.Series,
    split_config: Optional[Dict[str, Any]] = None,
    minimum_count: int = 2,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Build robust stratification labels for regression tasks.

    The target is always the primary stratification axis. Optional auxiliary features
    are searched as subsets, preferring the richest stable combination.
    """
    split_config = split_config or {}
    target_bins_requested = split_config.get("target_bins", 10)
    if (
        isinstance(target_bins_requested, bool)
        or not isinstance(target_bins_requested, int)
        or target_bins_requested < 2
    ):
        raise ValueError("split.target_bins must be an integer >= 2")

    auxiliary_specs = _coerce_auxiliary_specs(split_config.get("auxiliary_features"))
    available_auxiliary_specs: List[Dict[str, Any]] = []
    for spec in auxiliary_specs:
        if spec["column"] not in features.columns:
            logger.warning(
                "Auxiliary stratification column '%s' not found; skipping",
                spec["column"],
            )
            continue
        available_auxiliary_specs.append(spec)

    min_required = max(2, int(minimum_count))
    candidate_subsets = [
        [available_auxiliary_specs[index] for index in subset_indexes]
        for subset_size in range(len(available_auxiliary_specs), 0, -1)
        for subset_indexes in combinations(range(len(available_auxiliary_specs)), subset_size)
    ]
    candidate_subsets.append([])

    for target_bins in range(target_bins_requested, 1, -1):
        target_component, target_bins_used = _quantile_codes(target_raw, target_bins)
        base_parts = [target_component.map(lambda value: f"target:{value}")]
        candidate_subsets_tried: List[Dict[str, Any]] = []

        for used_auxiliary_specs in candidate_subsets:
            parts = base_parts.copy()
            auxiliary_bins_used: List[Dict[str, Any]] = []

            for spec in used_auxiliary_specs:
                aux_component, bins_used = _quantile_codes(
                    features[spec["column"]],
                    int(spec["bins"]),
                )
                parts.append(
                    aux_component.map(
                        lambda value, column=spec["column"]: f"{column}:{value}"
                    )
                )
                auxiliary_bins_used.append(
                    {
                        "column": spec["column"],
                        "requested_bins": int(spec["bins"]),
                        "used_bins": int(bins_used),
                    }
                )

            combined = pd.Series(
                [
                    "|".join(values)
                    for values in zip(*(part.astype(str).tolist() for part in parts))
                ],
                index=target_raw.index,
                dtype="object",
            )

            counts = combined.value_counts()
            if counts.empty:
                continue
            min_count_observed = int(counts.min())
            candidate_attempt = {
                "columns": [spec["column"] for spec in used_auxiliary_specs],
                "requested_bins": [int(spec["bins"]) for spec in used_auxiliary_specs],
                "n_strata": int(combined.nunique()),
                "minimum_count_observed": min_count_observed,
                "accepted": combined.nunique() > 1 and min_count_observed >= min_required,
            }
            candidate_subsets_tried.append(candidate_attempt)
            if combined.nunique() > 1 and min_count_observed >= min_required:
                metadata = {
                    "strategy": "regression_stratified",
                    "requested_target_bins": int(target_bins_requested),
                    "used_target_bins": int(target_bins_used),
                    "requested_auxiliary_features": auxiliary_specs,
                    "used_auxiliary_features": auxiliary_bins_used,
                    "candidate_subsets_tried": candidate_subsets_tried,
                    "minimum_count_required": min_required,
                    "minimum_count_observed": min_count_observed,
                    "n_strata": int(combined.nunique()),
                    "largest_stratum_size": int(counts.max()),
                }
                return combined, metadata

    fallback = pd.Series(["fallback"] * len(target_raw), index=target_raw.index, dtype="object")
    metadata = {
        "strategy": "random",
        "reason": "Unable to build stable regression strata; using fallback label",
        "requested_auxiliary_features": auxiliary_specs,
        "minimum_count_required": min_required,
        "n_strata": 1,
    }
    logger.warning("Falling back to a single stratum label; random split will be used")
    return fallback, metadata


def required_stratum_count(
    test_size: float,
    validation_size: float,
    n_splits: int,
    configured_minimum: Optional[int] = None,
) -> int:
    """
    Estimate a safe minimum stratum size for train/test, inner validation, and CV.
    """
    required = [2, int(n_splits)]
    if test_size > 0:
        required.append(int(math.ceil(1.0 / test_size)))
    if validation_size > 0 and n_splits > 1:
        effective_train_fraction = 1.0 - (1.0 / n_splits)
        required.append(
            int(math.ceil(1.0 / max(validation_size * effective_train_fraction, 1e-9)))
        )
    if configured_minimum is not None:
        required.append(int(configured_minimum))
    return max(required)


def _normalize_regime_mode(raw_mode: Optional[str]) -> str:
    mode = str(raw_mode or "train_quantile").strip().lower()
    if mode not in VALID_REGIME_MODES:
        raise ValueError(
            f"Unsupported regime mode '{raw_mode}'. Expected one of {sorted(VALID_REGIME_MODES)}."
        )
    return mode


def _resolve_regime_source_values(
    *,
    features: Optional[pd.DataFrame],
    target: pd.Series,
    regime_spec: Dict[str, Any],
) -> pd.Series:
    source = str(regime_spec.get("source", "target")).strip().lower()
    if source == "target":
        return target.astype(float)
    if source != "feature":
        raise ValueError(
            f"Unsupported regime source '{source}' for regime '{regime_spec.get('name', '')}'"
        )
    column = regime_spec.get("column")
    if features is None:
        raise ValueError(
            f"Feature data is required for feature-based regime '{regime_spec.get('name', '')}'"
        )
    if column not in features.columns:
        raise ValueError(
            f"Feature column '{column}' is missing for regime '{regime_spec.get('name', '')}'"
        )
    return cast(pd.Series, features[column].copy())


def _fixed_bin_ranges(
    edges: Sequence[float],
    labels: Sequence[str],
) -> List[Dict[str, Any]]:
    return [
        {
            "label": str(labels[index]),
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
        }
        for index in range(len(labels))
    ]


def fit_regime_schema(
    values: pd.Series,
    regime_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Fit a reusable regime schema on the reference split."""
    regime_name = str(regime_spec.get("name", "")).strip()
    if not regime_name:
        raise ValueError("Each regime definition requires a non-empty name")

    source = str(regime_spec.get("source", "target")).strip().lower()
    mode = _normalize_regime_mode(regime_spec.get("mode"))
    schema: Dict[str, Any] = {
        "name": regime_name,
        "mode": mode,
        "source": source,
        "column": regime_spec.get("column"),
    }

    if mode == "categorical":
        categories = [
            str(value)
            for value in pd.Series(values).dropna().astype("object").unique().tolist()
        ]
        schema["categories"] = categories
        schema["ranges"] = [{"label": category, "value": category} for category in categories]
        return schema

    if mode == "fixed_bins":
        edges = regime_spec.get("edges")
        if not isinstance(edges, list) or len(edges) < 2:
            raise ValueError(
                f"Regime '{regime_name}' fixed_bins mode requires an 'edges' list with at least two values"
            )
        labels = regime_spec.get("labels")
        if labels is None:
            labels = [f"{regime_name}_bin{idx + 1}" for idx in range(len(edges) - 1)]
        if not isinstance(labels, list) or len(labels) != len(edges) - 1:
            raise ValueError(
                f"Regime '{regime_name}' labels must match edges-1 in length"
            )
        schema["edges"] = [float(edge) for edge in edges]
        schema["labels"] = [str(label) for label in labels]
        schema["ranges"] = _fixed_bin_ranges(schema["edges"], schema["labels"])
        return schema

    bins = regime_spec.get("bins", 4)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError(f"Invalid bins for regime '{regime_name}': {bins}")

    series = cast(pd.Series, pd.Series(values).astype(float))
    unique_count = int(series.nunique(dropna=True))
    effective_bins = min(max(2, int(bins)), unique_count)
    if effective_bins < 2:
        label = f"{regime_name}_all"
        schema["edges"] = [float(series.min()), float(series.max())]
        schema["labels"] = [label]
        schema["ranges"] = [{"label": label, "lower": float(series.min()), "upper": float(series.max())}]
        return schema

    _, edges = pd.qcut(
        series,
        q=effective_bins,
        duplicates="drop",
        retbins=True,
    )
    edge_list = [float(edge) for edge in edges]
    labels = [f"{regime_name}_q{idx + 1}" for idx in range(len(edge_list) - 1)]
    schema["bins_requested"] = int(bins)
    schema["bins_used"] = int(len(labels))
    schema["edges"] = edge_list
    schema["labels"] = labels
    schema["ranges"] = _fixed_bin_ranges(edge_list, labels)
    return schema


def apply_regime_schema(
    values: pd.Series,
    schema: Dict[str, Any],
) -> pd.Series:
    """Apply a fitted regime schema to any compatible split."""
    mode = _normalize_regime_mode(schema.get("mode"))
    series = pd.Series(values, index=values.index)

    if mode == "categorical":
        categories = set(str(category) for category in schema.get("categories", []))
        labels = series.astype("object").map(lambda value: str(value) if pd.notna(value) else np.nan)
        if categories:
            labels = labels.where(labels.isin(categories), other=np.nan)
        return cast(pd.Series, labels.astype("object"))

    edges = [float(edge) for edge in schema.get("edges", [])]
    labels = [str(label) for label in schema.get("labels", [])]
    if len(edges) < 2 or len(labels) != len(edges) - 1:
        raise ValueError(f"Invalid regime schema for '{schema.get('name', '')}'")

    numeric_values = np.asarray(series, dtype=float)
    result = pd.Series(np.nan, index=series.index, dtype="object")
    for index, label in enumerate(labels):
        lower = edges[index]
        upper = edges[index + 1]
        if index == 0 and lower == upper:
            mask = np.isclose(numeric_values, lower, atol=1e-12)
        elif index == 0:
            mask = (numeric_values >= lower) & (numeric_values <= upper)
        elif lower == upper:
            mask = np.isclose(numeric_values, upper, atol=1e-12)
        else:
            mask = (numeric_values > lower) & (numeric_values <= upper)
        result.loc[mask] = label
    return result


def build_regime_labels(
    values: pd.Series,
    n_bins: int,
    prefix: str,
) -> Tuple[pd.Series, List[Dict[str, Any]]]:
    """
    Build human-readable quantile regime labels for reporting.
    """
    schema = fit_regime_schema(
        values,
        {"name": prefix, "mode": "train_quantile", "source": "target", "bins": n_bins},
    )
    return apply_regime_schema(values, schema), cast(List[Dict[str, Any]], schema["ranges"])


def get_split_strategy(split_config: Optional[Dict[str, Any]]) -> str:
    """Read the configured split strategy."""
    strategy = str((split_config or {}).get("strategy", "random")).strip().lower()
    if strategy not in {"random", "regression_stratified"}:
        raise ValueError(
            "split.strategy must be either 'random' or 'regression_stratified'"
        )
    return strategy
