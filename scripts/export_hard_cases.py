#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import DataLoader
from src.predictor import Predictor
from src.splitting import (
    apply_regime_schema,
    build_regression_stratification_labels,
    get_split_strategy,
    required_stratum_count,
)
from src.utils.model_utils import load_metadata, load_model_from_directory


COMPUTE_FEATURES_PATH = REPO_ROOT / "scripts" / "compute_feature_parameters.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the hardest holdout samples for manual data review. "
            "Candidates are the union of top quantile absolute-error and "
            "absolute-percentage-error cases on the test split."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="Training config YAML path.")
    parser.add_argument("--model-dir", required=True, type=Path, help="Directory containing trained model artifacts.")
    parser.add_argument("--raw-csv", required=True, type=Path, help="Raw CSV used to generate the processed dataset.")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path.")
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.95,
        help="Quantile cutoff for abs_error and abs_pct_error. Default: 0.95",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def load_compute_features_module():
    spec = importlib.util.spec_from_file_location(
        "compute_feature_parameters_module", COMPUTE_FEATURES_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {COMPUTE_FEATURES_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_valid_raw_rows(raw_csv_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    module = load_compute_features_module()

    valid_rows: List[Dict[str, Any]] = []
    skipped_rows: List[str] = []

    with raw_csv_path.open("r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])
        column_mapping = module.resolve_columns(reader.fieldnames)

        for row_number, row in enumerate(reader, start=2):
            try:
                source = module.parse_source_row(row, column_mapping, row_number)
                module.compute_feature_row(source, row_number)
                raw_record = {field: row.get(field, "") for field in fieldnames}
                raw_record["raw_csv_row_num"] = row_number
                valid_rows.append(raw_record)
            except ValueError as exc:
                skipped_rows.append(str(exc))

    return pd.DataFrame(valid_rows), skipped_rows


def apply_saved_regimes(
    features: pd.DataFrame,
    report_target: pd.Series,
    regime_schema: Dict[str, Any],
) -> pd.DataFrame:
    regime_columns: Dict[str, pd.Series] = {}
    for regime in regime_schema.get("regimes", []):
        regime_name = str(regime.get("name", "")).strip()
        if not regime_name:
            continue
        source = str(regime.get("source", "target")).strip().lower()
        if source == "feature":
            column = str(regime.get("column", "")).strip()
            if column not in features.columns:
                continue
            values = features[column]
        else:
            values = report_target
        regime_columns[f"regime_{regime_name}"] = apply_regime_schema(values, regime)
    return pd.DataFrame(regime_columns, index=features.index)


def build_review_dataset(
    *,
    config: Dict[str, Any],
    model_dir: Path,
    raw_csv_path: Path,
    quantile: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    data_config = config["data"]
    cv_config = config["cv"]
    model_config = config["model"]

    target_column = str(data_config["target_column"])
    target_transform_type = (
        str(data_config["target_transform"]["type"])
        if data_config.get("target_transform", {}).get("enabled")
        else None
    )
    target_mode = str(data_config.get("target_mode", "raw"))

    valid_raw_rows, skipped_rows = collect_valid_raw_rows(raw_csv_path)
    if skipped_rows:
        print(f"Skipped {len(skipped_rows)} raw rows while rebuilding alignment.")

    loader = DataLoader(required_columns=[target_column])
    features, _ = loader.load_data(
        str(data_config["file_path"]),
        target_column,
        target_transform=target_transform_type,
        target_mode=target_mode,
    )
    report_target_raw = loader.target_raw
    training_target_raw = loader.training_target_raw
    if report_target_raw is None or training_target_raw is None:
        raise ValueError("DataLoader did not provide raw targets.")

    if len(valid_raw_rows) != len(features):
        raise ValueError(
            "Raw/processed alignment mismatch: "
            f"{len(valid_raw_rows)} valid raw rows vs {len(features)} processed rows"
        )

    split_strategy = get_split_strategy(data_config.get("split", {}))
    n_splits = int(cv_config.get("n_splits", 5))
    validation_size = float(model_config.get("validation_size", 0.0) or 0.0)
    minimum_stratum_size = required_stratum_count(
        test_size=float(data_config.get("test_size", 0.2)),
        validation_size=validation_size,
        n_splits=n_splits,
        configured_minimum=data_config.get("split", {}).get("min_stratum_size"),
    )

    stratify_labels_full = None
    if split_strategy == "regression_stratified":
        stratify_labels_candidate, _ = build_regression_stratification_labels(
            features=features,
            target_raw=training_target_raw,
            split_config=data_config.get("split", {}),
            minimum_count=minimum_stratum_size,
        )
        if stratify_labels_candidate.nunique() > 1:
            stratify_labels_full = stratify_labels_candidate

    split_kwargs: Dict[str, Any] = {
        "test_size": float(data_config.get("test_size", 0.2)),
        "random_state": int(data_config.get("random_state", 42)),
    }
    split_inputs: List[Any] = [
        features,
        report_target_raw,
        valid_raw_rows.reset_index(drop=True),
    ]
    if stratify_labels_full is not None:
        split_kwargs["stratify"] = stratify_labels_full

    split_result = train_test_split(*split_inputs, **split_kwargs)
    X_train, X_test, y_train_report, y_test_report, raw_train, raw_test = split_result
    del X_train, y_train_report, raw_train

    model, preprocessor, feature_names = load_model_from_directory(str(model_dir))
    metadata = load_metadata(str(model_dir / "training_metadata.json"))
    predictor = Predictor(
        model=model,
        preprocessor=preprocessor,
        feature_names=feature_names,
        metadata=metadata,
    )
    y_pred = predictor.predict(X_test)

    report_array = y_test_report.to_numpy(dtype=float)
    pred_array = np.asarray(y_pred, dtype=float).reshape(-1)
    error_array = pred_array - report_array
    abs_error_array = np.abs(error_array)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_array = np.divide(pred_array, report_array)
        pct_error_array = np.divide(error_array, report_array) * 100.0
    abs_pct_error_array = np.abs(pct_error_array)
    abs_log_ratio_array = np.abs(np.log(ratio_array))

    result = raw_test.reset_index(drop=True).copy()
    result["actual_nexp_kn"] = report_array
    result["predicted_nexp_kn"] = pred_array
    result["error_kn"] = error_array
    result["abs_error_kn"] = abs_error_array
    result["pct_error"] = pct_error_array
    result["abs_pct_error"] = abs_pct_error_array
    result["pred_actual_ratio"] = ratio_array
    result["abs_log_ratio"] = abs_log_ratio_array

    informative_feature_columns = [
        "Npl (kN)",
        "lambda_bar",
        "e/h",
        "xi",
        "axial_flag",
        "section_family",
    ]
    for column in informative_feature_columns:
        if column in X_test.columns:
            result[column] = X_test[column].to_numpy()
    if "Npl (kN)" in X_test.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            result["actual_psi"] = report_array / X_test["Npl (kN)"].to_numpy(dtype=float)
            result["predicted_psi"] = pred_array / X_test["Npl (kN)"].to_numpy(dtype=float)

    regime_labels = apply_saved_regimes(
        X_test,
        y_test_report,
        metadata.get("regime_schema", {}),
    )
    if not regime_labels.empty:
        result = pd.concat([result, regime_labels.reset_index(drop=True)], axis=1)

    abs_error_threshold = float(np.nanquantile(result["abs_error_kn"], quantile))
    abs_pct_threshold = float(np.nanquantile(result["abs_pct_error"], quantile))
    result["is_top_abs_error"] = result["abs_error_kn"] >= abs_error_threshold
    result["is_top_abs_pct_error"] = result["abs_pct_error"] >= abs_pct_threshold
    result["hard_case_reason"] = np.select(
        [
            result["is_top_abs_error"] & result["is_top_abs_pct_error"],
            result["is_top_abs_error"],
            result["is_top_abs_pct_error"],
        ],
        [
            "top_abs_error_and_top_abs_pct_error",
            "top_abs_error",
            "top_abs_pct_error",
        ],
        default="",
    )

    abs_error_rank = result["abs_error_kn"].rank(method="dense", ascending=False, pct=True)
    abs_pct_rank = result["abs_pct_error"].rank(method="dense", ascending=False, pct=True)
    result["hardness_score"] = abs_error_rank + abs_pct_rank

    candidates = result[
        result["is_top_abs_error"] | result["is_top_abs_pct_error"]
    ].copy()
    candidates = candidates.sort_values(
        by=[
            "hardness_score",
            "abs_pct_error",
            "abs_error_kn",
            "abs_log_ratio",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    candidates.insert(0, "hard_case_rank", np.arange(1, len(candidates) + 1))
    summary = {
        "quantile": quantile,
        "top_abs_error_threshold": abs_error_threshold,
        "top_abs_pct_error_threshold": abs_pct_threshold,
        "n_test_samples": int(len(result)),
        "n_candidates": int(len(candidates)),
    }
    return candidates, summary


def main() -> int:
    args = parse_args()
    if not 0.0 < args.quantile < 1.0:
        raise ValueError("--quantile must be between 0 and 1")

    config = load_yaml(args.config)
    candidates, summary = build_review_dataset(
        config=config,
        model_dir=args.model_dir,
        raw_csv_path=args.raw_csv,
        quantile=args.quantile,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output, index=False)

    print(f"Saved {len(candidates)} hard-case review rows to {args.output}")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
