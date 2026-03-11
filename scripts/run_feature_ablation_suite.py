#!/usr/bin/env python3
"""
Run staged feature ablation / reparameterization screening experiments.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.domain_features import (  # noqa: E402
    AXIAL_INDICATOR_COLUMN,
    STEEL_AREA_RATIO_COLUMN,
    STEEL_CAPACITY_SHARE_COLUMN,
    STRENGTH_RATIO_COLUMN,
)
from src.feature_sets import (  # noqa: E402
    COMPACT_REPARAM_V1_FEATURES,
    COMPACT_REPARAM_V1_NO_NPL_FEATURES,
    CURRENT_21_FEATURES,
    HISTORICAL_18_FEATURES,
)


GENERATED_CONFIG_DIR = REPO_ROOT / "config" / "experiments" / "feature_ablation_generated"
OUTPUT_ROOT = REPO_ROOT / "output" / "feature_ablation"
LOG_ROOT = REPO_ROOT / "logs" / "feature_ablation"
DOC_ROOT = REPO_ROOT / "doc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run staged feature-ablation screening experiments.",
    )
    parser.add_argument(
        "--base-config",
        default=str(REPO_ROOT / "config" / "config.yaml"),
        help="Base config path used as the screening template.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and summarize from existing outputs only.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "stage1", "stage2", "finalists"],
        default="all",
        help="Which stage(s) to execute.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=False, sort_keys=False)


def ensure_unique(features: Iterable[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for feature in features:
        if feature not in seen:
            unique.append(feature)
            seen.add(feature)
    return unique


def build_stage1_variants() -> List[Dict[str, Any]]:
    hist18 = list(HISTORICAL_18_FEATURES)
    return [
        {
            "id": "D0_hist18",
            "description": "Historical 18-feature baseline",
            "features": hist18,
        },
        {
            "id": "D1_hist18_plus_npl",
            "description": "Historical 18 plus Npl",
            "features": hist18 + ["Npl (kN)"],
        },
        {
            "id": "D2_hist18_plus_b_over_h",
            "description": "Historical 18 plus b/h",
            "features": hist18 + ["b/h"],
        },
        {
            "id": "D3_hist18_plus_l_over_h",
            "description": "Historical 18 plus L/h",
            "features": hist18 + ["L/h"],
        },
        {
            "id": "D4_hist18_plus_npl_b_over_h",
            "description": "Historical 18 plus Npl and b/h",
            "features": hist18 + ["Npl (kN)", "b/h"],
        },
        {
            "id": "D5_hist18_plus_npl_l_over_h",
            "description": "Historical 18 plus Npl and L/h",
            "features": hist18 + ["Npl (kN)", "L/h"],
        },
        {
            "id": "D6_hist18_plus_b_over_h_l_over_h",
            "description": "Historical 18 plus b/h and L/h",
            "features": hist18 + ["b/h", "L/h"],
        },
        {
            "id": "D7_current21",
            "description": "Current 21-feature log(psi) mainline",
            "features": list(CURRENT_21_FEATURES),
        },
    ]


def build_stage2_variants(base_features: List[str]) -> List[Dict[str, Any]]:
    base = list(base_features)

    def without(*removed: str) -> List[str]:
        removed_set = set(removed)
        return [feature for feature in base if feature not in removed_set]

    return [
        {
            "id": "R1_eccentricity_compact_v1",
            "description": "Replace raw e1/e2 with axial_indicator while keeping normalized eccentricity descriptors",
            "features": ensure_unique(
                without("e1 (mm)", "e2 (mm)") + [AXIAL_INDICATOR_COLUMN]
            ),
        },
        {
            "id": "R2_material_ratios_v1",
            "description": "Replace fy/fc/Ac/As with strength_ratio and steel_capacity_share",
            "features": ensure_unique(
                without("fy (MPa)", "fc (MPa)", "Ac (mm^2)", "As (mm^2)")
                + [STRENGTH_RATIO_COLUMN, STEEL_CAPACITY_SHARE_COLUMN]
            ),
        },
        {
            "id": "R3_material_ratios_v2",
            "description": "Replace fy/fc/Ac/As with ratio-style composition variables",
            "features": ensure_unique(
                without("fy (MPa)", "fc (MPa)", "Ac (mm^2)", "As (mm^2)")
                + [
                    STRENGTH_RATIO_COLUMN,
                    STEEL_AREA_RATIO_COLUMN,
                    STEEL_CAPACITY_SHARE_COLUMN,
                ]
            ),
        },
        {
            "id": "R4_geometry_compact_v1",
            "description": "Drop Re and te while keeping dimensionless geometry descriptors",
            "features": without("Re (mm)", "te (mm)"),
        },
        {
            "id": "R5_compact_reparam_v1",
            "description": "Compact reparameterized 15-feature candidate",
            "features": list(COMPACT_REPARAM_V1_FEATURES),
        },
        {
            "id": "R6_compact_reparam_v1_no_npl",
            "description": "Compact reparameterized 14-feature candidate without Npl",
            "features": list(COMPACT_REPARAM_V1_NO_NPL_FEATURES),
        },
    ]


def configure_variant(
    base_config: Dict[str, Any],
    *,
    stage: str,
    variant: Dict[str, Any],
    use_optuna: bool = False,
    n_trials: int = 0,
) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)

    config.setdefault("data", {})
    config.setdefault("model", {})
    config.setdefault("paths", {})

    config["data"]["feature_selection"] = {
        "name": variant["id"],
        "include_features": variant["features"],
    }

    config["model"]["use_optuna"] = use_optuna
    config["model"]["n_trials"] = n_trials
    config["model"]["best_params_path"] = str(
        (LOG_ROOT / stage / f"{variant['id']}_best_params.json").relative_to(REPO_ROOT)
    )

    config["paths"]["output_dir"] = str(
        (OUTPUT_ROOT / stage / variant["id"]).relative_to(REPO_ROOT)
    )

    return config


def run_training(config_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "train.py"), "--config", str(config_path)],
        cwd=REPO_ROOT,
        check=True,
    )


def read_report(output_dir: Path) -> Dict[str, Any]:
    report_path = output_dir / "evaluation_report.json"
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_regime_worst_case(report: Dict[str, Any], regime_name: str) -> Dict[str, Any]:
    regime_results = report.get("test_regime_metrics_original_space") or {}
    regime_result = regime_results.get(regime_name) or {}
    worst_cov = regime_result.get("worst_cov_group") or {}
    worst_rmse = regime_result.get("worst_rmse_group") or {}
    return {
        "worst_cov_label": worst_cov.get("label"),
        "worst_cov": (worst_cov.get("metrics") or {}).get("cov"),
        "worst_rmse_label": worst_rmse.get("label"),
        "worst_rmse": (worst_rmse.get("metrics") or {}).get("rmse"),
    }


def summarize_variant(stage: str, variant: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    selection_metrics = report.get("selection_metrics_cv") or {}
    test_metrics = report.get("test_metrics_original_space") or {}
    summary = {
        "stage": stage,
        "id": variant["id"],
        "description": variant["description"],
        "n_features": report.get("n_features"),
        "feature_names": report.get("feature_names"),
        "cv_composite_score": selection_metrics.get("composite_objective"),
        "cv_rmse": selection_metrics.get("rmse"),
        "cv_r2": selection_metrics.get("r2"),
        "cv_cov": selection_metrics.get("cov"),
        "test_rmse": test_metrics.get("rmse"),
        "test_r2": test_metrics.get("r2"),
        "test_cov": test_metrics.get("cov"),
    }
    for regime_name in (
        "scale_npl",
        "section_family",
        "slenderness_state",
        "eccentricity_severity",
    ):
        summary[f"{regime_name}_worst"] = extract_regime_worst_case(report, regime_name)
    return summary


def write_summary_json(stage: str, summaries: List[Dict[str, Any]]) -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = LOG_ROOT / f"{stage}_summary_{date.today().strftime('%Y%m%d')}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    return output_path


def write_report_markdown(
    *,
    stage1_summaries: List[Dict[str, Any]],
    stage2_summaries: List[Dict[str, Any]],
    finalist_summaries: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    lines: List[str] = [
        "# Feature Ablation and Reparameterization Screening",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "## Baseline",
        "",
        "- Base config: `config/config.yaml`",
        "- Training target: `log(psi)`",
        "- Selection basis: CV composite objective in original `Nexp` space",
        "",
    ]

    def render_table(title: str, summaries: List[Dict[str, Any]]) -> None:
        lines.extend([f"## {title}", ""])
        if not summaries:
            lines.extend(["No runs were recorded.", ""])
            return
        lines.append("| ID | n_features | CV J | CV R2 | CV COV | Test R2 | Test COV |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for item in summaries:
            lines.append(
                "| {id} | {n_features} | {cv_composite_score:.4f} | {cv_r2:.4f} | {cv_cov:.4f} | {test_r2:.4f} | {test_cov:.4f} |".format(
                    id=item["id"],
                    n_features=item.get("n_features"),
                    cv_composite_score=float(item["cv_composite_score"]),
                    cv_r2=float(item["cv_r2"]),
                    cv_cov=float(item["cv_cov"]),
                    test_r2=float(item["test_r2"]),
                    test_cov=float(item["test_cov"]),
                )
            )
        lines.append("")

    render_table("Stage 1 Delete-Only Ablation", stage1_summaries)
    render_table("Stage 2 Reparameterization Screening", stage2_summaries)
    render_table("Finalist 200-Trial Optuna Runs", finalist_summaries)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def sort_summaries(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        summaries,
        key=lambda item: float("inf")
        if item.get("cv_composite_score") is None
        else float(item["cv_composite_score"]),
    )


def load_summary_json(stage_slug: str) -> List[Dict[str, Any]]:
    summary_path = LOG_ROOT / f"{stage_slug}_summary_{date.today().strftime('%Y%m%d')}.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def build_finalist_variants(
    stage1_summaries: List[Dict[str, Any]],
    stage2_summaries: List[Dict[str, Any]],
    *,
    top_k: int = 2,
) -> List[Dict[str, Any]]:
    combined = sort_summaries(stage1_summaries + stage2_summaries)
    finalists: List[Dict[str, Any]] = []
    for item in combined[:top_k]:
        finalists.append(
            {
                "id": item["id"],
                "description": f"Finalist from {item['stage']}",
                "features": list(item["feature_names"]),
            }
        )
    return finalists


def run_stage(
    *,
    stage: str,
    base_config: Dict[str, Any],
    variants: List[Dict[str, Any]],
    skip_train: bool,
    use_optuna: bool = False,
    n_trials: int = 0,
) -> List[Dict[str, Any]]:
    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, Any]] = []

    for variant in variants:
        resolved_config = configure_variant(
            base_config,
            stage=stage,
            variant=variant,
            use_optuna=use_optuna,
            n_trials=n_trials,
        )
        config_path = GENERATED_CONFIG_DIR / f"{stage}_{variant['id']}.yaml"
        save_yaml(config_path, resolved_config)

        if not skip_train:
            run_training(config_path)

        report = read_report(REPO_ROOT / resolved_config["paths"]["output_dir"])
        summaries.append(summarize_variant(stage, variant, report))

    return sort_summaries(summaries)


def main() -> int:
    args = parse_args()
    base_config = load_yaml(Path(args.base_config))

    stage1_summaries: List[Dict[str, Any]] = []
    stage2_summaries: List[Dict[str, Any]] = []
    finalist_summaries: List[Dict[str, Any]] = []

    if args.stage in {"all", "stage1"}:
        stage1_summaries = run_stage(
            stage="stage1",
            base_config=base_config,
            variants=build_stage1_variants(),
            skip_train=args.skip_train,
        )
        write_summary_json("feature_ablation_stage1", stage1_summaries)

    if args.stage in {"stage2", "finalists"}:
        stage1_summaries = load_summary_json("feature_ablation_stage1")

    if args.stage in {"all", "stage2"}:
        if not stage1_summaries:
            raise ValueError("Stage 2 requires at least one Stage 1 summary")
        winner_features = list(stage1_summaries[0]["feature_names"])
        stage2_summaries = run_stage(
            stage="stage2",
            base_config=base_config,
            variants=build_stage2_variants(winner_features),
            skip_train=args.skip_train,
        )
        write_summary_json("feature_ablation_stage2", stage2_summaries)

    if args.stage == "finalists":
        stage2_summaries = load_summary_json("feature_ablation_stage2")

    if args.stage == "finalists":
        finalist_summaries = run_stage(
            stage="finalists",
            base_config=base_config,
            variants=build_finalist_variants(stage1_summaries, stage2_summaries),
            skip_train=args.skip_train,
            use_optuna=True,
            n_trials=200,
        )
        write_summary_json("feature_ablation_finalists", finalist_summaries)

    report_path = DOC_ROOT / f"feature_ablation_screening_{date.today().strftime('%Y%m%d')}.md"
    write_report_markdown(
        stage1_summaries=stage1_summaries,
        stage2_summaries=stage2_summaries,
        finalist_summaries=finalist_summaries,
        output_path=report_path,
    )
    print(f"Wrote screening report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
