#!/usr/bin/env python3
"""
Run a focused eccentricity reparameterization experiment round.
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
    DOUBLE_CURVATURE_E_OVER_H_COLUMN,
    E_MIN_OVER_H_COLUMN,
    END_ASYMMETRY_RATIO_COLUMN,
    REVERSE_CURVATURE_FLAG_COLUMN,
    SINGLE_CURVATURE_E_OVER_H_COLUMN,
)
from src.feature_sets import (  # noqa: E402
    CURRENT_21_FEATURES,
    HISTORICAL_18_PLUS_BH_LH_FEATURES,
)


GENERATED_CONFIG_DIR = (
    REPO_ROOT / "config" / "experiments" / "eccentricity_reparameterization_generated"
)
OUTPUT_ROOT = REPO_ROOT / "output" / "eccentricity_reparameterization"
LOG_ROOT = REPO_ROOT / "logs" / "eccentricity_reparameterization"
DOC_ROOT = REPO_ROOT / "doc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a focused eccentricity reparameterization experiment suite.",
    )
    parser.add_argument(
        "--base-config",
        default=str(REPO_ROOT / "config" / "config.yaml"),
        help="Base config path used as the experiment template.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and summarize from existing outputs only.",
    )
    parser.add_argument(
        "--stage",
        choices=["screening", "finalist", "all"],
        default="screening",
        help="Which stage to run.",
    )
    parser.add_argument(
        "--finalist-trials",
        type=int,
        default=200,
        help="Optuna trials for the finalist rerun.",
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


def replace_features(
    base_features: List[str],
    *,
    remove: Iterable[str],
    add: Iterable[str],
) -> List[str]:
    removed = set(remove)
    kept = [feature for feature in base_features if feature not in removed]
    return ensure_unique(kept + list(add))


def build_screening_variants() -> List[Dict[str, Any]]:
    d7 = list(CURRENT_21_FEATURES)
    d6 = list(HISTORICAL_18_PLUS_BH_LH_FEATURES)

    mode_balance_add = [
        E_MIN_OVER_H_COLUMN,
        END_ASYMMETRY_RATIO_COLUMN,
        SINGLE_CURVATURE_E_OVER_H_COLUMN,
        DOUBLE_CURVATURE_E_OVER_H_COLUMN,
    ]
    asymmetry_flag_add = [
        E_MIN_OVER_H_COLUMN,
        END_ASYMMETRY_RATIO_COLUMN,
        REVERSE_CURVATURE_FLAG_COLUMN,
    ]
    mode_balance_full_add = [
        E_MIN_OVER_H_COLUMN,
        END_ASYMMETRY_RATIO_COLUMN,
        SINGLE_CURVATURE_E_OVER_H_COLUMN,
        DOUBLE_CURVATURE_E_OVER_H_COLUMN,
        REVERSE_CURVATURE_FLAG_COLUMN,
    ]

    return [
        {
            "id": "E1_d7_mode_balance_keep_ebar",
            "description": (
                "D7 anchor: replace raw end eccentricities with "
                "magnitude/asymmetry/mode components, keep e_bar"
            ),
            "features": replace_features(
                d7,
                remove=["e1 (mm)", "e2 (mm)", "e/h", "e1/e2"],
                add=mode_balance_add,
            ),
        },
        {
            "id": "E2_d7_asymmetry_flag_keep_ebar",
            "description": (
                "D7 anchor: replace raw end eccentricities with min-ratio and "
                "reverse-curvature flag, keep e/h and e_bar"
            ),
            "features": replace_features(
                d7,
                remove=["e1 (mm)", "e2 (mm)", "e1/e2"],
                add=asymmetry_flag_add,
            ),
        },
        {
            "id": "E3_d7_mode_balance_no_ebar",
            "description": "D7 anchor: test whether mode decomposition can replace e_bar",
            "features": replace_features(
                d7,
                remove=["e1 (mm)", "e2 (mm)", "e/h", "e1/e2", "e_bar"],
                add=mode_balance_full_add,
            ),
        },
        {
            "id": "E4_d6_mode_balance_keep_ebar",
            "description": (
                "D6 anchor: replace raw end eccentricities with "
                "magnitude/asymmetry/mode components, keep e_bar"
            ),
            "features": replace_features(
                d6,
                remove=["e1 (mm)", "e2 (mm)", "e/h", "e1/e2"],
                add=mode_balance_add,
            ),
        },
        {
            "id": "E5_d6_asymmetry_flag_keep_ebar",
            "description": (
                "D6 anchor: replace raw end eccentricities with min-ratio and "
                "reverse-curvature flag, keep e/h and e_bar"
            ),
            "features": replace_features(
                d6,
                remove=["e1 (mm)", "e2 (mm)", "e1/e2"],
                add=asymmetry_flag_add,
            ),
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


def read_report(report_path: Path) -> Dict[str, Any]:
    with report_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    selection_metrics = report.get("selection_metrics_cv") or {}
    test_metrics = report.get("test_metrics_original_space") or {}
    feature_names = report.get("feature_names")
    n_features = report.get("n_features")
    if n_features is None and isinstance(feature_names, list):
        n_features = len(feature_names)
    return {
        "n_features": n_features,
        "feature_names": feature_names,
        "cv_composite_score": selection_metrics.get("composite_objective"),
        "cv_rmse": selection_metrics.get("rmse"),
        "cv_r2": selection_metrics.get("r2"),
        "cv_cov": selection_metrics.get("cov"),
        "test_rmse": test_metrics.get("rmse"),
        "test_r2": test_metrics.get("r2"),
        "test_cov": test_metrics.get("cov"),
    }


def summarize_variant(stage: str, variant: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    summary = summarize_report(report)
    summary.update(
        {
            "stage": stage,
            "id": variant["id"],
            "description": variant["description"],
        }
    )
    return summary


def sort_summaries(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        summaries,
        key=lambda item: float("inf")
        if item.get("cv_composite_score") is None
        else float(item["cv_composite_score"]),
    )


def load_anchor_summaries() -> List[Dict[str, Any]]:
    anchors = [
        {
            "id": "A0_d6",
            "description": "Existing D6 200-trial compact anchor",
            "report_path": REPO_ROOT
            / "output"
            / "feature_ablation"
            / "finalists"
            / "D6_hist18_plus_b_over_h_l_over_h"
            / "evaluation_report.json",
            "n_features_hint": len(HISTORICAL_18_PLUS_BH_LH_FEATURES),
            "feature_names_hint": list(HISTORICAL_18_PLUS_BH_LH_FEATURES),
        },
        {
            "id": "A1_d7",
            "description": "Existing D7 current 21-feature mainline",
            "report_path": REPO_ROOT
            / "output"
            / "psi_over_npl_log_original_200"
            / "evaluation_report.json",
            "n_features_hint": len(CURRENT_21_FEATURES),
            "feature_names_hint": list(CURRENT_21_FEATURES),
        },
    ]

    summaries: List[Dict[str, Any]] = []
    for anchor in anchors:
        report = read_report(anchor["report_path"])
        summary = summarize_report(report)
        if summary.get("n_features") is None:
            summary["n_features"] = anchor["n_features_hint"]
        if summary.get("feature_names") is None:
            summary["feature_names"] = anchor["feature_names_hint"]
        summary.update(
            {
                "stage": "anchor",
                "id": anchor["id"],
                "description": anchor["description"],
            }
        )
        summaries.append(summary)
    return summaries


def write_summary_json(stage_slug: str, summaries: List[Dict[str, Any]]) -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = LOG_ROOT / f"{stage_slug}_summary_{date.today().strftime('%Y%m%d')}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    return output_path


def load_summary_json(stage_slug: str) -> List[Dict[str, Any]]:
    summary_path = LOG_ROOT / f"{stage_slug}_summary_{date.today().strftime('%Y%m%d')}.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_report_markdown(
    *,
    anchor_summaries: List[Dict[str, Any]],
    screening_summaries: List[Dict[str, Any]],
    finalist_summaries: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    lines: List[str] = [
        "# Eccentricity Reparameterization Round",
        "",
        f"Date: {date.today().isoformat()}",
        "",
    ]

    def render_table(title: str, summaries: List[Dict[str, Any]]) -> None:
        lines.extend([f"## {title}", ""])
        if not summaries:
            lines.extend(["No runs were recorded.", ""])
            return
        lines.append("| ID | n_features | CV J | CV R2 | CV COV | Test RMSE | Test R2 | Test COV |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for item in summaries:
            lines.append(
                "| {id} | {n_features} | {cv_composite_score:.4f} | {cv_r2:.4f} | {cv_cov:.4f} | {test_rmse:.2f} | {test_r2:.4f} | {test_cov:.4f} |".format(
                    id=item["id"],
                    n_features=int(item["n_features"]),
                    cv_composite_score=float(item["cv_composite_score"]),
                    cv_r2=float(item["cv_r2"]),
                    cv_cov=float(item["cv_cov"]),
                    test_rmse=float(item["test_rmse"]),
                    test_r2=float(item["test_r2"]),
                    test_cov=float(item["test_cov"]),
                )
            )
        lines.append("")

    render_table("Anchors", anchor_summaries)
    render_table("Screening", screening_summaries)
    render_table("Finalist", finalist_summaries)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_screening(
    *,
    base_config: Dict[str, Any],
    variants: List[Dict[str, Any]],
    skip_train: bool,
) -> List[Dict[str, Any]]:
    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, Any]] = []
    for variant in variants:
        config = configure_variant(base_config, stage="screening", variant=variant)
        config_path = GENERATED_CONFIG_DIR / f"screening_{variant['id']}.yaml"
        save_yaml(config_path, config)
        if not skip_train:
            run_training(config_path)
        report = read_report(REPO_ROOT / config["paths"]["output_dir"] / "evaluation_report.json")
        summaries.append(summarize_variant("screening", variant, report))
    return sort_summaries(summaries)


def run_finalist(
    *,
    base_config: Dict[str, Any],
    variant: Dict[str, Any],
    skip_train: bool,
    n_trials: int,
) -> Dict[str, Any]:
    config = configure_variant(
        base_config,
        stage="finalist",
        variant=variant,
        use_optuna=True,
        n_trials=n_trials,
    )
    config_path = GENERATED_CONFIG_DIR / f"finalist_{variant['id']}.yaml"
    save_yaml(config_path, config)
    if not skip_train:
        run_training(config_path)
    report = read_report(REPO_ROOT / config["paths"]["output_dir"] / "evaluation_report.json")
    return summarize_variant("finalist", variant, report)


def main() -> int:
    args = parse_args()
    base_config = load_yaml(Path(args.base_config))
    variants = build_screening_variants()
    anchor_summaries = load_anchor_summaries()
    screening_summaries: List[Dict[str, Any]] = []
    finalist_summaries: List[Dict[str, Any]] = []

    if args.stage in {"screening", "all"}:
        screening_summaries = run_screening(
            base_config=base_config,
            variants=variants,
            skip_train=args.skip_train,
        )
        write_summary_json("eccentricity_screening", screening_summaries)

    if args.stage == "finalist":
        screening_summaries = load_summary_json("eccentricity_screening")
        top_variant = next(
            variant for variant in variants if variant["id"] == screening_summaries[0]["id"]
        )
        finalist_summaries = [
            run_finalist(
                base_config=base_config,
                variant=top_variant,
                skip_train=args.skip_train,
                n_trials=args.finalist_trials,
            )
        ]
        write_summary_json("eccentricity_finalist", finalist_summaries)

    report_path = DOC_ROOT / f"eccentricity_reparameterization_{date.today().strftime('%Y%m%d')}.md"
    write_report_markdown(
        anchor_summaries=anchor_summaries,
        screening_summaries=screening_summaries,
        finalist_summaries=finalist_summaries,
        output_path=report_path,
    )
    print(f"Wrote eccentricity report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
