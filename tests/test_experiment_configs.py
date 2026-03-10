from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIGS = [
    REPO_ROOT / "config/experiments/raw_original_metric.yaml",
    REPO_ROOT / "config/experiments/log_original_metric.yaml",
    REPO_ROOT / "config/experiments/log_transformed_metric.yaml",
]


def test_experiment_configs_follow_psi_mainline_contract():
    for config_path in EXPERIMENT_CONFIGS:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        assert config["data"]["target_mode"] == "psi_over_npl"
        assert config["data"]["split"]["auxiliary_features"][0]["column"] == "lambda_bar"
        assert config["model"]["selection_objective"]["metric_space"] == "original_nexp"
        assert config["model"]["selection_objective"]["rmse_normalizer"] == "mean_actual"
        assert config["evaluation"]["regime_analysis"]["reference_split"] == "train"
        assert config["evaluation"]["regime_analysis"]["sort_metric"] == "cov"
        regime_names = [regime["name"] for regime in config["evaluation"]["regime_analysis"]["regimes"]]
        assert regime_names == [
            "axiality",
            "section_family",
            "slenderness_state",
            "scale_npl",
            "eccentricity_severity",
            "confinement_level",
        ]
