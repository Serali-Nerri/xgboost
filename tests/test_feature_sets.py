import pandas as pd
import pytest

from src.feature_sets import CURRENT_21_FEATURES, resolve_feature_selection
from src.preprocessor import Preprocessor


def test_resolve_feature_selection_supports_named_presets():
    available_columns = CURRENT_21_FEATURES + ["axial_flag", "section_family"]

    resolved = resolve_feature_selection(
        available_columns,
        columns_to_drop=["b (mm)", "h (mm)"],
        feature_selection_config={"preset": "current_21", "name": "baseline"},
    )

    assert resolved["name"] == "baseline"
    assert resolved["preset"] == "current_21"
    assert resolved["selection_source"] == "preset"
    assert resolved["selected_features"] == CURRENT_21_FEATURES
    assert resolved["selected_feature_count"] == 21


def test_resolve_feature_selection_rejects_drop_conflicts():
    with pytest.raises(ValueError, match="conflict with config.data.columns_to_drop"):
        resolve_feature_selection(
            ["a", "b"],
            columns_to_drop=["a"],
            feature_selection_config={"include_features": ["a"]},
        )


def test_preprocessor_include_features_respects_explicit_order():
    X = pd.DataFrame(
        {
            "f1": [1.0, 2.0],
            "f2": [3.0, 4.0],
            "f3": [5.0, 6.0],
            "label": ["a", "b"],
        }
    )

    preprocessor = Preprocessor(
        columns_to_drop=[],
        include_features=["f3", "f1"],
    )
    transformed = preprocessor.fit_transform(X)

    assert list(transformed.columns) == ["f3", "f1"]
    assert preprocessor.get_remaining_features() == ["f3", "f1"]
    assert "label" in preprocessor.get_dropped_columns()


def test_preprocessor_include_features_rejects_non_numeric_columns():
    X = pd.DataFrame({"numeric": [1.0, 2.0], "kind": ["axial", "eccentric"]})

    preprocessor = Preprocessor(
        columns_to_drop=[],
        include_features=["numeric", "kind"],
    )

    with pytest.raises(ValueError, match="must be numeric or boolean columns"):
        preprocessor.fit(X)
