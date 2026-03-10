"""
Preprocessor Module for CFST XGBoost Pipeline

This module handles data preprocessing including column dropping and data cleaning.
"""

import pandas as pd
from typing import List
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Data preprocessor for CFST XGBoost pipeline.

    Handles column removal (especially geometric parameters) and basic data cleaning.
    Modeled after scikit-learn's Transformer interface for consistency.
    """

    def __init__(self, columns_to_drop: List[str]):
        """
        Initialize Preprocessor.

        Args:
            columns_to_drop: List of column names to drop from the data
        """
        self.columns_to_drop = columns_to_drop
        self.remaining_features: List[str] = []
        self.auto_dropped_non_numeric: List[str] = []
        self.is_fitted = False
        logger.info(f"Preprocessor initialized with {len(columns_to_drop)} columns to drop: {columns_to_drop}")

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        """
        Fit the preprocessor to the data.

        Identifies which columns will remain after dropping specified columns.

        Args:
            X: Input features DataFrame

        Returns:
            self: Fitted preprocessor instance

        Raises:
            ValueError: If required columns to drop are not found in data
        """
        logger.info("Fitting preprocessor to data")

        # Check if columns to drop exist in the data
        missing_columns = set(self.columns_to_drop) - set(X.columns)
        if missing_columns:
            error_msg = f"Columns to drop not found in data: {missing_columns}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        candidate_features = [col for col in X.columns if col not in self.columns_to_drop]
        self.auto_dropped_non_numeric = [
            column
            for column in candidate_features
            if not pd.api.types.is_numeric_dtype(X[column])
            and not pd.api.types.is_bool_dtype(X[column])
        ]
        self.remaining_features = [
            column
            for column in candidate_features
            if column not in self.auto_dropped_non_numeric
        ]

        logger.info(f"Original features: {len(X.columns)}")
        logger.info(f"Columns to drop: {len(self.columns_to_drop)}")
        if self.auto_dropped_non_numeric:
            logger.warning(
                "Dropping non-numeric columns from model features: %s",
                self.auto_dropped_non_numeric,
            )
        logger.info(f"Remaining features: {len(self.remaining_features)}")
        logger.debug(f"Remaining feature names: {self.remaining_features}")

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by dropping specified columns.

        Args:
            X: Input features DataFrame

        Returns:
            DataFrame with specified columns removed

        Raises:
            ValueError: If preprocessor is not fitted or if data is missing expected columns
        """
        if not self.is_fitted:
            error_msg = "Preprocessor must be fitted before transform. Call fit() first."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Transforming data with shape {X.shape}")

        # Check if all remaining features are present
        missing_features = set(self.remaining_features) - set(X.columns)
        if missing_features:
            error_msg = f"Missing expected features: {missing_features}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Drop columns and return remaining features
        X_transformed = X[self.remaining_features].copy()

        logger.info(f"Transformed data shape: {X_transformed.shape}")
        logger.debug(f"Dropped {len(self.columns_to_drop)} columns: {self.columns_to_drop}")

        return X_transformed

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the preprocessor and transform the data in one step.

        Args:
            X: Input features DataFrame

        Returns:
            DataFrame with specified columns removed
        """
        return self.fit(X).transform(X)

    def get_remaining_features(self) -> List[str]:
        """
        Get the list of features that will remain after dropping.

        Returns:
            List of remaining feature names

        Raises:
            ValueError: If preprocessor is not fitted yet
        """
        if not self.is_fitted:
            error_msg = "Preprocessor must be fitted before getting remaining features. Call fit() first."
            logger.error(error_msg)
            raise ValueError(error_msg)

        return self.remaining_features.copy()

    def get_dropped_columns(self) -> List[str]:
        """
        Get the list of columns that will be dropped.

        Returns:
            List of columns to drop
        """
        return self.columns_to_drop.copy() + self.auto_dropped_non_numeric.copy()

    def is_column_dropped(self, column: str) -> bool:
        """
        Check if a specific column will be dropped.

        Args:
            column: Column name to check

        Returns:
            True if column will be dropped, False otherwise
        """
        return column in self.columns_to_drop

    def check_missing_values(self, X: pd.DataFrame) -> dict:
        """
        Check for missing values in the data.

        Args:
            X: DataFrame to check

        Returns:
            Dictionary with column names as keys and missing value counts as values
        """
        missing_counts = X.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]

        if missing_counts.empty:
            logger.info("No missing values found")
        else:
            logger.warning(f"Found missing values:\n{missing_counts}")

        return missing_counts.to_dict()

    def get_feature_stats(self, X: pd.DataFrame) -> dict:
        """
        Get basic statistics for the features.

        Args:
            X: Features DataFrame

        Returns:
            Dictionary with statistics
        """
        stats = {
            "n_samples": len(X),
            "n_features": len(X.columns),
            "feature_names": list(X.columns),
            "missing_values": self.check_missing_values(X),
            "numeric_features": X.select_dtypes(include=[np.number]).columns.tolist(),
        }

        logger.info(f"Feature stats: {len(X.columns)} features, {len(X)} samples")

        return stats
