'''
Model Utilities Module for CFST XGBoost Pipeline

This module handles saving and loading trained models, preprocessors, and metadata.
'''

import joblib
import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Any, Optional
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_model(
    model: Any,
    preprocessor: Any,
    feature_names: List[str],
    output_dir: str,
    model_name: str = "xgboost_model.pkl",
    preprocessor_name: str = "preprocessor.pkl",
    feature_names_name: str = "feature_names.json",
    metadata: Optional[dict] = None,
    metadata_name: str = "training_metadata.json"
) -> None:
    """
    Save trained model, preprocessor, and metadata.

    Args:
        model: Trained XGBoost model
        preprocessor: Fitted preprocessor
        feature_names: List of feature names
        output_dir: Output directory path
        model_name: Model file name (default: xgboost_model.pkl)
        preprocessor_name: Preprocessor file name (default: preprocessor.pkl)
        feature_names_name: Feature names file name (default: feature_names.json)
        metadata: Additional metadata to save (optional)
        metadata_name: Metadata file name (default: training_metadata.json)

    Raises:
        Exception: If saving fails
    """
    logger.info(f"Saving model and artifacts to {output_dir}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_path / model_name
    try:
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise Exception(f"Failed to save model: {str(e)}")

    # Save preprocessor
    if preprocessor is not None:
        preprocessor_path = output_path / preprocessor_name
        try:
            joblib.dump(preprocessor, preprocessor_path)
            logger.info(f"Preprocessor saved to {preprocessor_path}")
        except Exception as e:
            logger.error(f"Failed to save preprocessor: {str(e)}")
            raise Exception(f"Failed to save preprocessor: {str(e)}")

    # Save feature names
    feature_path = output_path / feature_names_name
    try:
        with open(feature_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        logger.info(f"Feature names saved to {feature_path}")
    except Exception as e:
        logger.error(f"Failed to save feature names: {str(e)}")
        raise Exception(f"Failed to save feature names: {str(e)}")

    # Save metadata if provided
    if metadata is not None:
        metadata_path = output_path / metadata_name
        try:
            # Convert any non-serializable objects
            serializable_metadata = _make_serializable(metadata)

            with open(metadata_path, 'w') as f:
                json.dump(serializable_metadata, f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
            # Don't raise exception for metadata, just log the error
            logger.warning("Model and preprocessor saved, but metadata failed")

    logger.info("Model and artifacts saved successfully")


def load_model(
    model_path: str,
    preprocessor_path: Optional[str] = None,
    feature_names_path: Optional[str] = None
) -> Tuple[Any, Optional[Any], Optional[List[str]]]:
    """
    Load trained model, preprocessor, and feature names.

    Args:
        model_path: Path to model file
        preprocessor_path: Path to preprocessor file (optional)
        feature_names_path: Path to feature names file (optional)

    Returns:
        Tuple of (model, preprocessor, feature_names)

    Raises:
        Exception: If loading fails
    """
    logger.info(f"Loading model from {model_path}")

    # Load model
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise Exception(f"Failed to load model: {str(e)}")

    # Load preprocessor
    preprocessor = None
    if preprocessor_path is not None and Path(preprocessor_path).exists():
        try:
            preprocessor = joblib.load(preprocessor_path)
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {str(e)}")
            logger.warning("Continuing without preprocessor")

    # Load feature names
    feature_names = None
    if feature_names_path is not None and Path(feature_names_path).exists():
        try:
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
            logger.info(f"Feature names loaded from {feature_names_path}")
            logger.info(f"Number of features: {len(feature_names)}")
        except Exception as e:
            logger.error(f"Failed to load feature names: {str(e)}")
            logger.warning("Continuing without feature names")

    return model, preprocessor, feature_names


def load_model_from_directory(
    model_dir: str,
    model_name: str = "xgboost_model.pkl",
    preprocessor_name: str = "preprocessor.pkl",
    feature_names_name: str = "feature_names.json"
) -> Tuple[Any, Optional[Any], Optional[List[str]]]:
    """
    Load model and artifacts from a directory.

    Args:
        model_dir: Directory containing model artifacts
        model_name: Model file name (default: xgboost_model.pkl)
        preprocessor_name: Preprocessor file name (default: preprocessor.pkl)
        feature_names_name: Feature names file name (default: feature_names.json)

    Returns:
        Tuple of (model, preprocessor, feature_names)
    """
    model_dir_path = Path(model_dir)

    if not model_dir_path.exists():
        error_msg = f"Model directory not found: {model_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    model_path = model_dir_path / model_name
    preprocessor_path = model_dir_path / preprocessor_name
    feature_names_path = model_dir_path / feature_names_name

    return load_model(
        str(model_path),
        str(preprocessor_path) if preprocessor_path.exists() else None,
        str(feature_names_path) if feature_names_path.exists() else None
    )


def save_metadata(metadata: dict, output_path: str) -> None:
    """
    Save metadata to JSON file.

    Args:
        metadata: Metadata dictionary
        output_path: Output file path

    Raises:
        Exception: If saving fails
    """
    try:
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Make metadata serializable
        serializable_metadata = _make_serializable(metadata)

        with open(output_path, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)

        logger.info(f"Metadata saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {str(e)}")
        raise Exception(f"Failed to save metadata: {str(e)}")


def load_metadata(metadata_path: str) -> dict:
    """
    Load metadata from JSON file.

    Args:
        metadata_path: Path to metadata file

    Returns:
        Metadata dictionary

    Raises:
        Exception: If loading fails
    """
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        logger.info(f"Metadata loaded from {metadata_path}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to load metadata: {str(e)}")
        raise Exception(f"Failed to load metadata: {str(e)}")


def _make_serializable(obj: Any) -> Any:
    """
    Convert non-serializable objects to serializable format.

    Args:
        obj: Object to convert

    Returns:
        Serializable version of the object
    """
    if isinstance(obj, dict):
        return {key: _make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return _make_serializable(obj.__dict__)
    else:
        return str(obj)


def validate_model(model: Any, X_sample: pd.DataFrame) -> bool:
    """
    Validate that a model can make predictions.

    Args:
        model: Trained model
        X_sample: Sample features for prediction

    Returns:
        True if model is valid, False otherwise
    """
    try:
        # Try to make a prediction
        prediction = model.predict(X_sample.iloc[:1])

        # Check if prediction is reasonable
        if prediction is None:
            logger.error("Model returned None prediction")
            return False

        if not np.isfinite(prediction).all():
            logger.error("Model returned non-finite prediction")
            return False

        logger.info("Model validation passed")
        return True
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False


def get_model_size(model_path: str) -> int:
    """
    Get the size of a model file in bytes.

    Args:
        model_path: Path to model file

    Returns:
        File size in bytes
    """
    try:
        return Path(model_path).stat().st_size
    except Exception as e:
        logger.error(f"Failed to get model size: {str(e)}")
        return 0


def list_model_files(model_dir: str) -> dict:
    """
    List all model-related files in a directory.

    Args:
        model_dir: Directory containing model files

    Returns:
        Dictionary with file information
    """
    model_dir_path = Path(model_dir)

    if not model_dir_path.exists():
        logger.warning(f"Model directory not found: {model_dir}")
        return {}

    files_info = {}
    for file_path in model_dir_path.iterdir():
        if file_path.is_file():
            files_info[file_path.name] = {
                'size_bytes': file_path.stat().st_size,
                'size_mb': file_path.stat().st_size / (1024 * 1024),
                'modified': file_path.stat().st_mtime
            }

    return files_info


def save_best_params(
    best_params: dict,
    best_score: float,
    trial_number: int,
    n_trials: int,
    output_path: str = 'logs/best_params.json',
    context_hash: Optional[str] = None,
    data_file: Optional[str] = None,
    study_name: Optional[str] = None,
    storage_url: Optional[str] = None,
) -> None:
    """
    Save best Optuna parameters to JSON file.

    Args:
        best_params: Dictionary of best hyperparameters
        best_score: Best RMSE score achieved
        trial_number: Trial number that achieved the best score
        n_trials: Total number of trials run
        output_path: Output file path (default: logs/best_params.json)
        context_hash: Hash of data/config context for safe reuse
        data_file: Absolute data file path used for optimization
        study_name: Optuna study name
        storage_url: Optuna storage URL
    """
    from datetime import datetime

    data = {
        'trial_number': trial_number,
        'best_rmse': best_score,
        'parameters': best_params,
        'timestamp': datetime.now().isoformat(),
        'n_trials_total': n_trials,
        'context_hash': context_hash,
        'data_file': data_file,
        'study_name': study_name,
        'storage_url': storage_url,
    }

    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Best parameters saved to {output_path}")
    logger.info(f"  Best RMSE: {best_score:.4f} (Trial {trial_number})")


def load_best_params(
    input_path: str = 'logs/best_params.json',
    expected_context_hash: Optional[str] = None,
) -> Optional[dict]:
    """
    Load best Optuna parameters from JSON file.

    Args:
        input_path: Path to best parameters file (default: logs/best_params.json)
        expected_context_hash: Expected context hash for strict compatibility check

    Returns:
        Dictionary of best parameters, or None if file doesn't exist or context mismatch

    Note:
        Returns None if file doesn't exist (first run without Optuna).
    """
    path = Path(input_path)
    if not path.exists():
        logger.warning(f"No saved parameters found at {input_path}")
        return None

    with open(path, 'r') as f:
        data = json.load(f)

    if 'parameters' not in data or not isinstance(data['parameters'], dict):
        logger.warning(f"Invalid best parameters file format at {input_path}")
        return None

    saved_context_hash = data.get('context_hash')
    if expected_context_hash is not None:
        if saved_context_hash is None:
            logger.warning(
                "Saved best parameters do not include context_hash; ignoring to avoid cross-dataset reuse"
            )
            return None
        if saved_context_hash != expected_context_hash:
            logger.warning(
                "Saved best parameters context hash mismatch "
                f"(expected={expected_context_hash}, found={saved_context_hash}); ignoring"
            )
            return None

    logger.info(f"Loaded best parameters from {input_path}")
    if 'best_rmse' in data and 'trial_number' in data:
        logger.info(f"  Best RMSE: {data['best_rmse']:.4f} (Trial {data['trial_number']})")
    if 'timestamp' in data:
        logger.info(f"  Saved on: {data['timestamp']}")
    if 'n_trials_total' in data:
        logger.info(f"  Total trials: {data['n_trials_total']}")
    if saved_context_hash:
        logger.info(f"  Context hash: {saved_context_hash}")
    if data.get('study_name'):
        logger.info(f"  Study name: {data['study_name']}")

    return data['parameters']
