#!/usr/bin/env python3
"""
Prediction Script for CFST XGBoost Pipeline

This script makes predictions using a trained XGBoost model:
1. Load trained model and preprocessor
2. Load input data
3. Make predictions
4. Export results

Usage:
    python predict.py --model models/my_model --input data/features.csv --output predictions.csv
    python predict.py --model models/my_model --input data/single_sample.csv --single
"""

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

from src.utils.logger import setup_logger
from src.utils.model_utils import load_metadata, load_model_from_directory
from src.predictor import Predictor, export_predictions

logger = setup_logger(__name__)


def make_predictions(model_dir: str, input_data_path: str,
                    output_path: str = None, single: bool = False) -> pd.DataFrame:
    """
    Make predictions using trained model.

    Args:
        model_dir: Directory containing trained model files
        input_data_path: Path to input data CSV file
        output_path: Path to save predictions (optional)
        single: Whether to treat input as single sample (default: False)

    Returns:
        DataFrame with predictions

    Raises:
        Exception: If any step fails
    """
    logger.info("=" * 80)
    logger.info("CFST XGBOOST PIPELINE - PREDICTION STARTED")
    logger.info("=" * 80)

    try:
        # Step 1: Load model and artifacts
        logger.info("Step 1: Loading model and artifacts...")

        if not Path(model_dir).exists():
            error_msg = f"Model directory not found: {model_dir}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        model, preprocessor, feature_names = load_model_from_directory(model_dir)
        metadata_path = Path(model_dir) / "training_metadata.json"
        metadata = load_metadata(str(metadata_path)) if metadata_path.exists() else {}
        logger.info(f"Model loaded from {model_dir}")
        logger.info(f"Feature names loaded: {len(feature_names) if feature_names else 0} features")

        # Step 2: Load input data
        logger.info("\nStep 2: Loading input data...")

        if not Path(input_data_path).exists():
            error_msg = f"Input data file not found: {input_data_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        input_df = pd.read_csv(input_data_path)
        logger.info(f"Input data loaded: {len(input_df)} samples, {len(input_df.columns)} features")

        # Step 3: Validate features
        logger.info("\nStep 3: Validating features...")
        logger.info("Deferring feature derivation/validation to Predictor")

        # Step 4: Initialize predictor
        logger.info("\nStep 4: Initializing predictor...")
        predictor = Predictor(model, preprocessor, feature_names, metadata=metadata)
        logger.info("Predictor initialized")

        # Step 5: Make predictions
        logger.info("\nStep 5: Making predictions...")

        if single:
            # Single prediction
            if len(input_df) != 1:
                logger.warning(f"Single prediction mode but input has {len(input_df)} rows. Using first row.")
                input_df = input_df.iloc[:1]

            prediction = predictor.predict_single(input_df)
            logger.info(f"Single prediction: {prediction:.4f}")

            # Create result DataFrame
            results_df = input_df.copy()
            results_df['prediction'] = prediction

        else:
            # Batch predictions
            logger.info(f"Making predictions for {len(input_df)} samples...")
            predictions = predictor.predict(input_df)

            # Create results DataFrame
            results_df = input_df.copy()
            results_df['prediction'] = predictions

            logger.info(f"Predictions completed: {len(predictions)} samples")
            logger.info(f"Prediction statistics:")
            logger.info(f"  Min: {predictions.min():.4f}")
            logger.info(f"  Max: {predictions.max():.4f}")
            logger.info(f"  Mean: {predictions.mean():.4f}")
            logger.info(f"  Std: {predictions.std():.4f}")

        # Step 6: Export predictions
        if output_path:
            logger.info("\nStep 6: Exporting predictions...")
            export_predictions(input_df, predictions if not single else [prediction], output_path)
            logger.info(f"Predictions exported to {output_path}")
        else:
            logger.info("\nStep 6: Skipping export (no output path provided)")

        # Step 7: Final summary
        logger.info("\n" + "=" * 80)
        logger.info("PREDICTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        if not single:
            logger.info(f"Total samples: {len(results_df)}")
            logger.info(f"Predictions range: [{results_df['prediction'].min():.4f}, {results_df['prediction'].max():.4f}]")
        else:
            logger.info(f"Single prediction: {prediction:.4f}")

        logger.info("=" * 80)

        return results_df

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def main():
    """Main entry point for the prediction script."""
    parser = argparse.ArgumentParser(
        description='Make predictions with trained CFST XGBoost Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Make predictions on dataset
  python predict.py --model models/xgboost_model --input data/features.csv --output predictions.csv

  # Single prediction
  python predict.py --model models/xgboost_model --input data/single_sample.csv --single

  # Display predictions without saving
  python predict.py --model models/xgboost_model --input data/features.csv

  # Custom output location
  python predict.py --model models/xgboost_model --input data/features.csv --output results/my_predictions.csv
        """
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Directory containing trained model files (model, preprocessor, etc.)'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input data CSV file (features only, no target)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to save predictions CSV file (optional)'
    )

    parser.add_argument(
        '--single',
        action='store_true',
        help='Treat input as single sample for prediction'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Check if model directory exists
    if not Path(args.model).exists():
        logger.error(f"Model directory not found: {args.model}")
        logger.error("Please check the directory path")
        sys.exit(1)

    # Check if required model files exist
    required_files = ['xgboost_model.pkl', 'feature_names.json']
    missing_files = []

    for file in required_files:
        file_path = Path(args.model) / file
        if not file_path.exists():
            missing_files.append(file)

    if missing_files:
        logger.error(f"Missing required files in model directory: {missing_files}")
        logger.error("Please ensure the model directory contains all trained model files")
        sys.exit(1)

    # Check if input file exists
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    try:
        # Run prediction
        results_df = make_predictions(args.model, args.input, args.output, args.single)

        # Display first few predictions
        if not args.single:
            print("\nFirst 5 predictions:")
            print(results_df[['prediction']].head().to_string())

        logger.info("Prediction completed successfully!")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.error("Use --verbose for more details")
        if args.verbose:
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
