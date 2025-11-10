#!/usr/bin/env python3
"""
Unified model runner for all NTRS earnings prediction models.

Usage:
    python run_model.py --model xgb
    python run_model.py --model rf --results-dir custom_results
    python run_model.py --all  # Run all models
"""

import argparse
import os
import sys
from pathlib import Path

from models.utils import load_data

# Model registry
MODEL_REGISTRY = {
    'ols': ('models.ols_model', 'OLSModel'),
    'lasso': ('models.lasso_model', 'LassoModel'),
    'ridge': ('models.ridge_model', 'RidgeModel'),
    'ar': ('models.ar_model', 'ARModel'),
    'rw': ('models.random_walk_model', 'RandomWalkModel'),
    'hvz': ('models.hvz_model', 'HVZModel'),
    'ri': ('models.ri_model', 'RIModel'),
    'rf': ('models.rf_model', 'RFModel'),
    'gbf': ('models.gbf_model', 'GBFModel'),
    'xgb': ('models.xgb_model', 'XGBModel'),
    'lightgbm': ('models.lightgbm_model', 'LightGBMModel'),
    'catboost': ('models.catboost_model', 'CatBoostModel'),
    'ann': ('models.ann_model', 'ANNModel'),
}


def get_model_class(model_name):
    """Dynamically import and return model class"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    module_path, class_name = MODEL_REGISTRY[model_name]

    # Dynamic import
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)

    return model_class


def run_single_model(model_name, data_path, results_dir):
    """Run a single model"""
    print(f"\n{'='*60}")
    print(f"Running {model_name.upper()} Model")
    print(f"{'='*60}")

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    print(f"Data loaded: {len(df)} rows")

    # Get model class and instantiate
    ModelClass = get_model_class(model_name)
    model = ModelClass()

    # Run forecast
    print(f"Running {model_name} forecast...")
    df_forecasts = model.run_forecast(df)

    # Save results
    output_file = os.path.join(results_dir, f"{model_name}_forecasts.csv")
    df_forecasts.to_csv(output_file, index=False)

    print(f"✓ {model_name.upper()} complete. Results saved to {output_file}")

    return output_file


def run_all_models(data_path, results_dir):
    """Run all models sequentially"""
    print(f"\n{'='*60}")
    print(f"Running ALL Models")
    print(f"{'='*60}\n")

    results = {}

    for model_name in MODEL_REGISTRY.keys():
        try:
            output_file = run_single_model(model_name, data_path, results_dir)
            results[model_name] = output_file
        except Exception as e:
            print(f"✗ {model_name.upper()} failed: {str(e)}")
            results[model_name] = None

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    successful = sum(1 for v in results.values() if v is not None)
    total = len(results)

    print(f"Completed: {successful}/{total} models")
    print("\nSuccessful:")
    for model, path in results.items():
        if path:
            print(f"  ✓ {model}")

    failed = [m for m, p in results.items() if p is None]
    if failed:
        print("\nFailed:")
        for model in failed:
            print(f"  ✗ {model}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run NTRS earnings prediction models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_model.py --model xgb
  python run_model.py --model rf --results-dir custom_results
  python run_model.py --all
  python run_model.py --model lasso --data data/custom_data.csv

Available models:
  """ + ", ".join(sorted(MODEL_REGISTRY.keys()))
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=list(MODEL_REGISTRY.keys()),
        help='Model to run'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all models'
    )

    parser.add_argument(
        '--data',
        type=str,
        default='data/full_fundamental_dataset.csv',
        help='Path to data file (default: data/full_fundamental_dataset.csv)'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Results output directory (default: results)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.model and not args.all:
        parser.error("Must specify either --model or --all")

    if args.model and args.all:
        parser.error("Cannot specify both --model and --all")

    # Run
    if args.all:
        run_all_models(args.data, args.results_dir)
    else:
        run_single_model(args.model, args.data, args.results_dir)


if __name__ == "__main__":
    main()
