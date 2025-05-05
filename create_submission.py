#!/usr/bin/env python
"""
Generate a submission file using a trained model.
Usage: create_submission.py <model_name> <model_path>
  - model_name: one of {lasso, ridge, lightgbm, xgboost, catboost}
  - model_path: path to the pickled model (.pkl)
Loads data/test_transformed.csv, runs inference, and saves to data/submission.csv
with columns matching data/sample_submission.csv (id,Calories).
"""
import os
import sys
import argparse
import pandas as pd
import joblib


def main():
    parser = argparse.ArgumentParser(
        description='Create submission CSV from a trained regression model.'
    )
    parser.add_argument('model_name', type=str,
                        help='Model name (e.g. lasso, ridge, lightgbm, xgboost, catboost)')
    parser.add_argument('model_path', type=str,
                        help='File path to the pickled model (.pkl)')
    args = parser.parse_args()

    # Validate model name (optional)
    valid_models = {'lasso', 'ridge', 'lightgbm', 'xgboost', 'catboost'}
    if args.model_name not in valid_models:
        print(f"Warning: model_name '{args.model_name}' not in expected {valid_models}")

    # Load test data
    test_csv = os.path.join('data', 'test_transformed.csv')
    if not os.path.exists(test_csv):
        sys.exit(f"Error: Test data not found at '{test_csv}'")
    df = pd.read_csv(test_csv)

    # Encode categorical 'Sex' if present
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    # Preserve IDs for submission
    if 'id' not in df.columns:
        sys.exit("Error: 'id' column not found in test data")
    ids = df['id']

    # Prepare features by dropping 'id'
    X = df.drop(columns=['id'])

    # Load model
    if not os.path.exists(args.model_path):
        sys.exit(f"Error: Model file not found at '{args.model_path}'")
    model = joblib.load(args.model_path)

    # Predict
    preds = model.predict(X)

    # Build submission DataFrame
    submission = pd.DataFrame({'id': ids, 'Calories': preds})
    submission = submission[['id', 'Calories']]

    # Write to CSV
    out_path = os.path.join('data', 'submission.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


if __name__ == '__main__':
    main()