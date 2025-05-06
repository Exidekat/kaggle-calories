#!/usr/bin/env python
"""
Ensemble regressors via simple averaging of their predictions.

This baseline ensemble leverages model diversity:
  - Combines regularized linear models (Lasso, Ridge) and boosting models (LightGBM, XGBoost, CatBoost).
  - Unweighted average often reduces variance and improves overall accuracy.

Script workflow:
  1. Load test features from data/test_transformed.csv.
  2. Load specified model pipelines from models/{model}_model.pkl.
  3. Predict with each model, handle infinite/missing values.
  4. Compute average prediction, clip negatives to zero.
  5. Write data/submission_votingregressor.csv with columns (id, Calories).

Usage:
  python ensemble_votingregressor.py [lasso ridge lightgbm xgboost catboost]
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib


def main():
    parser = argparse.ArgumentParser(
        description='Ensemble regressors via simple averaging.'
    )
    parser.add_argument(
        'models', nargs='*',
        help='Optional list of model names to ensemble (subset of: lasso, ridge, lightgbm, xgboost, catboost).'
    )
    parser.add_argument(
        '--test', type=str,
        default=os.path.join('data', 'test_fe_rev2.csv'),
        help='Path to engineered test data CSV'
    )
    args = parser.parse_args()

    valid_models = ['lasso', 'ridge', 'lightgbm', 'xgboost', 'catboost']
    if args.models:
        invalid = set(args.models) - set(valid_models)
        if invalid:
            sys.exit(f"Error: Unknown model name(s): {', '.join(invalid)}")
        model_names = args.models
    else:
        model_names = valid_models

    # Load test data
    test_csv = args.test
    if not os.path.exists(test_csv):
        sys.exit(f"Error: Test data not found at '{test_csv}'")
    df = pd.read_csv(test_csv)
    # Encode 'Sex' if present
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    # Preserve id and prepare features
    if 'id' not in df.columns:
        sys.exit("Error: 'id' column not found in test data")
    ids = df['id']
    X = df.drop(columns=['id'])
    # Clean infinite and missing values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())

    # Collect predictions
    preds = []
    for name in model_names:
        model_path = os.path.join('models', f'{name}_model.pkl')
        if not os.path.exists(model_path):
            sys.exit(f"Error: Model file not found at '{model_path}'")
        model = joblib.load(model_path)
        pred = model.predict(X)
        preds.append(pred)

    # Average predictions and enforce non-negative
    preds = np.column_stack(preds)
    avg_pred = preds.mean(axis=1)
    avg_pred = np.clip(avg_pred, 0, None)

    # Build submission
    submission = pd.DataFrame({'id': ids, 'Calories': avg_pred})
    out_path = os.path.join('data', 'submission_votingregressor.csv')
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


if __name__ == '__main__':
    main()