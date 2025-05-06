#!/usr/bin/env python
"""
Ensemble regressors via holdout blending with a Ridge meta-learner.

Blending approach steps:
  1. Split transformed training data into a blend set (e.g., 80%) and holdout set (20%).
  2. Train base models on the blend set only.
  3. Generate predictions on the holdout set to serve as meta-features.
  4. Train a Ridge meta-learner on these holdout predictions vs. true Calories.
  5. Apply base models to test data, then combine via the meta-learner.
  6. Clip negative predictions and write data/submission_holdout_blending.csv.

This simpler ensemble requires only a single holdout but may exhibit some holdout variance.

Usage:
  python ensemble_holdout_blending.py [lasso ridge lightgbm xgboost catboost]
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.base import clone


def main():
    parser = argparse.ArgumentParser(
        description='Ensemble regressors via holdout blending.'
    )
    parser.add_argument(
        'models', nargs='*',
        help='Optional list of model names to blend (subset of: lasso, ridge, lightgbm, xgboost, catboost).'
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

    # Load and prepare training data
    train_csv = os.path.join('data', 'train_transformed.csv')
    if not os.path.exists(train_csv):
        sys.exit(f"Error: Training data not found at '{train_csv}'")
    df = pd.read_csv(train_csv)
    # Encode 'Sex' if present
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
    # Define target and features
    if 'Calories' not in df.columns:
        sys.exit("Error: 'Calories' column not found in training data")
    y = df['Calories'].astype(float)
    drop_cols = ['Calories']
    if 'id' in df.columns:
        drop_cols.append('id')
    X = df.drop(columns=drop_cols)
    # Clean infinite and missing values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())

    # Split into blend (train) and holdout sets
    X_blend, X_hold, y_blend, y_hold = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load base model pipelines
    base_pipelines = {}
    for name in model_names:
        path = os.path.join('models', f'{name}_model.pkl')
        if not os.path.exists(path):
            sys.exit(f"Error: Model file not found at '{path}'")
        base_pipelines[name] = joblib.load(path)

    # Generate holdout predictions from base models
    holdout_preds = {}
    for name in model_names:
        # Clone pipeline and fit on blend set
        model = clone(base_pipelines[name])
        model.fit(X_blend, y_blend)
        # Predict on holdout set
        holdout_preds[name] = model.predict(X_hold)

    # Train meta-learner on holdout predictions
    meta_X_hold = pd.DataFrame(holdout_preds)
    meta_y_hold = y_hold.values
    meta_model = Ridge()
    meta_model.fit(meta_X_hold, meta_y_hold)
    # Optionally save meta-model
    os.makedirs('models', exist_ok=True)
    joblib.dump(meta_model, 'models/blending_meta_model.pkl')
    print("Trained meta-learner and saved to models/blending_meta_model.pkl")

    # Load and prepare test data
    test_csv = os.path.join('data', 'test_transformed.csv')
    if not os.path.exists(test_csv):
        sys.exit(f"Error: Test data not found at '{test_csv}'")
    df_test = pd.read_csv(test_csv)
    if 'Sex' in df_test.columns:
        df_test['Sex'] = df_test['Sex'].map({'male': 1, 'female': 0})
    if 'id' not in df_test.columns:
        sys.exit("Error: 'id' column not found in test data")
    ids = df_test['id']
    X_test = df_test.drop(columns=['id'])
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_test.mean())

    # Generate base model predictions on test set
    test_preds = {name: base_pipelines[name].predict(X_test) for name in model_names}
    meta_X_test = pd.DataFrame(test_preds)
    final_pred = meta_model.predict(meta_X_test)
    final_pred = np.clip(final_pred, 0, None)

    # Build submission
    submission = pd.DataFrame({'id': ids, 'Calories': final_pred})
    out_path = os.path.join('data', 'submission_holdout_blending.csv')
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


if __name__ == '__main__':
    main()