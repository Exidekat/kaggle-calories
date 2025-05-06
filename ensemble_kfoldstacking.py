#!/usr/bin/env python
"""
Ensemble regressors via 5-Fold stacking with a Ridge meta-learner.

Stacking approach builds a two-layer model:
  1. Out-of-fold (OOF) predictions from base models on training data (ensures honest labels).
  2. A Ridge meta-learner trained on these OOF predictions to learn optimal weights/combinations.

Script workflow:
  1. Load transformed training features from data/train_transformed.csv.
  2. Load base model pipelines from models/{model}_model.pkl.
  3. Generate OOF predictions using KFold (n_splits=5), fit each base model on training folds.
  4. Train Ridge on the OOF prediction matrix vs. true Calories.
  5. Load and preprocess test features, get base model predictions, then apply the trained meta-learner.
  6. Clip negatives, write data/submission_kfoldstacking.csv.

This method reduces overfitting in the meta-layer by using honest OOF predictions.

Usage:
  python ensemble_kfoldstacking.py [lasso ridge lightgbm xgboost catboost]
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.base import clone


def main():
    parser = argparse.ArgumentParser(
        description='Ensemble regressors via K-Fold stacking.'
    )
    parser.add_argument(
        'models', nargs='*',
        help='Optional list of model names to stack (subset of: lasso, ridge, lightgbm, xgboost, catboost).'
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

    # Load base model pipelines
    base_pipelines = {}
    for name in model_names:
        path = os.path.join('models', f'{name}_model.pkl')
        if not os.path.exists(path):
            sys.exit(f"Error: Model file not found at '{path}'")
        base_pipelines[name] = joblib.load(path)

    # Prepare out-of-fold predictions
    n_samples = X.shape[0]
    oof_preds = {name: np.zeros(n_samples) for name in model_names}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold+1}/5")
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        for name in model_names:
            # Clone pipeline and fit on fold
            model = clone(base_pipelines[name])
            model.fit(X_tr, y_tr)
            # Predict on validation fold
            oof_preds[name][val_idx] = model.predict(X_val)

    # Train meta-learner on out-of-fold predictions
    meta_X = pd.DataFrame(oof_preds)
    meta_y = y.values
    meta_model = Ridge()
    meta_model.fit(meta_X, meta_y)
    # Optionally save meta-model
    os.makedirs('models', exist_ok=True)
    joblib.dump(meta_model, 'models/stacking_meta_model.pkl')
    print("Trained meta-learner and saved to models/stacking_meta_model.pkl")

    # Load and prepare test data
    test_csv = args.test
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
    out_path = os.path.join('data', 'submission_kfoldstacking.csv')
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


if __name__ == '__main__':
    main()