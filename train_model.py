#!/usr/bin/env python
"""
Train and tune multiple regression models on engineered features.

This script reflects an iterative engineering process:
  1. Started with simple LinearRegression on raw data.
  2. Added feature engineering (log, exp, square, standardization, binning) to capture non-linear relationships.
  3. Switched to Optuna-based hyperparameter tuning for multiple models:
     Lasso, Ridge (linear), LightGBM, XGBoost, CatBoost (tree-based).
  4. Introduced 5-fold KFold cross-validation for robust metric estimates.
  5. Wrapped all models in StandardScaler pipelines to ensure numeric stability and avoid overflow/inf issues.
  6. Cleaned infinite and missing feature values after engineering transforms.
  7. Enabled CLI filtering of which models to tune and train.
  8. Allows adjustable Optuna trial counts for scalable tuning.

Final models are retrained on the full transformed dataset with optimal hyperparameters
and saved under models/{model}_model.pkl.
"""
import os
import warnings
import pandas as pd
import numpy as np
import joblib
import optuna
import xgboost as xgb
from sklearn.linear_model import Lasso, Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingRegressor

warnings.filterwarnings('ignore')
import argparse
import sys


def tune_model(name, model_cls, param_fn, X, y, cv, n_trials=20):
    """
    Tune hyperparameters for a given model using Optuna.
    Returns best_params and best_rmse (CV score to minimize).
    """
    def objective(trial):
        # Suggest hyperparameters
        params = param_fn(trial)
        # Fixed parameters per model
        if name == 'lightgbm':
            params.update({'random_state': 42})
        elif name == 'xgboost':
            # Prepare for XGBoost cv
            params.update({
                'objective': 'reg:squarederror',
                'seed': 42,
                'nthread': -1,
            })
            # Use XGBoost's built-in CV for speed and early stopping
            num_round = params.pop('n_estimators')
            dtrain = xgb.DMatrix(X, label=y)
            cv_res = xgb.cv(
                params,
                dtrain,
                num_boost_round=num_round,
                nfold=cv.n_splits,
                metrics='rmse',
                early_stopping_rounds=10,
                seed=42,
                verbose_eval=False,
            )
            # Return best CV RMSE
            return float(cv_res['test-rmse-mean'].min())
        elif name == 'catboost':
            params.update({'verbose': 0, 'random_seed': 42})
        # For sklearn-based models, use pipeline CV
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('reg', model_cls(**params))
        ])
        scores = cross_val_score(
            pipeline, X, y, cv=cv,
            scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        return -scores.mean()

    study = optuna.create_study(direction='minimize')
    # Run optimization for specified number of trials
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def main():
    # Parse command-line arguments for which models to train
    parser = argparse.ArgumentParser(
        description='Train and tune regression models on engineered features.'
    )
    parser.add_argument(
        'models', nargs='*',
        help='Optional list of model names to train (subset of: lasso, ridge, lightgbm, xgboost, catboost).'
    )
    parser.add_argument(
        '--trials', '-t', type=int, default=20,
        help='Number of Optuna trials per model (default: 20)'
    )
    parser.add_argument(
        '--train', type=str,
        default=os.path.join('data', 'train_fe_rev2.csv'),
        help='Path to engineered training data CSV'
    )
    parser.add_argument(
        '--folds', '-f', type=int, default=5,
        help='Number of CV folds (default: 5). Lower for faster tuning.'
    )
    parser.add_argument(
        '--ensemble', action='store_true',
        help='After training individual models, build a VotingRegressor ensemble and report its CV RMSE.'
    )
    args = parser.parse_args()
    # Ensure output directory exists
    os.makedirs('models', exist_ok=True)
    # Load transformed data
    df = pd.read_csv(args.train)

    # Encode categorical variable Sex if present
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

    # Define target and features
    if 'Calories' not in df.columns:
        raise KeyError("Target column 'Calories' not found in data")
    y = df['Calories'].astype(float)
    drop_cols = ['Calories']
    if 'id' in df.columns:
        drop_cols.append('id')
    # Prepare feature matrix
    X = df.drop(columns=drop_cols)
    # Replace infinite values from feature engineering (e.g. exp overflows)
    X = X.replace([np.inf, -np.inf], np.nan)
    # Drop any columns that are entirely NaN (all values were invalid)
    all_nan_cols = X.columns[X.isna().all()]
    if len(all_nan_cols) > 0:
        warnings.warn(f"Dropping columns with all missing/infinite values: {list(all_nan_cols)}")
        X = X.drop(columns=all_nan_cols)
    # Fill remaining missing values with column means
    if X.isnull().any().any():
        X = X.fillna(X.mean())

    # Cross-validation splitter
    # K-fold cross-validation with shuffling
    cv = KFold(n_splits=args.folds, shuffle=True, random_state=42)

    # Mapping of model classes and parameter spaces
    model_classes = {
        'lasso': Lasso,
        'ridge': Ridge,
        'lightgbm': LGBMRegressor,
        'xgboost': XGBRegressor,
        'catboost': CatBoostRegressor,
    }
    param_fns = {
        # Increase max_iter and set tolerance to ensure convergence
        'lasso': lambda tr: {
            'alpha': tr.suggest_loguniform('alpha', 1e-4, 1e2),
            'max_iter': 10000,
            'tol': 1e-4,
        },
        'ridge': lambda tr: {'alpha': tr.suggest_loguniform('alpha', 1e-4, 1e2)},
        'lightgbm': lambda tr: {
            'num_leaves': tr.suggest_int('num_leaves', 16, 128),
            'max_depth': tr.suggest_int('max_depth', 3, 12),
            'learning_rate': tr.suggest_loguniform('learning_rate', 1e-3, 0.3),
            'n_estimators': tr.suggest_int('n_estimators', 50, 500),
        },
        'xgboost': lambda tr: {
            'max_depth': tr.suggest_int('max_depth', 3, 12),
            'learning_rate': tr.suggest_loguniform('learning_rate', 1e-3, 0.3),
            'n_estimators': tr.suggest_int('n_estimators', 50, 500),
            'subsample': tr.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': tr.suggest_uniform('colsample_bytree', 0.5, 1.0),
            # Regularization and complexity control
            'gamma': tr.suggest_loguniform('gamma', 1e-8, 10.0),
            'min_child_weight': tr.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': tr.suggest_loguniform('reg_alpha', 1e-8, 10.0),
            'reg_lambda': tr.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        },
        'catboost': lambda tr: {
            'depth': tr.suggest_int('depth', 4, 10),
            'learning_rate': tr.suggest_loguniform('learning_rate', 1e-3, 0.3),
            'iterations': tr.suggest_int('iterations', 100, 1000),
            'l2_leaf_reg': tr.suggest_loguniform('l2_leaf_reg', 1e-2, 10),
        },
    }
    # Filter selected models if provided via CLI
    if args.models:
        invalid = set(args.models) - set(model_classes.keys())
        if invalid:
            sys.exit(f"Error: Unknown model name(s): {', '.join(invalid)}")
        model_classes = {k: v for k, v in model_classes.items() if k in args.models}
        param_fns = {k: v for k, v in param_fns.items() if k in args.models}

    # Tune, evaluate, and save each model
    fitted_models = []
    for name, cls in model_classes.items():
        print(f"Tuning {name} ({args.trials} trials)...")
        best_params, best_rmse = tune_model(
            name, cls, param_fns[name], X, y, cv, n_trials=args.trials
        )
        print(f"{name} best RMSE (CV): {best_rmse:.4f}")
        print(f"{name} best params: {best_params}")

        # Instantiate final model with best params: wrap in imputation+scaling pipeline
        params = best_params.copy()
        if name == 'lightgbm':
            params.update({'random_state': 42})
        elif name == 'xgboost':
            params.update({
                'objective': 'reg:squarederror',
                'random_state': 42,
                'n_jobs': -1,
            })
        elif name == 'catboost':
            params.update({'verbose': 0, 'random_seed': 42})
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('reg', cls(**params))
        ])

        # Cross-validate final model for RMSE and R2
        results = cross_validate(
            model, X, y, cv=cv,
            scoring=('neg_root_mean_squared_error', 'r2'), n_jobs=-1
        )
        rmse_mean = -results['test_neg_root_mean_squared_error'].mean()
        r2_mean = results['test_r2'].mean()
        print(f"{name} final CV RMSE: {rmse_mean:.4f}, R2: {r2_mean:.4f}")

        # Fit on full data
        model.fit(X, y)
        # Save with RMSE in filename
        model_path = f'models/{name}_{rmse_mean:.4f}_model.pkl'
        joblib.dump(model, model_path)
        # Also save standard path for ensemble loading
        simple_path = f'models/{name}_model.pkl'
        joblib.dump(model, simple_path)
        print(f"Saved tuned model to {model_path} and {simple_path}\n")
        # Keep for ensemble
        fitted_models.append((name, model))

    # If requested, evaluate and save a VotingRegressor ensemble of the trained models
    if args.ensemble and len(fitted_models) > 1:
        print("Evaluating VotingRegressor ensemble...")
        ensemble = VotingRegressor(estimators=fitted_models)
        # CV for ensemble
        results_ens = cross_validate(
            ensemble, X, y, cv=cv,
            scoring=('neg_root_mean_squared_error', 'r2'), n_jobs=-1
        )
        rmse_ens = -results_ens['test_neg_root_mean_squared_error'].mean()
        r2_ens = results_ens['test_r2'].mean()
        print(f"Ensemble CV RMSE: {rmse_ens:.4f}, R2: {r2_ens:.4f}")
        # Fit on full data and save ensemble
        ensemble.fit(X, y)
        ens_path = 'models/ensemble_model.pkl'
        joblib.dump(ensemble, ens_path)
        print(f"Saved ensemble model to {ens_path}\n")
    # All models (and ensemble) have been tuned, cross-validated, and saved.


if __name__ == '__main__':
    main()