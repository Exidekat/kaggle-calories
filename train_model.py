#!/usr/bin/env python
"""
Train and tune multiple regression models on engineered features:
  - L1 (Lasso), L2 (Ridge), LightGBM, XGBoost, and CatBoost.
Uses Optuna for hyperparameter optimization and ShuffleSplit cross-validation.
Loads data from data/train_transformed.csv and saves models to models/.
"""
import os
import warnings
import pandas as pd
import numpy as np
import joblib
import optuna
from sklearn.linear_model import Lasso, Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import ShuffleSplit, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


def tune_model(name, model_cls, param_fn, X, y, cv, n_trials=20):
    """
    Tune hyperparameters for a given model using Optuna.
    Returns best_params and best_rmse (CV score to minimize).
    """
    def objective(trial):
        params = param_fn(trial)
        # include fixed random seeds and objectives
        if name == 'lightgbm':
            params.update({'random_state': 42})
        elif name == 'xgboost':
            params.update({'objective': 'reg:squarederror', 'random_state': 42})
        elif name == 'catboost':
            params.update({'verbose': 0, 'random_seed': 42})
        # Wrap all models in a scaling pipeline to prevent overflow/inf issues
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('reg', model_cls(**params))
        ])
        scores = cross_val_score(
            pipeline, X, y, cv=cv,
            scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        rmse = -scores.mean()
        return rmse

    study = optuna.create_study(direction='minimize')
    # Run optimization for specified number of trials
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def main():
    # Ensure output directory exists
    os.makedirs('models', exist_ok=True)
    # Load transformed data
    df = pd.read_csv('data/train_transformed.csv')

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
    # Handle any remaining missing values
    if X.isnull().any().any():
        X = X.fillna(X.mean())

    # Cross-validation splitter
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

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
        },
        'catboost': lambda tr: {
            'depth': tr.suggest_int('depth', 4, 10),
            'learning_rate': tr.suggest_loguniform('learning_rate', 1e-3, 0.3),
            'iterations': tr.suggest_int('iterations', 100, 1000),
            'l2_leaf_reg': tr.suggest_loguniform('l2_leaf_reg', 1e-2, 10),
        },
    }

    # Tune, evaluate, and save each model
    for name, cls in model_classes.items():
        print(f"Tuning {name}...")
        best_params, best_rmse = tune_model(name, cls, param_fns[name], X, y, cv)
        print(f"{name} best RMSE (CV): {best_rmse:.4f}")
        print(f"{name} best params: {best_params}")

        # Instantiate final model with best params: wrap all in a scaling pipeline
        params = best_params.copy()
        if name == 'lightgbm':
            params.update({'random_state': 42})
        elif name == 'xgboost':
            params.update({'objective': 'reg:squarederror', 'random_state': 42})
        elif name == 'catboost':
            params.update({'verbose': 0, 'random_seed': 42})
        model = Pipeline([
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

        # Fit on full data and save
        model.fit(X, y)
        model_path = f'models/{name}_{rmse_mean:.4f}_model.pkl'
        joblib.dump(model, model_path)
        print(f"Saved tuned model to {model_path}\n")

    # All models have been tuned, cross-validated, and saved.


if __name__ == '__main__':
    main()