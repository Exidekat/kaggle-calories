#!/usr/bin/env python
"""
Perform feature engineering transforms to capture non-linear patterns and improve model stability.

Transforms applied for each numeric feature:
  - log1p ({col}_log) to compress large values.
  - exp ({col}_exp) to expand small differences (may yield very large values).
  - square ({col}_square) to model quadratic effects.
  - standardized (z-score) ({col}_std) to normalize distributions.
Binning applied to:
  - Age and Duration via quartiles (falling back to equal-width if needed) as {col}_bin.

Engineering rationale:
  - Log/square capture non-linear relationships.
  - Standardization ensures numeric stability for solvers and tree algorithms.
  - Bins allow models to learn threshold effects (e.g., age groups).
  - The combination of transforms supplies diverse features for linear and boosting models.

Outputs:
  - data/train_transformed.csv
  - data/test_transformed.csv
"""
import os
import pandas as pd
import numpy as np


def transform(df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
    """
    Add log and exp transformations for numeric features in df,
    excluding any columns in exclude_cols.
    """
    if exclude_cols is None:
        exclude_cols = []
    # Identify numeric feature columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude identifier/target columns
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    # Fill missing values in numeric columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    # Apply log1p and exp transforms (cap exp to float32 max to avoid overflow)
    max_float32 = np.finfo(np.float32).max
    for col in numeric_cols:
        df[f"{col}_log"] = np.log1p(df[col])
        # exponential transform
        exp_col = f"{col}_exp"
        df[exp_col] = np.exp(df[col])
        # cap to maximum representable float32 to prevent infinite or too-large values
        df[exp_col] = df[exp_col].clip(upper=max_float32)

    # Add squared and standardized features
    for col in numeric_cols:
        # squared term
        df[f"{col}_square"] = df[col] ** 2
        # standardized term
        mean = df[col].mean()
        std = df[col].std()
        if std != 0:
            df[f"{col}_std"] = (df[col] - mean) / std
        else:
            df[f"{col}_std"] = 0.0

    # Binning for selected features: Age and Duration
    for bin_col in ['Age', 'Duration']:
        if bin_col in df.columns:
            try:
                df[f"{bin_col}_bin"] = pd.qcut(df[bin_col], q=4, labels=False, duplicates='drop')
            except Exception:
                # fallback to simple cut
                df[f"{bin_col}_bin"] = pd.cut(df[bin_col], bins=4, labels=False)
    return df


def main():
    # Ensure output directory exists
    os.makedirs('data', exist_ok=True)

    # Process training data
    train_path = 'data/train.csv'
    train = pd.read_csv(train_path)
    train_trans = transform(train.copy(), exclude_cols=['id', 'Calories'])
    out_train = 'data/train_transformed.csv'
    train_trans.to_csv(out_train, index=False)
    print(f"Saved transformed train data to {out_train}")

    # Process test data
    test_path = 'data/test.csv'
    test = pd.read_csv(test_path)
    test_trans = transform(test.copy(), exclude_cols=['id'])
    out_test = 'data/test_transformed.csv'
    test_trans.to_csv(out_test, index=False)
    print(f"Saved transformed test data to {out_test}")


if __name__ == '__main__':
    main()