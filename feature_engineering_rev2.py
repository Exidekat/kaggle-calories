"""
feature_engineering_rev2.py

Standalone script to perform advanced feature engineering for the Playground S5E5 calories dataset.
Generates base transforms (sex encoding, relative intensity), manual high-correlation interaction,
and automated polynomial interaction features on workout metrics, then selects by correlation.
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures



def calculate_lbm(row):
    """Calculate lean body mass (LBM) using sex-specific formula."""
    weight = row['Weight']
    height = row['Height']
    sex = row['Sex']
    if sex == 0:
        # male
        return 0.32810 * weight + 0.33929 * height - 29.5336
    else:
        # female
        return 0.29569 * weight + 0.41813 * height - 43.2933


def calculate_bmr(row):
    """Calculate basal metabolic rate (BMR) using Harris-Benedict equations."""
    weight = row['Weight']
    height = row['Height']
    age = row['Age']
    sex = row['Sex']
    if sex == 0:
        # male
        return 13.397 * weight + 4.799 * height - 5.677 * age + 88.362
    else:
        # female
        return 9.247 * weight + 3.098 * height - 4.33 * age + 447.593


def create_base_features(df):
    """Create base features: encode sex, compute relative intensity and classical anthropometrics."""
    df = df.copy()
    # Encode sex: male=0, female=1
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    # Relative exercise intensity (% of max HR)
    df['RelativeIntensity'] = df['Heart_Rate'] / (220 - df['Age']) * 100
    # Anthropometric estimates (optional)
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['BSA'] = 0.007184 * (df['Weight'] ** 0.425) * (df['Height'] ** 0.725)
    df['LBM'] = df.apply(calculate_lbm, axis=1)
    df['BMR'] = df.apply(calculate_bmr, axis=1)
    # Manual high-correlation feature discovered via prior analysis
    df['Feat_Dur_HR2_RelInt'] = (
        df['Duration'] * (df['Heart_Rate'] ** 2) * df['RelativeIntensity']
    )
    return df


def generate_poly_interactions(df, features, degree=3, interaction_only=True):
    """
    Generate interaction-only polynomial features for a given subset of columns.
    Returns a DataFrame of new features (excluding originals).
    """
    poly = PolynomialFeatures(
        degree=degree,
        interaction_only=interaction_only,
        include_bias=False
    )
    arr = poly.fit_transform(df[features])
    # compatibility for sklearn <1.0
    try:
        names = poly.get_feature_names_out(features)
    except AttributeError:
        names = poly.get_feature_names(features)
    df_poly = pd.DataFrame(arr, columns=names, index=df.index)
    # drop original base columns, keep only true interactions
    for f in features:
        if f in df_poly.columns:
            df_poly.drop(columns=f, inplace=True)
    return df_poly


def select_by_correlation(df_base, df_candidates, target_col, threshold=0.3):
    """
    Select candidate features whose absolute Pearson correlation with target exceeds threshold.
    Returns list of selected feature names.
    """
    corrs = df_candidates.corrwith(df_base[target_col]).abs()
    selected = corrs[corrs >= threshold].sort_values(ascending=False)
    return selected.index.tolist(), corrs.loc[selected.index]


def feature_engineering(train_df, test_df, target_col='Calories', corr_threshold=0.3):
    """
    Orchestrate full pipeline: base features, poly interactions, correlation-based selection.
    Returns (train_feats, test_feats, selected_features, correlations).
    """
    # Base transforms
    train = create_base_features(train_df.drop(columns=['id'])).copy()
    test_ids = test_df['id'] if 'id' in test_df else None
    test = create_base_features(test_df.drop(columns=['id'], errors='ignore')).copy()

    # Define subset for advanced interactions (workout metrics)
    workout_feats = ['Duration', 'Heart_Rate', 'Body_Temp', 'RelativeIntensity']
    # Generate polynomial interactions up to cubic
    poly_train = generate_poly_interactions(train, workout_feats, degree=3)
    poly_test = generate_poly_interactions(test, workout_feats, degree=3)

    # Select only the most predictive of those interactions
    selected, corrs = select_by_correlation(
        pd.concat([train, poly_train], axis=1),
        poly_train,
        target_col,
        threshold=corr_threshold
    )
    print(f"Selected {len(selected)} interaction features with |corr|>={corr_threshold}:")
    for feat in selected:
        print(f"  {feat}: corr={corrs[feat]:.4f}")

    # Assemble final feature sets
    train_feats = pd.concat([train, poly_train[selected]], axis=1)
    test_feats = pd.concat([test, poly_test[selected]], axis=1)
    if test_ids is not None:
        test_feats.insert(0, 'id', test_ids)

    return train_feats, test_feats, selected, corrs


def main():
    parser = argparse.ArgumentParser(
        description="Feature engineering rev2 for calories dataset"
    )
    parser.add_argument(
        '--train', type=str, required=True,
        help="Path to train.csv (with id, features, Calories)"
    )
    parser.add_argument(
        '--test', type=str, required=True,
        help="Path to test.csv (with id and features)"
    )
    parser.add_argument(
        '--corr_threshold', type=float, default=0.3,
        help="Absolute correlation threshold for feature selection"
    )
    parser.add_argument(
        '--out_train', type=str, default='train_fe_rev2.csv',
        help="Output path for engineered train features"
    )
    parser.add_argument(
        '--out_test', type=str, default='test_fe_rev2.csv',
        help="Output path for engineered test features"
    )
    args = parser.parse_args()

    # Load data
    df_train = pd.read_csv(args.train)
    df_test = pd.read_csv(args.test)

    # Run feature engineering
    train_feats, test_feats, selected, corrs = feature_engineering(
        df_train, df_test, target_col='Calories',
        corr_threshold=args.corr_threshold
    )

    # Save outputs
    train_feats.to_csv(args.out_train, index=False)
    test_feats.to_csv(args.out_test, index=False)
    print(f"Saved engineered train to {args.out_train}, test to {args.out_test}")


if __name__ == '__main__':
    main()