#!/usr/bin/env python3
"""Train a lightweight Bus fare model to avoid memory blowups.

This script:
- Loads a sample of `indian_bus_fare_dataset.csv` (default 100k rows)
- Keeps only the most relevant features to limit one-hot cardinality
- Uses OneHotEncoder (sparse) + TruncatedSVD to reduce dimensionality
- Trains a smaller RandomForestRegressor and injects it into the
  existing `transport_model_pipeline.pkl` as the `bus_pipeline` key.
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUS_CSV = os.path.join(BASE_DIR, 'indian_bus_fare_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'transport_model_pipeline.pkl')

def load_sample(path, n_samples=100000, random_state=42):
    df = pd.read_csv(path)
    if len(df) > n_samples:
        return df.sample(n_samples, random_state=random_state).reset_index(drop=True)
    return df

def build_and_inject():
    print('Loading sample bus data...')
    df = load_sample(BUS_CSV, n_samples=100000)

    # Select a small set of features to limit cardinality
    keep_cols = []
    for c in df.columns:
        lc = c.lower()
        if 'fare' in lc or 'price' in lc or 'cost' in lc:
            target_col = c
        if lc in ('source', 'destination', 'bus type', 'duration (hours)', 'total seats'):
            keep_cols.append(c)

    print('Using features:', keep_cols)
    X = df[keep_cols].copy()
    y = pd.to_numeric(df[target_col], errors='coerce')
    mask = ~y.isna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    # Normalize column names to short names that main.py expects
    # Map: 'Duration (hours)' -> 'Duration_Hours' etc.
    rename_map = {}
    for c in X.columns:
        nc = c
        if c.lower() == 'duration (hours)':
            nc = 'Duration_hours'
        if c.lower() == 'total seats':
            nc = 'Total_Seats'
        if c.lower() == 'bus type':
            nc = 'Bus_Type'
        if c.lower() == 'source':
            nc = 'Start_Location'
        if c.lower() == 'destination':
            nc = 'Destination'
        rename_map[c] = nc
    X = X.rename(columns=rename_map)

    numeric_features = [c for c in X.columns if X[c].dtype.kind in 'biufc']
    categorical_features = [c for c in X.columns if c not in numeric_features]

    print('Numeric features:', numeric_features)
    print('Categorical features:', categorical_features)

    # Fill na
    X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
    X[categorical_features] = X[categorical_features].fillna('UNKNOWN')

    # Preprocessor: OneHotEncoder (sparse) for categorical -> TruncatedSVD to reduce dims
    # Create OneHotEncoder compatible with installed scikit-learn
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    except TypeError:
        # Older scikit-learn used 'sparse' param name
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)

    # Reduce to a modest number of components to avoid memory and SVD issues
    svd_components = 12
    cat_pipeline = Pipeline([
        ('ohe', ohe),
        ('svd', TruncatedSVD(n_components=svd_components, random_state=42))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features) if numeric_features else ('num', 'passthrough', []),
            ('cat', cat_pipeline, categorical_features) if categorical_features else ('cat', 'passthrough', [])
        ],
        remainder='drop'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Training lightweight bus model on', X_train.shape[0], 'samples...')
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print('Evaluation on test split:')
    print(f'  MAE: {mae:.2f}')
    print(f'  RMSE: {rmse:.2f}')
    print(f'  R2: {r2:.4f}')

    # Inject into hybrid model file
    if os.path.exists(MODEL_PATH):
        try:
            hybrid = joblib.load(MODEL_PATH)
            if not isinstance(hybrid, dict):
                print('Existing model not dict; creating new hybrid dict')
                hybrid = {'type': 'hybrid'}
        except Exception:
            hybrid = {'type': 'hybrid'}
    else:
        hybrid = {'type': 'hybrid'}

    hybrid['bus_pipeline'] = pipeline
    # keep existing flight_pipeline and lookup if present
    joblib.dump(hybrid, MODEL_PATH)
    print('Injected lightweight bus_pipeline into', MODEL_PATH)

if __name__ == '__main__':
    build_and_inject()
