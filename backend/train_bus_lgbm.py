#!/usr/bin/env python3
"""Train a stronger Bus fare model using LightGBM with frequency encoding.

This script:
- Loads `indian_bus_fare_dataset.csv` (can sample to limit memory)
- Builds features: Source, Destination, Bus_Type, Total_Seats, Duration_hours, Estimated Distance_km
- Frequency-encodes high-cardinality categorical features (map rare -> 'OTHER') and label-encodes them
- Trains an LGBMRegressor and injects the trained pipeline into `transport_model_pipeline.pkl`
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from bus_pipeline import FrequencyEncoder, WrapperPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import lightgbm as lgb
except Exception:
    lgb = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUS_CSV = os.path.join(BASE_DIR, 'indian_bus_fare_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'transport_model_pipeline.pkl')


# FrequencyEncoder and WrapperPipeline are implemented in `bus_pipeline.py` and imported above


def prepare(df):
    # Identify target
    target_col = None
    for col in df.columns:
        if 'price' in col.lower() or 'fare' in col.lower() or 'cost' in col.lower():
            target_col = col
            break
    if target_col is None:
        raise RuntimeError('Target column not found in bus CSV')

    # Ensure columns exist and rename for consistency
    cols_map = {}
    if 'Duration (hours)' in df.columns:
        df = df.rename(columns={'Duration (hours)': 'Duration_hours'})
    if 'Total Seats' in df.columns:
        df = df.rename(columns={'Total Seats': 'Total_Seats'})
    if 'Bus Type' in df.columns:
        df = df.rename(columns={'Bus Type': 'Bus_Type'})

    # Create estimated distance (approx) if not present
    if 'Distance_km' not in df.columns:
        df['Distance_km'] = pd.to_numeric(df.get('Duration_hours', 0).astype(float)) * 60.0

    # Keep relevant columns
    features = ['Source', 'Destination', 'Bus_Type', 'Total_Seats', 'Duration_hours', 'Distance_km']
    for f in features:
        if f not in df.columns:
            df[f] = 0

    df = df[features + [target_col]].copy()
    df = df.dropna(subset=[target_col])
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])

    return df, target_col, features


def train_and_save(sample_size=300000):
    if lgb is None:
        print('lightgbm not installed. Install with `pip install lightgbm` and re-run.')
        return

    print('Loading bus data...')
    df = pd.read_csv(BUS_CSV)
    print('Raw rows:', len(df))
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        print('Sampled rows:', len(df))

    df_prep, target_col, features = prepare(df)
    X = df_prep[features]
    y = df_prep[target_col]

    # Fill numeric NA
    numeric_feats = ['Total_Seats', 'Duration_hours', 'Distance_km']
    for n in numeric_feats:
        X[n] = pd.to_numeric(X[n], errors='coerce').fillna(0.0)

    cat_feats = ['Source', 'Destination', 'Bus_Type']
    # Encoder
    enc = FrequencyEncoder(cols=cat_feats, top_k=300)
    enc.fit(X)
    X_enc = enc.transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)

    # LightGBM regressor
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=64,
        n_jobs=-1,
        random_state=42
    )

    print('Training LightGBM model...')
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='l1',
            early_stopping_rounds=50,
            verbose=50
        )
    except TypeError:
        # Older/newer sklearn-lightgbm wrappers may not accept those args; fall back
        print('Warning: fit() did not accept early_stopping or verbose args, falling back to basic fit()')
        model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f'Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}')

    # Build a pipeline object that applies enc then model
    pipe = WrapperPipeline(enc, model, features, numeric_feats)

    # Inject into hybrid model file (preserve other keys if present)
    hybrid = {}
    if os.path.exists(MODEL_PATH):
        try:
            hybrid = joblib.load(MODEL_PATH)
            if not isinstance(hybrid, dict):
                hybrid = {'type': 'hybrid'}
        except Exception:
            hybrid = {'type': 'hybrid'}
    hybrid['type'] = 'hybrid'
    hybrid['bus_pipeline'] = pipe
    # keep any existing lookup or flight_pipeline if present
    joblib.dump(hybrid, MODEL_PATH)
    print('Saved hybrid model with LightGBM bus_pipeline to', MODEL_PATH)


if __name__ == '__main__':
    train_and_save(sample_size=200000)
