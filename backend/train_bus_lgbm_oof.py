#!/usr/bin/env python3
"""Train Bus model with out-of-fold target encoding and LightGBM.

Produces a pickled hybrid model `transport_model_pipeline.pkl` with a picklable
WrapperPipeline that contains TargetEncoder (from bus_pipeline.py) and LightGBM model.
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from bus_pipeline import TargetEncoder, WrapperPipeline

try:
    import lightgbm as lgb
except Exception:
    lgb = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUS_CSV = os.path.join(BASE_DIR, 'indian_bus_fare_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'transport_model_pipeline.pkl')


def prepare(df):
    # prepare same columns as earlier
    if 'Duration (hours)' in df.columns:
        df = df.rename(columns={'Duration (hours)': 'Duration_hours'})
    if 'Total Seats' in df.columns:
        df = df.rename(columns={'Total Seats': 'Total_Seats'})
    if 'Bus Type' in df.columns:
        df = df.rename(columns={'Bus Type': 'Bus_Type'})

    if 'Distance_km' not in df.columns:
        df['Distance_km'] = pd.to_numeric(df.get('Duration_hours', 0).astype(float)) * 60.0

    target_col = None
    for c in df.columns:
        if 'price' in c.lower() or 'fare' in c.lower() or 'cost' in c.lower():
            target_col = c
            break
    if target_col is None:
        raise RuntimeError('Target column not found')

    features = ['Source', 'Destination', 'Bus_Type', 'Total_Seats', 'Duration_hours', 'Distance_km']
    for f in features:
        if f not in df.columns:
            df[f] = 0

    df = df[features + [target_col]].copy()
    df = df.dropna(subset=[target_col])
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])
    return df, target_col, features


def oof_target_encoding(df, target_col, cat_cols, n_splits=5, m=10):
    """Compute out-of-fold target encoding for categorical cols.

    Returns a DataFrame with new columns '<col>_te' for each categorical column.
    """
    X = df.copy()
    y = X[target_col].values
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = pd.DataFrame(index=X.index)

    global_mean = y.mean()

    for col in cat_cols:
        oof_col = np.zeros(len(X), dtype=float)
        for train_idx, val_idx in kf.split(X):
            train_vals = X.iloc[train_idx]
            val_vals = X.iloc[val_idx]
            grp = train_vals.groupby(col)[target_col].agg(['sum', 'count'])
            enc = (grp['sum'] + m * global_mean) / (grp['count'] + m)
            val_series = val_vals[col].astype(str).map(enc).fillna(global_mean).astype(float)
            oof_col[val_idx] = val_series.values
        oof[col + '_te'] = oof_col
    return oof


def train(sample_size=200000):
    if lgb is None:
        print('lightgbm not installed. Install and re-run')
        return

    print('Loading bus CSV...')
    df_raw = pd.read_csv(BUS_CSV)
    print('Total rows:', len(df_raw))
    if sample_size and len(df_raw) > sample_size:
        df = df_raw.sample(sample_size, random_state=42).reset_index(drop=True)
        print('Sampled rows:', len(df))
    else:
        df = df_raw

    df_prep, target_col, features = prepare(df)

    cat_cols = ['Source', 'Destination', 'Bus_Type']
    numeric_cols = ['Total_Seats', 'Duration_hours', 'Distance_km']

    print('Computing OOF target encoding...')
    oof = oof_target_encoding(df_prep, target_col, cat_cols, n_splits=5, m=20)

    # Combine numeric + oof encoded
    X_oof = pd.concat([df_prep[numeric_cols].reset_index(drop=True), oof.reset_index(drop=True)], axis=1)
    y = df_prep[target_col].values

    # simple holdout for evaluation
    X_train, X_val, y_train, y_val = train_test_split(X_oof, y, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=128, n_jobs=-1, random_state=42)
    print('Training on OOF features...')
    try:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='l1', early_stopping_rounds=50, verbose=50)
    except TypeError:
        print('Fit did not accept early stopping args; falling back')
        model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    print(f'OOF features evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}')

    # Fit final TargetEncoder on full data
    te = TargetEncoder(cols=cat_cols, m=20)
    te.fit(df_prep[cat_cols], df_prep[target_col])

    # Transform full dataset and train final model
    X_full_enc = pd.concat([df_prep[numeric_cols].reset_index(drop=True), te.transform(df_prep[cat_cols]).reset_index(drop=True)], axis=1)
    print('Training final model on full encoded data...')
    final_model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=128, n_jobs=-1, random_state=42)
    try:
        final_model.fit(X_full_enc, df_prep[target_col], verbose=50)
    except TypeError:
        final_model.fit(X_full_enc, df_prep[target_col])

    # Build a WrapperPipeline and save
    pipe = WrapperPipeline(encoder=te, model=final_model, feature_cols=cat_cols + numeric_cols, numeric_feats=numeric_cols)

    # Inject into hybrid model
    hybrid = {'type': 'hybrid', 'bus_pipeline': pipe}
    joblib.dump(hybrid, MODEL_PATH)
    print('Saved hybrid with OOF target-encoded LightGBM bus pipeline to', MODEL_PATH)


if __name__ == '__main__':
    train(sample_size=200000)
