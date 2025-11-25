#!/usr/bin/env python3
"""
train_activity_model.py
Train XGBoost to predict Activity_Cost.

Default usage:
python train_activity_model.py --data synthetic_travel_dataset.csv --out activity_model_pipeline.pkl
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

REQUIRED_COLUMNS = [
    "Start_Location",
    "Destination",
    "Country",
    "Duration_Days",
    "Base_Currency",
    "User_Selected_Currency",
    "Currency_Rate",
    "Accommodation_Type",
    "Transport_Type",
    "Activity_Cost",
]

def load_and_prepare(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")
    df = df.copy()
    df = df.dropna(subset=["Activity_Cost"])
    return df

def build_and_train(df, out_path, random_state=42):
    feature_cols = [
        "Start_Location",
        "Destination",
        "Country",
        "Duration_Days",
        "Base_Currency",
        "User_Selected_Currency",
        "Currency_Rate",
        "Accommodation_Type",
        "Transport_Type",
    ]
    target_col = "Activity_Cost"

    X = df[feature_cols].copy()
    y = pd.to_numeric(df[target_col], errors="coerce")
    mask = ~y.isna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    numeric_features = ["Duration_Days", "Currency_Rate"]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
    X[categorical_features] = X[categorical_features].fillna("MISSING")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
    )

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("xgb", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)

    pipeline.fit(
        X_train,
        y_train
    )

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print("Activity model evaluation:")
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2  : {r2:.4f}")

    joblib.dump(pipeline, out_path)
    print(f"Saved pipeline to {out_path}")

def main(args):
    df = load_and_prepare(args.data)
    build_and_train(df, args.out, random_state=args.random_state)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Activity Cost XGBoost model")
    parser.add_argument("--data", type=str, default="synthetic_travel_dataset.csv", help="Path to CSV file")
    parser.add_argument("--out", type=str, default="activity_model_pipeline.pkl", help="Output pipeline file")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)