#!/usr/bin/env python3
"""
train_transportation_model.py

Train a hybrid transportation cost prediction model that:
- Uses ML for Bus fares (from indian_bus_fare_dataset.csv)
- Uses ML for Flight fares (from all flight datasets)
- Uses lookup tables for Train fares (from part_1.csv and train_dataset_part2.csv)

The model returns a dictionary with:
- 'bus_pipeline': ML pipeline for Bus fares
- 'flight_pipeline': ML pipeline for Flight fares  
- 'lookup': TrainPriceLookup object for Train lookups
- 'type': 'hybrid'
"""

import argparse
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TrainPriceLookup:
    """Handles Train pricing lookup from part_1 and part_2 datasets."""
    
    def __init__(self, part_1_df=None, part_2_df=None):
        self.base_map = {}
        self.distance_df = None
        
        if part_1_df is not None:
            self._load_base_prices(part_1_df)
        if part_2_df is not None:
            self._load_distance_prices(part_2_df)
    
    def _load_base_prices(self, df):
        """Load base prices from part_1.csv using the 'Base price' column."""
        try:
            df = df.copy()
            cols_lower = {c.lower(): c for c in df.columns}
            
            if 'base price' in cols_lower:
                base_col = cols_lower['base price']
                key_col = df.columns[0]
                self.base_map = dict(zip(
                    df[key_col].astype(str).str.strip(), 
                    pd.to_numeric(df[base_col], errors='coerce').fillna(0.0).astype(float)
                ))
                print(f"[OK] Loaded base prices for {len(self.base_map)} train types from part_1.csv")
                return
            print("Warning: Could not detect base price column in part_1.csv")
        except Exception as e:
            print(f"Warning: Could not load base prices: {e}")
    
    def _load_distance_prices(self, df):
        """Load distance price lookup table from train_dataset_part2.csv."""
        try:
            self.distance_df = df.copy()
            print(f"[OK] Loaded distance price table with {len(self.distance_df)} rows from train_dataset_part2.csv")
        except Exception as e:
            print(f"Warning: Could not load distance prices: {e}")
    
    def get_base_price(self, train_type):
        """Get base price for a train type."""
        if not train_type:
            return 0.0
        
        name_mapping = {
            'ordinary passenger': 'Ord_Passenger',
            'ordinary': 'Ord_Passenger',
            'ordinary local': 'Ord_LocalSuburb',
            'ordinary local suburb': 'Ord_LocalSuburb',
            'ordinary sleeper': 'Ord_SL',
            'ordinary first class': 'Ord_FC',
            'express 2s': 'Exp_2S',
            'express sleeper': 'Exp_SL',
            'express first class': 'Exp_FC',
            'ac chair car': 'AC_CC',
            'ac 3 tier': 'AC_3A',
            'ac 3a': 'AC_3A',
            'ac 2 tier': 'AC_2A',
            'ac 2a': 'AC_2A',
            'ac 1 tier': 'AC_1A',
            'ac 1a': 'AC_1A',
            'garibrath chair car': 'GR_CC',
            'garibrath 3 tier': 'GR_3A',
            'rajdhani 3 tier': 'Raj_3A',
            'rajdhani 2 tier': 'Raj_2A',
            'rajdhani 1 tier': 'Raj_1A',
            'shatabdi chair car': 'Sha_CC',
            'shatabdi executive': 'Sha_EC',
            'jan shatabdi 2s': 'JS_2S',
            'jan shatabdi chair car': 'JS_CC',
            'yuva': 'Yuva_Other',
            'yuva 18-35': 'Yuva_18_35',
        }
        
        key = str(train_type).strip().lower()
        mapped_key = name_mapping.get(key)
        
        if mapped_key and mapped_key in self.base_map:
            return float(self.base_map[mapped_key])
        
        key_orig = str(train_type).strip()
        if key_orig in self.base_map:
            return float(self.base_map[key_orig])
        
        lowered = {k.lower(): v for k, v in self.base_map.items()}
        return float(lowered.get(key, 0.0))
    
    def get_distance_price(self, train_type, distance_km):
        """Get distance price by matching distance tier and train class column."""
        if self.distance_df is None or distance_km is None:
            return 0.0
        
        try:
            import re
            df = self.distance_df.copy()
            
            if 'Distance' in df.columns:
                df['dist_mid'] = df['Distance'].apply(self._parse_distance_tier)
            else:
                return 0.0
            
            df = df.dropna(subset=['dist_mid'])
            if df.empty:
                return 0.0

            class_to_cols = {
                'ordinary': ['Ord_LocalSuburb', 'Ord_Passenger', 'Ord_SL', 'Ord_FC'],
                'ordinary passenger': ['Ord_Passenger', 'Ord_LocalSuburb', 'Ord_SL', 'Ord_FC'],
                'ordinary local': ['Ord_LocalSuburb', 'Ord_Passenger', 'Ord_SL', 'Ord_FC'],
                'express': ['Exp_2S', 'Exp_SL', 'Exp_FC'],
                'ac': ['AC_CC', 'AC_3A', 'AC_2A', 'AC_1A'],
                'garibrath': ['GR_CC', 'GR_3A'],
                'rajdhani': ['Raj_3A', 'Raj_2A', 'Raj_1A'],
                'shatabdi': ['Sha_CC', 'Sha_EC'],
                'jan-shatabdi': ['JS_2S', 'JS_CC'],
                'yuva': ['Yuva_Other', 'Yuva_18_35']
            }
            
            train_type_lower = str(train_type).lower() if train_type else ''
            candidates = class_to_cols.get(train_type_lower, [])
            
            if not candidates:
                return 0.0

            distance_float = float(distance_km)
            df['delta'] = (df['dist_mid'] - distance_float).abs()
            closest_row = df.loc[df['delta'].idxmin()]
            
            for col in candidates:
                if col in closest_row.index:
                    try:
                        val = pd.to_numeric(closest_row[col], errors='coerce')
                        if not pd.isna(val) and float(val) > 0:
                            return float(val)
                    except Exception:
                        pass
            
            return 0.0
        except Exception as e:
            print(f"Warning: Could not get distance price: {e}")
            return 0.0
    
    @staticmethod
    def _parse_distance_tier(s):
        """Parse distance tier strings like '01-05' or '16-20' into midpoint."""
        try:
            import re
            if pd.isna(s):
                return None
            s = str(s).strip()
            nums = re.findall(r"\d+", s)
            if not nums:
                return None
            nums = [int(n) for n in nums]
            if len(nums) == 1:
                return float(nums[0])
            a, b = nums[0], nums[1]
            return float((a + b) / 2.0)
        except Exception:
            return None


def load_bus_data(path):
    """Load and prepare bus fare data from indian_bus_fare_dataset.csv"""
    print(f"Loading Bus data from {path}...")
    df = pd.read_csv(path)
    print(f"  Bus dataset shape: {df.shape}")
    print(f"  Bus dataset columns: {df.columns.tolist()}")
    return df


def load_flight_data(flight_data_paths):
    """Load and combine all flight datasets."""
    print("Loading Flight data...")
    flight_dfs = []
    
    for path in flight_data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  Loaded {path}: shape {df.shape}")
            flight_dfs.append(df)
        else:
            print(f"  Warning: {path} not found")
    
    if flight_dfs:
        combined = pd.concat(flight_dfs, ignore_index=True)
        print(f"  Combined flight datasets: shape {combined.shape}")
        return combined
    else:
        print("  No flight data files found!")
        return None


def prepare_bus_data(df):
    """Prepare bus data for ML model."""
    print("\nPreparing Bus data...")
    
    if df.empty:
        print("  Bus dataset is empty!")
        return None, None, [], []
    
    # Find target column (price/fare/cost)
    target_col = None
    for col in df.columns:
        if 'price' in col.lower() or 'fare' in col.lower() or 'cost' in col.lower():
            target_col = col
            break
    
    if target_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[-1]
        else:
            print("  No numeric target column found in bus data!")
            return None, None, [], []
    
    print(f"  Using '{target_col}' as target variable")
    
    df = df.dropna(subset=[target_col])
    
    if df.empty:
        print("  No valid data after removing missing values!")
        return None, None, [], []
    
    feature_cols = [col for col in df.columns if col != target_col]
    
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    print(f"  Numeric features ({len(numeric_features)}): {numeric_features[:5] if numeric_features else 'None'}...")
    print(f"  Categorical features ({len(categorical_features)}): {categorical_features[:5] if categorical_features else 'None'}...")
    
    X = df[feature_cols].copy()
    y = pd.to_numeric(df[target_col], errors='coerce')
    
    mask = ~y.isna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    print(f"  Final Bus dataset: {X.shape[0]} samples")
    
    return X, y, numeric_features, categorical_features


def prepare_flight_data(df):
    """Prepare flight data for ML model."""
    print("\nPreparing Flight data...")
    
    if df is None or df.empty:
        print("  Flight dataset is empty!")
        return None, None, [], []
    
    target_col = None
    for col in df.columns:
        if 'price' in col.lower() or 'fare' in col.lower() or 'cost' in col.lower():
            target_col = col
            break
    
    if target_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[-1]
        else:
            print("  No numeric target column found in flight data!")
            return None, None, [], []
    
    print(f"  Using '{target_col}' as target variable")
    
    df = df.dropna(subset=[target_col])
    
    if df.empty:
        print("  No valid data after removing missing values!")
        return None, None, [], []
    
    feature_cols = [col for col in df.columns if col != target_col]
    
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    print(f"  Numeric features ({len(numeric_features)}): {numeric_features[:5] if numeric_features else 'None'}...")
    print(f"  Categorical features ({len(categorical_features)}): {categorical_features[:5] if categorical_features else 'None'}...")
    
    X = df[feature_cols].copy()
    y = pd.to_numeric(df[target_col], errors='coerce')
    
    mask = ~y.isna()
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    print(f"  Final Flight dataset: {X.shape[0]} samples")
    
    return X, y, numeric_features, categorical_features


def build_ml_pipeline(X_bus, y_bus, X_flight, y_flight):
    """Build ML models for Bus and Flight."""
    print("\n" + "="*60)
    print("BUILDING ML MODELS")
    print("="*60)
    
    bus_pipeline = None
    flight_pipeline = None
    
    if X_bus is not None and y_bus is not None and len(X_bus) > 0:
        print("\nBuilding Bus fare prediction model...")
        try:
            bus_model = _build_single_model(X_bus, y_bus, "Bus")
            bus_pipeline = bus_model
        except Exception as e:
            print(f"  Error building bus model: {e}")
    
    if X_flight is not None and y_flight is not None and len(X_flight) > 0:
        print("\nBuilding Flight fare prediction model...")
        try:
            flight_model = _build_single_model(X_flight, y_flight, "Flight")
            flight_pipeline = flight_model
        except Exception as e:
            print(f"  Error building flight model: {e}")
    
    return bus_pipeline, flight_pipeline


def _build_single_model(X, y, mode_name):
    """Build a single regression model."""
    print(f"  {mode_name} - Dataset size: {X.shape}")
    
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
    X[categorical_features] = X[categorical_features].fillna("UNKNOWN")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features) if numeric_features else ("num", "passthrough", []),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features) if categorical_features else ("cat", "passthrough", []),
        ],
        remainder="drop"
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            verbose=0
        ))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"  Training {mode_name} model...")
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"  {mode_name} Model Evaluation:")
    print(f"    MAE: {mae:.2f}")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    R2: {r2:.4f}")
    
    return pipeline


def build_and_save_hybrid_model(bus_pipeline, flight_pipeline, train_lookup, output_path):
    """Build hybrid model combining ML and lookup tables."""
    print("\n" + "="*60)
    print("CREATING HYBRID TRANSPORTATION MODEL")
    print("="*60)
    
    hybrid_model = {
        'type': 'hybrid',
        'bus_pipeline': bus_pipeline,
        'flight_pipeline': flight_pipeline,
        'lookup': train_lookup,
    }
    
    print(f"\nSaving hybrid model to {output_path}...")
    joblib.dump(hybrid_model, output_path)
    print(f"Model saved successfully!")
    
    return hybrid_model


def main(args):
    """Main training pipeline."""
    print("="*60)
    print("HYBRID TRANSPORTATION COST PREDICTION MODEL")
    print("="*60)
    
    bus_df = None
    flight_df = None
    
    bus_path = os.path.join(os.path.dirname(__file__), args.bus_data)
    if os.path.exists(bus_path):
        bus_df = load_bus_data(bus_path)
    else:
        print(f"Warning: Bus data file not found: {bus_path}")
    
    flight_data_paths = [
        os.path.join(os.path.dirname(__file__), f)
        for f in args.flight_data.split(',')
    ]
    flight_df = load_flight_data(flight_data_paths)
    
    print("\nLoading Train data...")
    train_dir = os.path.dirname(__file__)
    part_1_path = os.path.join(train_dir, args.part_1_csv)
    part_2_path = os.path.join(train_dir, args.part_2_csv)
    
    part_1_df = None
    part_2_df = None
    
    if os.path.exists(part_1_path):
        part_1_df = pd.read_csv(part_1_path, encoding='latin-1')
        print(f"  Loaded part_1.csv: shape {part_1_df.shape}")
    else:
        print(f"  Warning: {part_1_path} not found")
    
    if os.path.exists(part_2_path):
        part_2_df = pd.read_csv(part_2_path, encoding='latin-1')
        print(f"  Loaded part_2.csv: shape {part_2_df.shape}")
    else:
        print(f"  Warning: {part_2_path} not found")
    
    train_lookup = TrainPriceLookup(part_1_df, part_2_df)
    
    X_bus, y_bus, bus_num_features, bus_cat_features = prepare_bus_data(bus_df) if bus_df is not None else (None, None, [], [])
    X_flight, y_flight, flight_num_features, flight_cat_features = prepare_flight_data(flight_df) if flight_df is not None else (None, None, [], [])
    
    bus_pipeline, flight_pipeline = build_ml_pipeline(X_bus, y_bus, X_flight, y_flight)
    
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    hybrid_model = build_and_save_hybrid_model(bus_pipeline, flight_pipeline, train_lookup, output_path)
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"\nHybrid model saved to: {output_path}")
    print("\nModel components:")
    print(f"  - Bus ML Pipeline: {'Yes' if bus_pipeline else 'No'}")
    print(f"  - Flight ML Pipeline: {'Yes' if flight_pipeline else 'No'}")
    print(f"  - Train Lookup Tables: {'Yes' if train_lookup else 'No'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train hybrid transportation cost prediction model"
    )
    parser.add_argument(
        "--bus-data",
        type=str,
        default="indian_bus_fare_dataset.csv",
        help="Path to bus fare dataset"
    )
    parser.add_argument(
        "--flight-data",
        type=str,
        default="flight_data_BOM_BLR.csv,flight_data_DEL_BLR.csv,flight_data_DEL_BOM.csv,flight_data_DEL_CCU.csv,flight_data_DEL_HYD.csv",
        help="Comma-separated paths to flight datasets"
    )
    parser.add_argument(
        "--part-1-csv",
        type=str,
        default="Part_1.csv",
        help="Path to train part_1.csv (base prices)"
    )
    parser.add_argument(
        "--part-2-csv",
        type=str,
        default="train_dataset_part2.csv",
        help="Path to train part_2.csv (distance prices)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="transport_model_pipeline.pkl",
        help="Output path for the trained model"
    )
    
    args = parser.parse_args()
    main(args)
