#!/usr/bin/env python3
"""
Project Evaluation Script
Evaluates the Travel Cost Estimator project using 6 different metrics
"""

import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Load models
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

print("="*70)
print("TRAVEL COST ESTIMATOR - PROJECT EVALUATION")
print("="*70)

# Load test data
try:
    df = pd.read_csv(os.path.join(MODEL_DIR, 'synthetic_travel_dataset.csv'))
    print(f"\n[OK] Loaded dataset: {df.shape[0]} samples, {df.shape[1]} features")
except Exception as e:
    print(f"\n[ERROR] Error loading dataset: {e}")
    exit(1)

# Load models
models = {}
model_files = {
    'accommodation': 'accommodation_model_pipeline.pkl',
    'activity': 'activity_model_pipeline.pkl',
    'food': 'food_model_pipeline.pkl',
    'transport': 'transport_model_pipeline.pkl'
}

for name, file in model_files.items():
    try:
        models[name] = joblib.load(os.path.join(MODEL_DIR, file))
        print(f"[OK] Loaded {name} model")
    except Exception as e:
        print(f"[ERROR] Failed to load {name} model: {e}")

print("\n" + "="*70)
print("EVALUATION METRICS")
print("="*70)

# ============================================================================
# METRIC 1: Mean Absolute Error (MAE)
# ============================================================================
print("\n1. MEAN ABSOLUTE ERROR (MAE)")
print("-" * 70)
print("Description: Average absolute difference between predicted and actual")
print("values. Lower is better. Measures prediction accuracy in original units.")
print()

mae_results = {}

# Accommodation MAE
if 'accommodation' in models:
    try:
        X_acc = df[['Duration_Days', 'Accommodation_Type', 'Base_Currency', 
                    'User_Selected_Currency', 'Currency_Rate', 'Transport_Type', 'Distance_km']].copy()
        y_acc = pd.to_numeric(df['Accommodation_Cost'], errors='coerce')
        mask = ~y_acc.isna()
        X_acc = X_acc[mask].head(100)
        y_acc = y_acc[mask].head(100)
        
        pred_acc = models['accommodation'].predict(X_acc)
        mae_acc = mean_absolute_error(y_acc, pred_acc)
        mae_results['Accommodation'] = mae_acc
        print(f"  Accommodation MAE: ${mae_acc:.2f}")
    except Exception as e:
        print(f"  Accommodation MAE: Error - {e}")

# Activity MAE
if 'activity' in models:
    try:
        X_act = df[['Start_Location', 'Destination', 'Country', 'Duration_Days',
                    'Base_Currency', 'User_Selected_Currency', 'Currency_Rate',
                    'Accommodation_Type', 'Transport_Type']].copy()
        y_act = pd.to_numeric(df['Activity_Cost'], errors='coerce')
        mask = ~y_act.isna()
        X_act = X_act[mask].head(100)
        y_act = y_act[mask].head(100)
        
        pred_act = models['activity'].predict(X_act)
        mae_act = mean_absolute_error(y_act, pred_act)
        mae_results['Activity'] = mae_act
        print(f"  Activity MAE: ${mae_act:.2f}")
    except Exception as e:
        print(f"  Activity MAE: Error - {e}")

avg_mae = np.mean(list(mae_results.values())) if mae_results else 0
print(f"\n  >> Average MAE across models: ${avg_mae:.2f}")

# ============================================================================
# METRIC 2: Root Mean Squared Error (RMSE)
# ============================================================================
print("\n2. ROOT MEAN SQUARED ERROR (RMSE)")
print("-" * 70)
print("Description: Square root of average squared differences. Penalizes")
print("larger errors more heavily. Lower is better.")
print()

rmse_results = {}

if 'accommodation' in models:
    try:
        rmse_acc = np.sqrt(mean_squared_error(y_acc, pred_acc))
        rmse_results['Accommodation'] = rmse_acc
        print(f"  Accommodation RMSE: ${rmse_acc:.2f}")
    except:
        pass

if 'activity' in models:
    try:
        rmse_act = np.sqrt(mean_squared_error(y_act, pred_act))
        rmse_results['Activity'] = rmse_act
        print(f"  Activity RMSE: ${rmse_act:.2f}")
    except:
        pass

avg_rmse = np.mean(list(rmse_results.values())) if rmse_results else 0
print(f"\n  >> Average RMSE across models: ${avg_rmse:.2f}")

# ============================================================================
# METRIC 3: R² Score (Coefficient of Determination)
# ============================================================================
print("\n3. R² SCORE (COEFFICIENT OF DETERMINATION)")
print("-" * 70)
print("Description: Proportion of variance explained by the model. Range: 0-1.")
print("Higher is better. 1.0 = perfect predictions, 0.0 = baseline model.")
print()

r2_results = {}

if 'accommodation' in models:
    try:
        r2_acc = r2_score(y_acc, pred_acc)
        r2_results['Accommodation'] = r2_acc
        print(f"  Accommodation R²: {r2_acc:.4f} ({r2_acc*100:.2f}% variance explained)")
    except:
        pass

if 'activity' in models:
    try:
        r2_act = r2_score(y_act, pred_act)
        r2_results['Activity'] = r2_act
        print(f"  Activity R²: {r2_act:.4f} ({r2_act*100:.2f}% variance explained)")
    except:
        pass

avg_r2 = np.mean(list(r2_results.values())) if r2_results else 0
print(f"\n  >> Average R2 across models: {avg_r2:.4f}")

# ============================================================================
# METRIC 4: Mean Absolute Percentage Error (MAPE)
# ============================================================================
print("\n4. MEAN ABSOLUTE PERCENTAGE ERROR (MAPE)")
print("-" * 70)
print("Description: Average percentage error between predictions and actuals.")
print("Scale-independent metric. Lower is better. <10% = excellent, <20% = good.")
print()

mape_results = {}

if 'accommodation' in models:
    try:
        mape_acc = mean_absolute_percentage_error(y_acc, pred_acc) * 100
        mape_results['Accommodation'] = mape_acc
        print(f"  Accommodation MAPE: {mape_acc:.2f}%")
    except:
        pass

if 'activity' in models:
    try:
        mape_act = mean_absolute_percentage_error(y_act, pred_act) * 100
        mape_results['Activity'] = mape_act
        print(f"  Activity MAPE: {mape_act:.2f}%")
    except:
        pass

avg_mape = np.mean(list(mape_results.values())) if mape_results else 0
print(f"\n  >> Average MAPE across models: {avg_mape:.2f}%")

# ============================================================================
# METRIC 5: API Response Time & Throughput
# ============================================================================
print("\n5. API RESPONSE TIME & THROUGHPUT")
print("-" * 70)
print("Description: Measures backend performance - how fast predictions are made.")
print("Lower response time and higher throughput indicate better performance.")
print()

response_times = []
num_requests = 50

for i in range(num_requests):
    try:
        start = time.time()
        # Simulate prediction
        sample = X_acc.iloc[i % len(X_acc):i % len(X_acc) + 1]
        _ = models['accommodation'].predict(sample)
        end = time.time()
        response_times.append((end - start) * 1000)  # Convert to ms
    except:
        pass

if response_times:
    avg_response = np.mean(response_times)
    p95_response = np.percentile(response_times, 95)
    throughput = 1000 / avg_response  # requests per second
    
    print(f"  Average Response Time: {avg_response:.2f} ms")
    print(f"  95th Percentile Response Time: {p95_response:.2f} ms")
    print(f"  Throughput: {throughput:.2f} requests/second")
    print(f"\n  >> Performance Rating: {'Excellent' if avg_response < 100 else 'Good' if avg_response < 500 else 'Needs Improvement'}")

# ============================================================================
# METRIC 6: Model Robustness & Error Distribution
# ============================================================================
print("\n6. MODEL ROBUSTNESS & ERROR DISTRIBUTION")
print("-" * 70)
print("Description: Analyzes prediction error patterns and model stability.")
print("Checks for outliers and consistent performance across data ranges.")
print()

if 'accommodation' in models:
    try:
        errors = np.abs(y_acc - pred_acc)
        
        print(f"  Error Statistics:")
        print(f"    Min Error: ${errors.min():.2f}")
        print(f"    Max Error: ${errors.max():.2f}")
        print(f"    Median Error: ${errors.median():.2f}")
        print(f"    Std Dev: ${errors.std():.2f}")
        
        # Percentage of predictions within acceptable range
        within_10_pct = (errors / y_acc <= 0.10).sum() / len(errors) * 100
        within_20_pct = (errors / y_acc <= 0.20).sum() / len(errors) * 100
        
        print(f"\n  Prediction Accuracy:")
        print(f"    Within 10% of actual: {within_10_pct:.1f}%")
        print(f"    Within 20% of actual: {within_20_pct:.1f}%")
        
        print(f"\n  >> Robustness Rating: {'Excellent' if within_10_pct > 70 else 'Good' if within_10_pct > 50 else 'Fair'}")
    except Exception as e:
        print(f"  Error analyzing robustness: {e}")

# ============================================================================
# OVERALL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("OVERALL PROJECT EVALUATION SUMMARY")
print("="*70)

print(f"\n1. Prediction Accuracy (MAE): ${avg_mae:.2f}")
print(f"2. Error Magnitude (RMSE): ${avg_rmse:.2f}")
print(f"3. Model Fit (R2): {avg_r2:.4f} ({avg_r2*100:.1f}%)")
print(f"4. Percentage Error (MAPE): {avg_mape:.2f}%")
print(f"5. Response Time: {avg_response:.2f} ms")
print(f"6. Robustness: {within_10_pct:.1f}% within 10% accuracy")

# Overall grade
overall_score = 0
if avg_mape < 15: overall_score += 20
elif avg_mape < 25: overall_score += 15
if avg_r2 > 0.8: overall_score += 20
elif avg_r2 > 0.6: overall_score += 15
if avg_response < 100: overall_score += 20
elif avg_response < 200: overall_score += 15
if within_10_pct > 60: overall_score += 20
elif within_10_pct > 40: overall_score += 15
overall_score += 20  # Base score for working system

print(f"\n{'='*70}")
print(f"OVERALL PROJECT SCORE: {overall_score}/100")
print(f"GRADE: {'A+' if overall_score >= 90 else 'A' if overall_score >= 80 else 'B' if overall_score >= 70 else 'C'}")
print(f"{'='*70}\n")
