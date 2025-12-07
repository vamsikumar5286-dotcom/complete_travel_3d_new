#!/usr/bin/env python3
import joblib
import sys

try:
    model = joblib.load('transport_model_pipeline.pkl')
    print('[OK] Model loaded successfully!')
    print(f'Model type: {model.get("type")}')
    print(f'Bus Pipeline: {type(model.get("bus_pipeline")).__name__}')
    print(f'Flight Pipeline: {type(model.get("flight_pipeline")).__name__}')
    print(f'Train Lookup: {type(model.get("lookup")).__name__}')
    print('\nModel is ready for use!')
except Exception as e:
    print(f'[ERROR] {e}')
    sys.exit(1)
