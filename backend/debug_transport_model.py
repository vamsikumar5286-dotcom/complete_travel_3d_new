import os
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, 'transport_model_pipeline.pkl')

def get_feature_names_from_preprocessor(preprocessor):
    # Support ColumnTransformer with two transformers: num and cat
    feature_names = []
    try:
        for name, trans, cols in preprocessor.transformers:
            if trans == 'passthrough':
                feature_names.extend(list(cols))
            else:
                if hasattr(trans, 'get_feature_names_out'):
                    try:
                        # For OneHotEncoder
                        out = trans.get_feature_names_out(cols)
                        feature_names.extend(list(out))
                    except Exception:
                        # For scaler or other transformers, just use cols
                        feature_names.extend(list(cols))
                else:
                    feature_names.extend(list(cols))
    except Exception:
        # Fallback: try attributes on preprocessor
        try:
            # scikit-learn >=1.0
            num_cols = preprocessor.transformers_[0][2]
            cat_cols = preprocessor.transformers_[1][2]
            feature_names.extend(list(num_cols))
            cat_trans = preprocessor.named_transformers_.get('cat')
            if hasattr(cat_trans, 'get_feature_names_out'):
                feature_names.extend(list(cat_trans.get_feature_names_out(cat_cols)))
            else:
                feature_names.extend(list(cat_cols))
        except Exception:
            pass
    return feature_names

def inspect():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return

    obj = joblib.load(MODEL_PATH)
    if not isinstance(obj, dict):
        print('Loaded model is not the expected hybrid dict. Keys:', type(obj))
        return

    bus_pipe = obj.get('bus_pipeline')
    if bus_pipe is None:
        print('No bus_pipeline found in hybrid model.')
        return

    print('Bus pipeline type:', type(bus_pipe))
    if not hasattr(bus_pipe, 'named_steps'):
        print('bus_pipeline has no named_steps; cannot inspect further')
        return

    preprocessor = bus_pipe.named_steps.get('preprocessor')
    regressor = bus_pipe.named_steps.get('regressor')

    if preprocessor is None or regressor is None:
        print('Preprocessor or regressor not found in bus_pipeline')
        return

    feat_names = get_feature_names_from_preprocessor(preprocessor)
    print(f'Number of features after preprocessing: {len(feat_names)}')

    try:
        importances = regressor.feature_importances_
    except Exception as e:
        print('Regressor has no feature_importances_ attribute:', e)
        return

    if len(importances) != len(feat_names):
        print('Warning: number of importances != number of feature names')

    # build pairs and sort
    pairs = []
    for i, imp in enumerate(importances):
        name = feat_names[i] if i < len(feat_names) else f'feature_{i}'
        pairs.append((name, float(imp)))

    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    print('\nTop 30 feature importances for Bus model:')
    for name, imp in pairs_sorted[:30]:
        print(f'  {name}: {imp:.6f}')

    # Print a sample transform for a synthetic row
    try:
        sample = pd.DataFrame({
            'Start_Location': ['Mumbai'],
            'Destination': ['Goa'],
            'Transport_Type': ['Bus'],
            'Distance_km': [500.0],
            'Passengers': [1],
            'Train_Class': ['N/A'],
            'Country': ['India'],
            'Duration_Days': [1],
            'Accommodation_Type': ['Hotel'],
            'Base_Currency': ['INR'],
            'User_Selected_Currency': ['INR'],
            'Currency_Rate': [1.0]
        })
        transformed = preprocessor.transform(sample)
        print('\nPreprocessed sample shape:', transformed.shape)
        print('First 20 preprocessed values:', transformed.flatten()[:20])
    except Exception as e:
        print('Could not transform sample through preprocessor:', e)

if __name__ == '__main__':
    inspect()
