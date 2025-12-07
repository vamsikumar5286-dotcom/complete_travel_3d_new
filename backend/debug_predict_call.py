import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'transport_model_pipeline.pkl')

m = joblib.load(MODEL_PATH)
print('Loaded hybrid keys:', list(m.keys()))
pipe = m.get('bus_pipeline')
print('Bus pipeline:', type(pipe))

bus_input = pd.DataFrame({
    'Start_Location': ['Mumbai'],
    'Destination': ['Goa'],
    'Bus_Type': ['Volvo'],
    'Total_Seats': [40],
    'Duration_hours': [500.0/60.0]
})
print('Bus input:', bus_input.to_dict(orient='records'))
transformed = None
# If this is an sklearn Pipeline with named_steps, try to access its preprocessor
if hasattr(pipe, 'named_steps') and pipe.named_steps.get('preprocessor') is not None:
    pre = pipe.named_steps.get('preprocessor')
    try:
        transformed = pre.transform(bus_input)
        print('Preprocessed shape:', getattr(transformed, 'shape', None))
        vals = transformed.flatten()[:40]
        print('Preprocessed (first 40):', vals.tolist())
    except Exception as e:
        print('Preprocess failed:', e)
else:
    # Attempt to use our WrapperPipeline internals (encoder + numeric_feats)
    try:
        if hasattr(pipe, 'numeric_feats') and hasattr(pipe, 'encoder'):
            # Build numeric part; ensure numeric cols exist like the WrapperPipeline does
            df = bus_input.copy()
            for n in pipe.feature_cols:
                if n not in df.columns:
                    df[n] = 0
            cat_cols = [c for c in pipe.feature_cols if c not in pipe.numeric_feats]
            num_df = df[pipe.numeric_feats].astype(float).fillna(0.0).reset_index(drop=True)
            te_df = pipe.encoder.transform(df[cat_cols]).reset_index(drop=True)
            X_pred = pd.concat([num_df, te_df], axis=1)
            print('Preprocessed shape (Wrapper):', X_pred.shape)
            vals = X_pred.values.flatten()[:40]
            print('Preprocessed (first 40) (Wrapper):', vals.tolist())
            transformed = X_pred
        else:
            print('No preprocessor or wrapper encoder available to show transformed features.')
    except Exception as e:
        print('Wrapper preprocessor transform failed:', e)

pred = pipe.predict(bus_input)
try:
    p0 = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
    print('Predicted single-way fare:', p0)
except Exception:
    print('Predicted output (raw):', pred)
