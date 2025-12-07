from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Import the CORS middleware
from pydantic import BaseModel
import joblib 
import os
import pandas as pd
import numpy as np
from train_utils import TrainPriceLookup

# --- Configuration for Model Paths ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------
# FastAPI Application Setup
# ---------------------------
app = FastAPI(title="Travel Estimation API")

# ---------------------------
# CORS Configuration (Crucial Fix for "Failed to Fetch")
# ---------------------------
# Replace the list below with the exact URL your frontend is running on.
# Common development URLs are http://localhost:3000, http://localhost:5173, etc.
# Using "*" allows ALL origins during development.
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:8080", # Add any other ports you might be using
    "*" # Allows all origins for maximum compatibility during development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (POST, GET, etc.)
    allow_headers=["*"], # Allow all headers
)

# ---------------------------
# Request Models
# ---------------------------
class AccommodationInput(BaseModel):
    accommodation_type: str
    nights: int
    city: str
    season: str

class ActivityInput(BaseModel):
    activity_type: str
    duration_hours: float
    city: str
    season: str

class FoodInput(BaseModel):
    meal_type: str
    city: str
    days: int
    budget_level: str

class TransportInput(BaseModel):
    transport_type: str
    distance_km: float
    city: str
    passengers: int
    train_class: str | None = None
    bus_type: str | None = None

class EstimateInput(BaseModel):
    startLocation: str
    destination: str
    country: str
    transportType: str
    trainType: str | None = None
    busType: str | None = None
    numberOfDays: int
    accommodationType: str
    foodPreference: str
    activityPreference: str
    distanceKm: float

# ---------------------------
# Global Model Variables
# ---------------------------
accommodation_model = None
activity_model = None
food_model = None
transport_model = None

@app.on_event("startup")
def load_models():
    global accommodation_model, activity_model, food_model, transport_model

    print("Loading models...")

    try:
        def load_pipeline(filename):
            """Loads a model pipeline using joblib."""
            path = os.path.join(MODEL_DIR, filename)
            obj = joblib.load(path)
            # Handle dict format (for transport model with lookup tables)
            if isinstance(obj, dict) and 'pipeline' in obj:
                return obj
            # Otherwise return as-is (for other models)
            return obj

        accommodation_model = load_pipeline("accommodation_model_pipeline.pkl")
        activity_model = load_pipeline("activity_model_pipeline.pkl")
        food_model = load_pipeline("food_model_pipeline.pkl")
        transport_model = load_pipeline("transport_model_pipeline.pkl")
        # If the loaded transport model contains a lookup object, ensure it's populated.
        try:
            if isinstance(transport_model, dict) and 'lookup' in transport_model:
                lk = transport_model.get('lookup')
                needs_reload = False
                if lk is None:
                    needs_reload = True
                else:
                    # If base_map is missing or empty or distance_df missing, reload from CSVs
                    if (not hasattr(lk, 'base_map')) or (hasattr(lk, 'base_map') and not lk.base_map) or (not hasattr(lk, 'distance_df')) or (getattr(lk, 'distance_df', None) is None):
                        needs_reload = True
                if needs_reload:
                    try:
                        p1 = os.path.join(MODEL_DIR, 'Part_1.csv')
                        p2 = os.path.join(MODEL_DIR, 'train_dataset_part2.csv')
                        p1_df = pd.read_csv(p1, encoding='latin-1') if os.path.exists(p1) else None
                        p2_df = pd.read_csv(p2, encoding='latin-1') if os.path.exists(p2) else None
                        transport_model['lookup'] = TrainPriceLookup(p1_df, p2_df)
                        print('Reloaded Train lookup tables from CSV')
                    except Exception as e:
                        print(f'Could not reload train lookup tables: {e}')
        except Exception:
            pass

        print("All models loaded successfully.")
    except Exception as e:
        print(f"Model Loading Error -> {e}")
        # Re-raise the exception to stop the server if loading fails
        # raise e 

# ---------------------------
# Endpoints
# ---------------------------


@app.get('/train/classes')
def train_classes():
    """Return a list of available train classes derived from CSV lookup files and training data."""
    # Default canonical list (always included)
    default_list = [
        'AC_CC','AC_3A','AC_2A','AC_1A','GR_CC','GR_3A','Raj_3A','Raj_2A','Raj_1A',
        'Sha_CC','Sha_EC','JS_2S','JS_CC','Yuva_Other','Yuva_18_35','Ord_LocalSuburb','Ord_Passenger',
        'Exp_2S','Exp_SL','Exp_FC','2S','SL'
    ]

    classes = set(default_list)
    loaded_any = False

    # Try part_1.csv (expects 'Train_Type' column)
    try:
        p1 = os.path.join(MODEL_DIR, 'part_1.csv')
        if os.path.exists(p1):
            try:
                df1 = pd.read_csv(p1)
            except Exception as e:
                try:
                    df1 = pd.read_csv(p1, encoding='latin-1')
                except Exception as e2:
                    print(f"train/classes: failed to read part_1.csv: {e2}")
                    df1 = None
            if df1 is not None and 'Train_Type' in df1.columns:
                classes.update(df1['Train_Type'].astype(str).str.strip().unique().tolist())
                loaded_any = True
    except Exception as e:
        print(f"train/classes: unexpected error reading part_1.csv: {e}")

    # Try train_dataset_part2.csv (columns are distance tiers + class columns)
    try:
        p2 = os.path.join(MODEL_DIR, 'train_dataset_part2.csv')
        if os.path.exists(p2):
            try:
                df2 = pd.read_csv(p2)
            except Exception as e:
                try:
                    df2 = pd.read_csv(p2, encoding='latin-1')
                except Exception as e2:
                    print(f"train/classes: failed to read train_dataset_part2.csv: {e2}")
                    df2 = None
            if df2 is not None:
                for c in df2.columns:
                    if str(c).strip().lower() not in ('distance', 'dist'):
                        classes.add(str(c).strip())
                loaded_any = True
    except Exception as e:
        print(f"train/classes: unexpected error reading train_dataset_part2.csv: {e}")

    # Also include any Train_Class values from the main synthetic dataset
    try:
        sd = os.path.join(MODEL_DIR, 'synthetic_travel_dataset.csv')
        if os.path.exists(sd):
            try:
                ds = pd.read_csv(sd)
            except Exception as e:
                try:
                    ds = pd.read_csv(sd, encoding='latin-1')
                except Exception as e2:
                    print(f"train/classes: failed to read synthetic_travel_dataset.csv: {e2}")
                    ds = None
            if ds is not None and 'Train_Class' in ds.columns:
                classes.update(ds['Train_Class'].astype(str).str.strip().unique().tolist())
                loaded_any = True
    except Exception as e:
        print(f"train/classes: unexpected error reading synthetic_travel_dataset.csv: {e}")

    if not loaded_any:
        print("train/classes: No CSVs loaded or CSV parsing failed. Returning default canonical list.")
    else:
        print("train/classes: Loaded additional classes from CSVs (merged with defaults).")

    # Remove empty / NaN-like entries (safely convert non-strings and trim)
    cleaned = []
    removed = []
    for item in classes:
        if item is None:
            removed.append(str(item))
            continue
        s = str(item).strip()
        if not s:
            removed.append(str(item))
            continue
        if s.lower() in ('nan', 'none', 'null'):
            removed.append(s)
            continue
        cleaned.append(s)

    if removed:
        print(f"train/classes: removed invalid entries -> {removed}")

    classes_list = sorted(list(set(cleaned)))
    return {"train_classes": classes_list}


@app.get('/bus/types')
def bus_types():
    """Return a list of available bus types from the bus fare dataset."""
    types = set()
    loaded = False

    try:
        bus_csv = os.path.join(MODEL_DIR, 'indian_bus_fare_dataset.csv')
        if os.path.exists(bus_csv):
            try:
                df = pd.read_csv(bus_csv)
            except Exception as e:
                try:
                    df = pd.read_csv(bus_csv, encoding='latin-1')
                except Exception as e2:
                    print(f"bus/types: failed to read indian_bus_fare_dataset.csv: {e2}")
                    df = None

            if df is not None:
                # Find the bus type column (case-insensitive)
                bus_col = [c for c in df.columns if 'bus' in c.lower()]
                if bus_col:
                    col_name = bus_col[0]
                    types.update(df[col_name].astype(str).str.strip().unique().tolist())
                    loaded = True
                    print(f"bus/types: loaded {len(types)} bus types from CSV")
    except Exception as e:
        print(f"bus/types: unexpected error: {e}")

    # Default bus types (fallback)
    default_types = ['Non-AC Sleeper', 'Volvo', 'Non-AC Seater', 'Luxury', 'AC Seater', 'AC Sleeper']
    types.update(default_types)

    # Clean and sort
    cleaned = []
    for t in types:
        if t and str(t).strip().lower() not in ('nan', 'none', 'null', ''):
            cleaned.append(str(t).strip())

    types_list = sorted(list(set(cleaned)))
    return {"bus_types": types_list}


@app.post("/predict/accommodation")
def predict_accommodation(data: AccommodationInput):
    if accommodation_model is None:
        return {"error": "Accommodation model not loaded"}

    # 7 required features: Duration_Days, Accommodation_Type, Base_Currency, 
    # User_Selected_Currency, Currency_Rate, Transport_Type, Distance_km
    df = pd.DataFrame({
        'Duration_Days': [data.nights],              
        'Accommodation_Type': [data.accommodation_type],
        'Base_Currency': ['USD'],                    
        'User_Selected_Currency': ['USD'],           
        'Currency_Rate': [1.0],                      
        'Transport_Type': ['Plane'],                 
        'Distance_km': [0.0]                         
    })
    
    df = df.astype({'Duration_Days': np.int64, 'Currency_Rate': np.float64, 'Distance_km': np.float64})

    try:
        pred = accommodation_model.predict(df)[0]
        return {"predicted_cost": float(pred)}
    except Exception as e:
        return {"error": f"Prediction failed with model: {str(e)}"}


@app.post("/predict/activity")
def predict_activity(data: ActivityInput):
    if activity_model is None:
        return {"error": "Activity model not loaded"}

    # 9 required features: Start_Location, Destination, Country, Duration_Days, 
    # Base_Currency, User_Selected_Currency, Currency_Rate, Accommodation_Type, Transport_Type
    
    df = pd.DataFrame({
        'Start_Location': [data.city],              
        'Destination': [data.city],                 
        'Country': ['USA'],                         
        'Duration_Days': [1],                       
        'Base_Currency': ['USD'],                   
        'User_Selected_Currency': ['USD'],          
        'Currency_Rate': [1.0],                     
        'Accommodation_Type': ['Hotel'],            
        'Transport_Type': ['Plane']
    })
    
    df = df.astype({'Duration_Days': np.int64, 'Currency_Rate': np.float64})

    try:
        pred = activity_model.predict(df)[0]
        return {"predicted_cost": float(pred)}
    except Exception as e:
        return {"error": f"Prediction failed with model: {str(e)}"}

@app.post("/predict/food")
def predict_food(data: FoodInput):
    if food_model is None:
        return {"error": "Food model not loaded"}

    # Assumed 11 required features: Food_Type, Budget_Level + 9 Contextual features.
    df = pd.DataFrame({
        'Food_Type': [data.meal_type],              
        'Budget_Level': [data.budget_level],        
        'Duration_Days': [data.days],               
        'Start_Location': [data.city],              
        'Destination': [data.city],                 
        
        # Default/Dummy features:
        'Country': ['USA'],                         
        'Base_Currency': ['USD'],                   
        'User_Selected_Currency': ['USD'],          
        'Currency_Rate': [1.0],                     
        'Accommodation_Type': ['Hotel'],            
        'Transport_Type': ['Plane']
    })
    
    df = df.astype({'Duration_Days': np.int64, 'Currency_Rate': np.float64})

    try:
        pred = food_model.predict(df)[0]
        return {"predicted_cost": float(pred)}
    except Exception as e:
        return {"error": f"Prediction failed with model: {str(e)}"}

@app.post("/predict/transport")
def predict_transport(data: TransportInput):
    if transport_model is None:
        return {"error": "Transport model not loaded"}

    # 12 required features: 
    # Start_Location, Destination, Country, Duration_Days, Base_Currency, User_Selected_Currency, 
    # Currency_Rate, Accommodation_Type, Transport_Type, Distance_km, Passengers, Train_Class
    
    df = pd.DataFrame({
        # Mapped Features:
        'Start_Location': [data.city],               
        'Destination': [data.city],                  
        'Transport_Type': [data.transport_type],
        'Distance_km': [data.distance_km],
        'Passengers': [data.passengers],
        'Train_Class': [data.train_class or 'N/A'],                      

        # Default/Dummy features:
        'Country': ['USA'],                          
        'Duration_Days': [1],                        
        'Accommodation_Type': ['Hotel'],             
        'Base_Currency': ['USD'],                    
        'User_Selected_Currency': ['USD'],           
        'Currency_Rate': [1.0]                       
    })

    # Ensure correct numerical types
    df = df.astype({
        'Duration_Days': np.int64, 
        'Currency_Rate': np.float64, 
        'Distance_km': np.float64, 
        'Passengers': np.int64
    })

    try:
        # If Train, use lookup tables only
        if data.transport_type.lower() == 'train' and isinstance(transport_model, dict) and transport_model.get('lookup'):
            lookup = transport_model.get('lookup')
            base = lookup.get_base_price(data.train_class)
            dist_cost = lookup.get_distance_price(data.train_class, data.distance_km)
            total = base + dist_cost
            return {"predicted_cost": float(total)}

        # If hybrid dict with separate pipelines, use the appropriate one
        if isinstance(transport_model, dict):
            ttype = data.transport_type.lower()
            # Bus
            if ttype == 'bus' and transport_model.get('bus_pipeline') is not None:
                pipe = transport_model.get('bus_pipeline')
                # Build a bus-specific input DataFrame matching the lightweight bus pipeline
                try:
                    # The lightweight bus pipeline expects: Start_Location, Destination, Bus_Type, Total_Seats, Duration_hours
                    duration_hours = max(1.0, float(data.distance_km) / 60.0) if data.distance_km is not None else 1.0
                    total_seats = 40  # default fallback
                    bus_input = pd.DataFrame({
                        'Start_Location': [data.city],
                        'Destination': [getattr(data, 'destination', data.city) if hasattr(data, 'destination') else data.city],
                        'Bus_Type': [data.bus_type or 'Volvo'],
                        'Total_Seats': [total_seats],
                        'Duration_hours': [duration_hours]
                    })
                except Exception:
                    # Fallback to original df if construction fails
                    bus_input = df
                try:
                    print('--- PREDICT_TRANSPORT (BUS) DEBUG START ---')
                    print('Input dataframe for prediction:\n', bus_input.to_dict(orient='records'))
                    pred = None
                    # Two pipeline types supported: sklearn Pipeline with named_steps, or our WrapperPipeline
                    if hasattr(pipe, 'named_steps') and pipe.named_steps.get('preprocessor') is not None:
                        pre = pipe.named_steps.get('preprocessor')
                        try:
                            # Attempt to extract feature names and transformed vector
                            feat_names = []
                            try:
                                for name, trans, cols in pre.transformers:
                                    if trans == 'passthrough':
                                        feat_names.extend(list(cols))
                                    else:
                                        if hasattr(trans, 'get_feature_names_out'):
                                            feat_names.extend(list(trans.get_feature_names_out(cols)))
                                        else:
                                            feat_names.extend(list(cols))
                            except Exception:
                                # fallback
                                try:
                                    num_cols = pre.transformers_[0][2]
                                    cat_cols = pre.transformers_[1][2]
                                    feat_names.extend(list(num_cols))
                                    cat_trans = pre.named_transformers_.get('cat')
                                    if hasattr(cat_trans, 'get_feature_names_out'):
                                        feat_names.extend(list(cat_trans.get_feature_names_out(cat_cols)))
                                    else:
                                        feat_names.extend(list(cat_cols))
                                except Exception:
                                    pass

                            transformed = pre.transform(bus_input)
                            print('Preprocessed feature vector shape:', getattr(transformed, 'shape', None))
                            print('Preprocessed feature names (first 40):', feat_names[:40])
                            print('Preprocessed values (first row, first 40):', transformed.flatten()[:40].tolist())
                        except Exception as e:
                            print('Could not extract preprocessed features:', e)

                        pred = pipe.predict(bus_input)[0]
                        try:
                            print('Raw model prediction (single-way):', float(pred))
                        except Exception:
                            print('Raw model prediction (single-way) (raw):', pred)
                    else:
                        # Fallback for WrapperPipeline (our custom wrapper)
                        try:
                            if hasattr(pipe, 'numeric_feats') and hasattr(pipe, 'encoder') and hasattr(pipe, 'feature_cols'):
                                df_for = bus_input.copy()
                                # ensure all expected feature cols exist
                                for n in pipe.feature_cols:
                                    if n not in df_for.columns:
                                        df_for[n] = 0
                                cat_cols = [c for c in pipe.feature_cols if c not in pipe.numeric_feats]
                                num_df = df_for[pipe.numeric_feats].astype(float).fillna(0.0).reset_index(drop=True)
                                te_df = pipe.encoder.transform(df_for[cat_cols]).reset_index(drop=True)
                                X_pred = pd.concat([num_df, te_df], axis=1)
                                print('Preprocessed feature vector shape:', X_pred.shape)
                                print('Preprocessed values (first row, first 40):', X_pred.values.flatten()[:40].tolist())
                                # Call wrapper predict
                                pr = pipe.predict(bus_input)
                                # wrapper may return array-like
                                pred = pr[0] if hasattr(pr, '__len__') else pr
                                try:
                                    print('Raw model prediction (single-way):', float(pred))
                                except Exception:
                                    print('Raw model prediction (single-way) (raw):', pred)
                            else:
                                # Last-resort: try to call predict and log whatever it returns
                                pr = pipe.predict(bus_input)
                                pred = pr[0] if hasattr(pr, '__len__') else pr
                                print('Raw model prediction (single-way) (raw):', pred)
                        except Exception as e:
                            print('Error during BUS prediction logging (wrapper):', e)
                    print('--- PREDICT_TRANSPORT (BUS) DEBUG END ---')
                except Exception as e:
                    print('Error during BUS prediction logging:', e)
                try:
                    return {"predicted_cost": float(pred)}
                except Exception:
                    return {"error": "Prediction produced non-numeric output"}
            # Flight
            if ttype in ('flight', 'plane', 'air') and transport_model.get('flight_pipeline') is not None:
                pipe = transport_model.get('flight_pipeline')
                pred = pipe.predict(df)[0]
                return {"predicted_cost": float(pred)}
            # Legacy single-pipeline key
            if transport_model.get('pipeline') is not None:
                pipe = transport_model.get('pipeline')
                pred = pipe.predict(df)[0]
                return {"predicted_cost": float(pred)}

        # If transport_model is a single pipeline-like object
        if hasattr(transport_model, 'predict'):
            pred = transport_model.predict(df)[0]
            return {"predicted_cost": float(pred)}

        return {"error": "No suitable transport model available"}
    except Exception as e:
        # Catch any prediction errors that might still occur
        return {"error": f"Prediction failed with model: {str(e)}"}

@app.post("/api/estimate")
def estimate_trip(data: EstimateInput):
    """Aggregate endpoint that calls all prediction models and returns a comprehensive breakdown."""
    try:
        # Currency conversion: USD to INR (1 USD ≈ 83 INR - update this rate as needed)
        USD_TO_INR = 83.0
        
        # Predict accommodation cost (in original currency, will convert if needed)
        accom_pred_res = predict_accommodation(AccommodationInput(
            accommodation_type=data.accommodationType,
            nights=data.numberOfDays,
            city=data.startLocation,
            season="moderate"
        ))
        accommodation_cost = accom_pred_res.get("predicted_cost", 100)

        # Predict activity cost (in original currency, will convert if needed)
        activity_pred_res = predict_activity(ActivityInput(
            activity_type="sightseeing",
            duration_hours=4.0,
            city=data.destination,
            season="moderate"
        ))
        activity_cost = activity_pred_res.get("predicted_cost", 50)

        # Predict food cost based on preference (in USD, convert to INR)
        # Low: $7/day, Medium: $11/day, High: $25/day
        food_cost_usd_map = {"Low": 7 * data.numberOfDays, "Medium": 11 * data.numberOfDays, "High": 25 * data.numberOfDays}
        food_cost_usd = food_cost_usd_map.get(data.foodPreference, 50)
        food_cost = food_cost_usd * USD_TO_INR  # Convert to INR

        # Predict transport cost (in INR from hybrid Train lookup tables or model)
        # Forward the selected trainType and busType into TransportInput
        transport_pred_res = predict_transport(TransportInput(
            transport_type=data.transportType,
            distance_km=data.distanceKm,
            city=data.startLocation,
            passengers=1,
            train_class=(data.trainType if hasattr(data, 'trainType') else None),
            bus_type=(data.busType if hasattr(data, 'busType') else None)
        ))
        transport_cost = transport_pred_res.get("predicted_cost", 200)

        # Multiply transport cost by 2 for round-trip (source to destination and back)
        transport_cost = transport_cost * 2

        # Convert accommodation and activity to INR if they're in USD
        accommodation_cost_inr = accommodation_cost * USD_TO_INR
        activity_cost_inr = activity_cost * USD_TO_INR

        # Calculate total in INR
        total_cost = accommodation_cost_inr + transport_cost + food_cost + activity_cost_inr

        return {
            "estimatedCost": round(total_cost, 2),
            "currency": "INR",
            "breakdown": {
                "accommodation": round(accommodation_cost_inr, 2),
                "transportation": round(transport_cost, 2),
                "food": round(food_cost, 2),
                "activity": round(activity_cost_inr, 2)
            },
            "message": "Estimation calculated successfully in Indian Rupees."
        }
    except Exception as e:
        return {
            "error": f"Estimation failed: {str(e)}",
            "estimatedCost": 0,
            "currency": "INR",
            "breakdown": {"accommodation": 0, "transportation": 0, "food": 0, "activity": 0}
        }