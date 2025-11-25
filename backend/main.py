from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # Import the CORS middleware
from pydantic import BaseModel
import joblib 
import os
import pandas as pd
import numpy as np

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
            return joblib.load(path)

        accommodation_model = load_pipeline("accommodation_model_pipeline.pkl")
        activity_model = load_pipeline("activity_model_pipeline.pkl")
        food_model = load_pipeline("food_model_pipeline.pkl")
        transport_model = load_pipeline("transport_model_pipeline.pkl")
        
        print("✅ All models loaded successfully.")
    except Exception as e:
        print(f"❌ Model Loading Error → {e}")
        # Re-raise the exception to stop the server if loading fails
        # raise e 

# ---------------------------
# Endpoints
# ---------------------------

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
        'Train_Class': ['N/A'],                      

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
        pred = transport_model.predict(df)[0]
        return {"predicted_cost": float(pred)}
    except Exception as e:
        # Catch any prediction errors that might still occur
        return {"error": f"Prediction failed with model: {str(e)}"}