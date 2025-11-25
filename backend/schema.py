# schema.py
from pydantic import BaseModel, Field
from typing import Optional

class TripRequest(BaseModel):
    Start_Location: Optional[str] = Field(None)
    Destination: Optional[str] = Field(None)
    Country: Optional[str] = Field(None)
    Duration_Days: Optional[float] = Field(None)
    Transport_Type: Optional[str] = Field(None)
    Train_Class: Optional[str] = Field(None)
    Accommodation_Type: Optional[str] = Field(None)
    Distance_km: Optional[float] = Field(None)

class TripResponse(BaseModel):
    Transport_Cost: float
    Accommodation_Cost: float
    Food_Cost: float
    Activity_Cost: float
    Total_Expense: float