import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
from datetime import datetime
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(title="Sales Prediction API", description="Predict daily sales for Rossmann stores", version="1.0")

pickle_in = open('notebooks/xgboost.pkl','rb')
xgboost = pickle.load(pickle_in)

# Define the input data schema
class PredictionRequest(BaseModel):
    start_date: str  # Format: YYYY-MM-DD
    end_date: str    # Format: YYYY-MM-DD
    store_type: int  # Values: 0, 1, 2, 3

# Define the output data schema
class PredictionResponse(BaseModel):
    Date: str
    PredictedSales: float

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Sales Prediction API. Use /predict to get sales forecasts."}

# Prediction endpoint
@app.post("/predict", response_model=List[PredictionResponse])
async def predict_sales(request: PredictionRequest):
    try:
        # Parse and validate input data
        start_date = pd.to_datetime(request.start_date)
        end_date = pd.to_datetime(request.end_date)
        store_type = request.store_type

        # Validate store type
        if store_type not in [0, 1, 2, 3]:
            raise HTTPException(status_code=400, detail="Invalid store_type. Valid values are: 0, 1, 2, 3.")
        
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date)
        dates_df = pd.DataFrame({"Date": date_range})
        dates_df["Year"] = dates_df["Date"].dt.year
        dates_df["Month"] = dates_df["Date"].dt.month
        dates_df["DayOfWeek"] = dates_df["Date"].dt.dayofweek

        # Add store type
        dates_df["StoreType"] = int(store_type)

        # Ensure columns align with the model's training data
        required_features = model.feature_names_in_
        for col in required_features:
            if col not in dates_df:
                dates_df[col] = 0  # Add missing columns as zeros
        dates_df = dates_df[required_features]

        # Make predictions
        predictions = xgboost.predict(dates_df)
        dates_df["PredictedSales"] = predictions

        # Format response
        results = [
            PredictionResponse(Date=row["Date"].strftime("%Y-%m-%d"), PredictedSales=row["PredictedSales"])
            for _, row in dates_df.iterrows()
        ]
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
#run the API with the uvicorn 
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
        
