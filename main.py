from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ml-lab-backend-production.up.railway.app",
        "http://localhost:3000",  # For local frontend
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

try:
    dt_model = joblib.load(os.path.join(MODEL_DIR, 'decision_tree.joblib'))
    lr_model = joblib.load(os.path.join(MODEL_DIR, 'logistic_regression.joblib'))
    print("Models loaded successfully from backend/models/")
except Exception as e:
    print(f"Error loading models: {e}")
    dt_model = None
    lr_model = None

class PredictionRequest(BaseModel):
    Food: Optional[str] = None
    Music: Optional[str] = None
    Clothing: Optional[str] = None
    Language: Optional[str] = None
    Landmark: Optional[str] = None
    
    class Config:
        extra = "allow"

def get_prediction_dataframe(request: PredictionRequest):
    # Get all available options for defaults
    default_options = {
        "Food": ['Sushi', 'Pizza', 'Tacos', 'Injera', 'Curry', 'Burger', 'Croissant'],
        "Music": ['J-Pop', 'Opera', 'Mariachi', 'Ethio-Jazz', 'Bollywood', 'Pop', 'Chanson'],
        "Clothing": ['Kimono', 'Designer Suit', 'Poncho', 'Habesha Kemis', 'Saree', 'Jeans', 'Beret'],
        "Language": ['Japanese', 'Italian', 'Spanish', 'Amharic', 'Hindi', 'English', 'French'],
        "Landmark": ['Mt. Fuji', 'Colosseum', 'Chichen Itza', 'Lalibela', 'Taj Mahal', 'Statue of Liberty', 'Eiffel Tower']
    }
    
    # Convert request to DataFrame, using defaults for missing fields
    request_dict = request.dict()
    data = {}
    for field in ['Food', 'Music', 'Clothing', 'Language', 'Landmark']:
        if field in request_dict and request_dict[field]:
            data[field] = [request_dict[field]]
        else:
            # Use first option as default
            data[field] = [default_options[field][0]]
    
    return pd.DataFrame(data)

@app.get("/")
def read_root():
    return {"message": "Global Culture Predictor API"}

@app.post("/predict/dt")
def predict_dt(request: PredictionRequest):
    if not dt_model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    request_dict = request.dict()
    # Check if we have at least one field
    if not any(request_dict.get(field) for field in ['Food', 'Music', 'Clothing', 'Language', 'Landmark']):
        raise HTTPException(status_code=400, detail="At least one field required")
        
    df = get_prediction_dataframe(request)
    
    try:
        prediction = dt_model.predict(df)[0]
        proba = dt_model.predict_proba(df)[0]
        confidence = max(proba)
        
        return {
            "model": "Decision Tree",
            "prediction": str(prediction),
            "confidence": float(confidence),
            "path": "Analyzed features -> Matched pattern -> Result" 
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/lr")
def predict_lr(request: PredictionRequest):
    if not lr_model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    request_dict = request.dict()
    # Check if we have at least one field
    if not any(request_dict.get(field) for field in ['Food', 'Music', 'Clothing', 'Language', 'Landmark']):
        raise HTTPException(status_code=400, detail="At least one field required")

    df = get_prediction_dataframe(request)
    
    try:
        prediction = lr_model.predict(df)[0]
        proba = lr_model.predict_proba(df)[0]
        confidence = max(proba)
        
        # Get class labels from the classifier step
        classes = lr_model.named_steps['classifier'].classes_
        
        all_probs = {
            str(country): float(prob) 
            for country, prob in zip(classes, proba)
        }
        
        return {
            "model": "Logistic Regression",
            "prediction": str(prediction),
            "confidence": float(confidence),
            "probabilities": all_probs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/options")
def get_options():
    # Extract categories from the encoder in the pipeline
    # Pipeline -> preprocessor (ColumnTransformer) -> transformers_ -> [0] -> (name, transformer, columns)
    if not lr_model:
        return {}
        
    try:
        # Access the OneHotEncoder object
        # Note: transformers_ returns a list of tuples. The first one is our 'cat' transformer.
        # The transformer object is the second element [1].
        encoder = lr_model.named_steps['preprocessor'].transformers_[0][1]
        
        # categories_ is a list of arrays, one for each feature
        categories = encoder.categories_
        feature_names = ['Food', 'Music', 'Clothing', 'Language', 'Landmark']
        
        options = {}
        for i, name in enumerate(feature_names):
            options[name] = list(categories[i])
            
        return options
    except Exception as e:
        print(f"Error extracting options: {e}")
        # Fallback if extraction fails
        return {
            "Food": ['Sushi', 'Pizza', 'Tacos', 'Injera', 'Curry', 'Burger', 'Croissant'],
            "Music": ['J-Pop', 'Opera', 'Mariachi', 'Ethio-Jazz', 'Bollywood', 'Pop', 'Chanson'],
            "Clothing": ['Kimono', 'Designer Suit', 'Poncho', 'Habesha Kemis', 'Saree', 'Jeans', 'Beret'],
            "Language": ['Japanese', 'Italian', 'Spanish', 'Amharic', 'Hindi', 'English', 'French'],
            "Landmark": ['Mt. Fuji', 'Colosseum', 'Chichen Itza', 'Lalibela', 'Taj Mahal', 'Statue of Liberty', 'Eiffel Tower']
        }
