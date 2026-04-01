import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI(title="Bitcoin Fraud Detection API")

# Define Core Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(filename, subfolders):
    """Search for a file in root or specific subfolders."""
    root_path = os.path.join(ROOT_DIR, filename)
    if os.path.exists(root_path): return root_path
    for folder in subfolders:
        layered_path = os.path.join(ROOT_DIR, *folder.split('/'), filename)
        if os.path.exists(layered_path): return layered_path
    return None

# Resolve and Load Model Assets
MODEL_PATH = find_file("base_svm.joblib", ["results/models", "result/models"])
FEAT_CSV_PATH = find_file("selected_feature_names.csv", ["results/checkpoints", "result/checkpoints"])

if not MODEL_PATH or not FEAT_CSV_PATH:
    raise RuntimeError("Could not find base_svm.joblib or selected_feature_names.csv")

model = joblib.load(MODEL_PATH)
feat_df = pd.read_csv(FEAT_CSV_PATH)
required_features = feat_df["feature"].tolist()

# Define Request Scema
class TransactionData(BaseModel):
    features: dict # A dictionary of all 74 feature values

@app.get("/")
def home():
    return {"status": "online", "model": "SVM", "features_required": len(required_features)}

@app.post("/predict")
def predict(data: TransactionData):
    try:
        # Convert dictionary to DataFrame
        df = pd.DataFrame([data.features])
        
        # Validate features
        missing = [f for f in required_features if f not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing[:5]}...")
            
        # Reorder and predict
        X = df[required_features].values
        proba = model.predict_proba(X)[0, 1]
        prediction = 1 if proba >= 0.17 else 0
        
        return {
            "probability": round(float(proba), 4),
            "prediction": int(prediction),
            "risk_level": "🚨 HIGH RISK" if prediction == 1 else "✅ LOW RISK"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
