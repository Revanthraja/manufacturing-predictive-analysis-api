from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from pydantic import BaseModel
import pickle
import os

app = FastAPI()

DATA_FILE = "uploaded_data.csv"
MODEL_FILE = "model.pkl"
data = None
MODEL = None

class PredictionRequest(BaseModel):
    Temperature: float
    Run_Time: float
    Shift: str
    Machine_Type: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global data

    #if not file.filename.endswith("csv"):
     #   raise HTTPException(status_code=400, detail="Invalid file format. Only CSV files are supported.")

    data = pd.read_csv(file.file)
    required_columns = {"Machine_ID", "Temperature", "Run_Time", "Shift", "Machine_Type", "Downtime_Flag"}
    if not required_columns.issubset(data.columns):
        raise HTTPException(status_code=400, detail=f"Dataset must contain these columns: {', '.join(required_columns)}")

    data.to_csv(DATA_FILE, index=False)
    return {"message": "File uploaded successfully.", "rows": len(data)}

@app.post("/train")
def train_model():
    global MODEL

    if not os.path.exists(DATA_FILE):
        raise HTTPException(status_code=400, detail="No data uploaded. Please upload data first.")

    data = pd.read_csv(DATA_FILE)
    categorical_cols = ["Shift", "Machine_Type"]
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    X = data.drop(["Machine_ID", "Downtime_Flag"], axis=1)
    y = data["Downtime_Flag"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    MODEL = LogisticRegression()
    MODEL.fit(X_train, y_train)

    y_pred = MODEL.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(MODEL, f)

    return {"accuracy": accuracy, "f1_score": f1}

@app.post("/predict")
def predict(request: PredictionRequest):
    if not os.path.exists(MODEL_FILE):
        raise HTTPException(status_code=400, detail="Model is not trained. Train the model first.")
    
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    
    # Ensure all features are present
    input_data = pd.DataFrame([{
        "Temperature": request.Temperature,
        "Run_Time": request.Run_Time,
        "Shift_Morning": 1 if request.Shift == "Morning" else 0,
        "Shift_Evening": 1 if request.Shift == "Evening" else 0,
        "Shift_Night": 1 if request.Shift == "Night" else 0,
        "Machine_Type_Type_A": 1 if request.Machine_Type == "Type_A" else 0,
        "Machine_Type_Type_B": 1 if request.Machine_Type == "Type_B" else 0,
        "Machine_Type_Type_C": 1 if request.Machine_Type == "Type_C" else 0,
    }])
    
    # Ensure input_data matches training features
    expected_features = model.feature_names_in_
    missing_features = set(expected_features) - set(input_data.columns)
    for feature in missing_features:
        input_data[feature] = 0  # Add missing features with a default value
    
    input_data = input_data[expected_features]  # Reorder columns to match training order
    
    prediction = model.predict(input_data)[0]
    confidence = max(model.predict_proba(input_data)[0])
    
    return {"Downtime": "Yes" if prediction == 1 else "No", "Confidence": round(confidence, 2)}

# Run using `uvicorn api:app --reload`
