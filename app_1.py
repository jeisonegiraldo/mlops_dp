from fastapi import FastAPI, file, uploadedfile,
from io import StringIO
import pandas as pd
from joblib import load


app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict/")
def predict(file: uploadedfile.UploadedFile = file(...)):
    
    # Load the pre-trained model
    classifier = load("linear_regression.joblib")

    
    features_df = pd.read_csv("selected_features.csv")
    features = features_df['0'].to_list()

    content = await file.read()

    df = pd.read_csv(StringIO(content.decode("utf-8")))
    data = df[features]

    predictions =classifier.predict(df)

    return {"predictions": predictions.tolist()
    }