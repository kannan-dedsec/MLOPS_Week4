import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
import os

MODEL_PATH = "model/model.joblib"
DATA_PATH = "data/data.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("species", axis=1)
    y = df["species"]
    return X, y

def test_model_exists():
    assert os.path.exists(MODEL_PATH), f"{MODEL_PATH} not found"

def test_model_accuracy_threshold():
    model = joblib.load(MODEL_PATH)
    X, y = load_data()
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    assert acc >= 0.99, f"Model accuracy too low: {acc}"
