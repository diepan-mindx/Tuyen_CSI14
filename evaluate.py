import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def evaluate_model():
    # load model
    data = joblib.load("heart_model.h5")
    model = data["model"]
    imputer = data["imputer"]
    scaler = data["scaler"]

    # load test data
    df = pd.read_csv("test_data.csv")
    X = df.drop("num", axis=1)
    y = df["num"]
  
    # preprocess test data
    X = pd.get_dummies(X, drop_first=True)
    
    X = imputer.transform(X)
    X = scaler.transform(X)
    
    # make predictions
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    return acc