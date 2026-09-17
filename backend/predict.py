import pandas as pd
import joblib
import os


# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "churn_model.pkl"
)


# ---------------- LOAD MODEL ----------------
model = joblib.load(MODEL_PATH)


# ---------------- PREDICT CHURN ----------------
def predict_churn(customer_data):

    df = pd.DataFrame([customer_data])

    # Get churn probability
    probability = model.predict_proba(df)[0][1] * 100

    # Get prediction
    prediction = model.predict(df)[0]

    # Risk segmentation
    if probability >= 70:
        risk = "High"
    elif probability >= 40:
        risk = "Medium"
    else:
        risk = "Low"

    # Return result
    return {
        "prediction": (
            "Likely to Churn"
            if prediction == 1
            else "Likely to Stay"
        ),
        "probability": round(probability, 2),
        "risk": risk
    }