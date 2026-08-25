from typing import Any, Dict, Literal

import joblib
import pandas as pd

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    FastAPI = None
    BaseModel = object


MODEL_FILE = "loan_rbf_model.pkl"
MODEL_PATH = r"C:\Users\ליאור\loan_rbf_model.pkl"
MODEL_FEATURES = [
    "Person Income",
    "Credit History",
    "Credit Score",
    "Loan percentage",
    "Home Onwership_OTHER",
    "Home Onwership_OWN",
    "Home Onwership_RENT",
    "Previous Loan_Yes",
]


def load_model():
    """Load the saved trained model from its explicit file path."""
    return joblib.load(MODEL_PATH)


def encode_features_from_user_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert raw user input into the exact feature order used by the RBF model.
    """
    home_ownership = payload.get("home_ownership", "OTHER")
    previous_loan = payload.get("previous_loan", "No")

    encoded = {
        "Person Income": float(payload["person_income"]),
        "Credit History": float(payload["credit_history"]),
        "Credit Score": float(payload["credit_score"]),
        "Loan percentage": float(payload["loan_percentage"]),
        "Home Onwership_OTHER": 1.0 if home_ownership == "OTHER" else 0.0,
        "Home Onwership_OWN": 1.0 if home_ownership == "OWN" else 0.0,
        "Home Onwership_RENT": 1.0 if home_ownership == "RENT" else 0.0,
        "Previous Loan_Yes": 1.0 if previous_loan == "Yes" else 0.0,
    }

    return {key: encoded[key] for key in MODEL_FEATURES}


def predict_loan(features):
    """
    Predict loan eligibility from either raw input or already encoded feature values.
    The features must match the columns after get_dummies.
    """
    model = load_model()

    if isinstance(features, dict):
        if all(key in features for key in MODEL_FEATURES):
            encoded = {key: features[key] for key in MODEL_FEATURES}
        else:
            encoded = encode_features_from_user_input(features)
        X = pd.DataFrame([encoded], columns=MODEL_FEATURES)

    elif isinstance(features, list):
        X = pd.DataFrame(features, columns=MODEL_FEATURES)

    elif isinstance(features, pd.DataFrame):
        X = features.copy()

    else:
        raise TypeError("features must be a dict, list, or pandas DataFrame")

    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    return model.predict(X)


if FastAPI is not None:
    class LoanRequest(BaseModel):
        person_income: float
        home_ownership: Literal["RENT", "OWN", "MORTGAGE", "OTHER"]
        credit_history: float
        credit_score: int
        loan_percentage: float
        previous_loan: Literal["Yes", "No"]

    MODEL_INFO = {
        "model_name": "Support Vector Classifier (SVC)",
        "kernel": "rbf",
        "model_file": MODEL_FILE,
        "pipeline": ["ColumnTransformer(scale_num_only)", "SVC(kernel='rbf')"],
        "accuracy": 0.858,
        "margin": 0.058541571844188206,
        "training_features": [
            "Person Income",
            "Home Onwership",
            "Credit History",
            "Credit Score",
            "Loan percentage",
            "Previous Loan",
        ],
        "features": MODEL_FEATURES,
        "feature_order_after_get_dummies": MODEL_FEATURES,
    }

    app = FastAPI(title="Loan Eligibility API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    @app.get("/model-info")
    def model_info():
        return MODEL_INFO

    @app.post("/predict")
    def predict_loan_api(payload: LoanRequest):
        encoded_features = encode_features_from_user_input(payload.model_dump())
        prediction = predict_loan(encoded_features)

        value = prediction[0] if hasattr(prediction, "__len__") and len(prediction) > 0 else prediction
        is_eligible = bool(value)

        return {
            "input": payload.model_dump(),
            "encoded_features": encoded_features,
            "feature_order": MODEL_FEATURES,
            "prediction": prediction.tolist() if hasattr(prediction, "tolist") else prediction,
            "eligible": is_eligible,
        }

    if __name__ == "__main__":
        import uvicorn

        uvicorn.run("main_2:app", host="0.0.0.0", port=8000, reload=True)
