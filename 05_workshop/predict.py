import pickle
from typing import Literal
from pydantic import BaseModel, Field
from pydantic import ConfigDict

from fastapi import FastAPI
import uvicorn



class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gender: Literal["male", "female"]
    seniorcitizen: Literal[0, 1]
    partner: Literal["yes", "no"]
    dependents: Literal["yes", "no"]
    phoneservice: Literal["yes", "no"]
    multiplelines: Literal["no", "yes", "no_phone_service"]
    internetservice: Literal["dsl", "fiber_optic", "no"]
    onlinesecurity: Literal["no", "yes", "no_internet_service"]
    onlinebackup: Literal["no", "yes", "no_internet_service"]
    deviceprotection: Literal["no", "yes", "no_internet_service"]
    techsupport: Literal["no", "yes", "no_internet_service"]
    streamingtv: Literal["no", "yes", "no_internet_service"]
    streamingmovies: Literal["no", "yes", "no_internet_service"]
    contract: Literal["month-to-month", "one_year", "two_year"]
    paperlessbilling: Literal["yes", "no"]
    paymentmethod: Literal[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)",
    ]
    tenure: int = Field(..., ge=0)
    monthlycharges: float = Field(..., ge=0.0)
    totalcharges: float = Field(..., ge=0.0)


class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool


app = FastAPI(title="customer-churn-prediction")

with open('model.bin', 'rb') as f_in:
    model_artifact = pickle.load(f_in)

# Support both legacy (dv, model) tuples and sklearn pipelines
if isinstance(model_artifact, tuple):
    dv, model = model_artifact
    pipeline = None
    artifact_description = "legacy artifact: (DictVectorizer, LogisticRegression)"
else:
    pipeline = model_artifact
    dv = model = None
    artifact_description = f"sklearn pipeline artifact: {pipeline.__class__.__name__}"

print(f"Startup -> {artifact_description}")


def predict_single(customer):
    print(f"Predict -> using {artifact_description}")
    if pipeline is not None:
        result = pipeline.predict_proba(customer)[0, 1]
    else:
        X = dv.transform(customer)
        result = model.predict_proba(X)[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    prob = predict_single(customer.model_dump())

    return PredictResponse(
        churn_probability=prob,
        churn=prob >= 0.5
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
