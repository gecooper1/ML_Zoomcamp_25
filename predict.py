# importing packages
import pickle
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd

from fastapi import FastAPI
import uvicorn

# defining our expected input and output as Pydantic classes
class Applicant(BaseModel):
    model_config = ConfigDict(extra='forbid')
    age: int = Field(..., ge=0)
    income: int = Field(..., ge=0)
    loan_amount: int = Field(..., ge=0)
    credit_score: int = Field(..., ge=0)
    months_employed: int = Field(..., ge=0)
    num_credit_lines: int = Field(..., ge=0)
    interest_rate: float = Field(..., ge=0.0)
    dti_ratio: float = Field(..., ge=0.0, le=1.0)

class PredictResponse(BaseModel):
    default_probability: float
    default: bool

# loading the trained model
with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# function to generate a single default prediction for an applicant
def predict_single(applicant):
    
    # converting the input JSON to a pandas dataframe
    applicant_df = pd.DataFrame([applicant])

    result = pipeline.predict_proba(applicant_df)[0, 1]
    return float(result)

# defining our predict app
app = FastAPI(title="loan-default-prediction")

# generating a prediction response for the predict app
@app.post("/predict")
def predict(applicant: Applicant) -> PredictResponse:
    prob = predict_single(applicant.model_dump())

    return PredictResponse(
        default_probability = prob,
        default = bool(prob >= 0.5)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)