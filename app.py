import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import Field, computed_field, BaseModel
from typing import Literal, Annotated
import pickle
from joblib import load

# import model
with open('load_risk_model.pkl', 'rb') as f:
    model= pickle.load(f)
# model = load('model.pkl')


app= FastAPI()

# create pydantic model to velidate the data
class UserInput(BaseModel):
    age: Annotated[int, Field(..., gt=0, description='Current Age of the client')]
    income_lpa:  Annotated[float, Field(..., gt=0, description='Income of the client in one year')]
    credit_score : Annotated[int, Field(..., gt=0, description='Previous credit score of the client')]
    loan_amount: Annotated[float, Field(..., gt=0, description='Loan amount client want')]
    loan_tenure_months : Annotated[int, Field(..., gt=0,lt=100,  description='in how many month you want to back the loan')]
    employment_type : Annotated[Literal['unemployed', 'salaried', 'self-employed'], Field(..., description='Current employement status')]
    education_level : Annotated[Literal['highschool', 'postgraduate', 'graduate'], Field(..., description='Education level')]

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 24:
            return 'young'
        elif self.age < 40:
            return 'adult'
        elif self.age < 60:
            return 'Senior'
        else:
            return 'Old'
        
@app.post('/predict')
def predict_customer(data: UserInput):
    input_data = pd.DataFrame([{
        'income_lpa':data.income_lpa,
        'credit_score': data.credit_score,
        'loan_amount': data.loan_amount,
        'loan_tenure_months': data.loan_tenure_months,
        'employment_type' : data.employment_type,
        'education_level': data.education_level,
        'age_group' : data.age_group


    }])

    prediction= model.predict(input_data)[0]
    return JSONResponse(status_code=200, content={'prediction':prediction})