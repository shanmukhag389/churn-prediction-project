from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import joblib
import pandas as pd


# ---------------------------------
# Initialize FastAPI
# ---------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using Machine Learning",
    version="1.0"
)


# ---------------------------------
# Load ML artifacts
# ---------------------------------

model = joblib.load("model.pkl")

scaler = joblib.load("scaler.pkl")

feature_columns = joblib.load("feature_columns.pkl")


# ---------------------------------
# Frontend setup
# ---------------------------------

templates = Jinja2Templates(
    directory="templates"
)


app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ---------------------------------
# Home page
# ---------------------------------

@app.get("/")
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# ---------------------------------
# Prediction route
# ---------------------------------

@app.post("/predict")
def predict(

    request: Request,

    senior_citizen: int = Form(...),

    partner: str = Form(...),

    dependents: str = Form(...),

    tenure_months: int = Form(...),

    phone_service: str = Form(...),

    paperless_billing: str = Form(...),

    monthly_charges: float = Form(...),

    total_charges: float = Form(...),

    multiple_lines: str = Form(...),

    internet_service: str = Form(...),

    online_security: str = Form(...),

    online_backup: str = Form(...),

    device_protection: str = Form(...),

    tech_support: str = Form(...),

    streaming_tv: str = Form(...),

    streaming_movies: str = Form(...),

    contract: str = Form(...),

    payment_method: str = Form(...)

):


    # ---------------------------------
    # Create input dataframe
    # ---------------------------------

    input_data = {


        "Senior Citizen": senior_citizen,

        "Partner": partner,

        "Dependents": dependents,

        "Tenure Months": tenure_months,

        "Phone Service": phone_service,

        "Paperless Billing": paperless_billing,

        "Monthly Charges": monthly_charges,

        "Total Charges": total_charges,

        "Multiple Lines": multiple_lines,

        "Internet Service": internet_service,

        "Online Security": online_security,

        "Online Backup": online_backup,

        "Device Protection": device_protection,

        "Tech Support": tech_support,

        "Streaming TV": streaming_tv,

        "Streaming Movies": streaming_movies,

        "Contract": contract,

        "Payment Method": payment_method

    }


    input_df = pd.DataFrame([input_data])


    # ---------------------------------
    # One hot encoding
    # ---------------------------------

    input_df = pd.get_dummies(input_df)



    # ---------------------------------
    # Match training columns
    # ---------------------------------

    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    print("Input columns:", len(input_df.columns))
    print(input_df.columns)

    # ---------------------------------
    # Scaling
    # ---------------------------------


    # ---------------------------------
    # Prediction
    # ---------------------------------

    prediction = model.predict(
        input_df
    )


    if prediction[0] == 1:

        result = "Customer will churn"

    else:

        result = "Customer will stay"



    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": result
        }
    )