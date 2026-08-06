from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import joblib

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load ML artifacts once when the application starts
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/predict")
def predict(
    request: Request,
    senior_citizen: str = Form(...),
    partner: str = Form(...),
    tenure: int = Form(...)
):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "senior": senior_citizen,
            "partner": partner,
            "tenure": tenure
        }
    )