from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")


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