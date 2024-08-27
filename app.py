from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory='templates')


# Load the model from disk
knn_model = pickle.load( open('knn_model.pkl','rb'))
scaling = pickle.load(open('scaling.pkl','rb'))


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/predict")
def predict(request: Request,
            age: int = Form(...),
            sex: int = Form(...),
            cp: int = Form(...),
            trestbps: int = Form(...),
            chol: int = Form(...),
            fbs: int = Form(...),
            restecg: int = Form(...),
            thalach: int = Form(...),
            exang: int = Form(...),
            oldpeak: float = Form(...),
            slope: int = Form(...),
            ca: int = Form(...),
            thal: int = Form(...)):
    data = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]   
    data = np.array(data).reshape(1, -1)
    scaled_data = scaling.transform(data) 
    prediction = knn_model.predict(scaled_data)
    return templates.TemplateResponse("home.html", {"request": request, "prediction_text": f"The heart disease is {prediction}"})


if __name__ == "__main__":
    uvicorn.run(app)






# from fastapi import FastAPI
# import uvicorn
# import pickle
# import numpy as np
# import pandas as pd
# from pydantic import BaseModel

# app = FastAPI()


# class input_var(BaseModel):
#     age: int
#     sex: int
#     cp: int
#     trestbps: int
#     chol: int    
#     fbs: int
#     restecg: int
#     thalach: int
#     exang: int
#     oldpeak: float
#     slope: int
#     ca: int
#     thal: int


# # Load the model from disk
# knn_model = pickle.load( open('knn_model.pkl','rb'))
# scaling = pickle.load(open('scaling.pkl','rb'))


# @app.get('/')
# def hello():
#     return {"message":"Hello"}

# @app.post('/predict')
# def prediction(data: input_var):
#     data = pd.json_normalize(data.model_dump())
#     data = pd.DataFrame(scaling.transform(data), columns = data.columns)
#     predicted = knn_model.predict(data)
#     return f"The heart disease prediction is: {predicted}"


# if __name__ == "__main__":
#     uvicorn.run(app)