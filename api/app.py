import os
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.management_companyAPI import router as create_company_router, load_all_company_routes
from models.predict import ModelPredictor
from models.translator import AutoMultiLangTranslator

BASE_DIR = os.path.dirname(__file__)

app = FastAPI(title="Dementia Forecast API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

app.include_router(create_company_router)
load_all_company_routes(app)

@app.get("/")
def read_root():
    return {"message": "Dementia Forecast API is running"}

predictor = ModelPredictor()

class PredictionInputRequest(BaseModel):
    question_list: list
    biometric_data_list: list

class PredictionResponse(BaseModel):
    statusCode: int
    message: str
    risk_score: int

@app.post("/prediction/input", response_model=PredictionResponse)
def prediction_input(request: PredictionInputRequest):
    try:
        data = {
            "question_list": request.question_list,
            "biometric_data_list": request.biometric_data_list
        }
        prediction_result = predictor.predict(data)
        return {
            "statusCode": 200,
            "message": "예측 입력값 전송 성공",
            "risk_score": prediction_result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail="예측 입력값 전송 실패")

@app.post("/prediction/result", response_model=PredictionResponse)
def prediction_result(request: PredictionInputRequest):
    try:
        data = {
            "question_list": request.question_list,
            "biometric_data_list": request.biometric_data_list
        }
        prediction_result = predictor.predict(data)
        return {
            "statusCode": 200,
            "message": "예측 결과 전송 성공",
            "risk_score": prediction_result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail="예측 결과 전송 실패")
