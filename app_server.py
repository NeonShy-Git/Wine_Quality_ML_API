import uvicorn
from fastapi import FastAPI
from wine_engine import WineRequest, PredictionResponse, predict_wine_quality, load_model
from settings import settings

app = FastAPI(
    title="Wine Intelligence API",
    description="API avanzada para la predicción de calidad de vinos usando Gradient Boosting.",
    version="2.0.0"
)

@app.on_event("startup")
async def startup():
    load_model()

@app.get("/")
async def health_check():
    return {"status": "online", "model": "HistGradientBoosting"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: WineRequest):
    return predict_wine_quality(request)

if __name__ == "__main__":
    uvicorn.run(
        "app_server:app", 
        host=settings.uvicorn_host, 
        port=settings.uvicorn_port, 
        reload=True
    )