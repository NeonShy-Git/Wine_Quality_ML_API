import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    model_filename: str = "hgb_wine_model.pkl"
    scaler_filename: str = "hgb_wine_scaler.pkl"
    
    models_dir: Path = BASE_DIR / "saved_models"
    
    uvicorn_host: str = "0.0.0.0"
    uvicorn_port: int = 8000

settings = Settings()