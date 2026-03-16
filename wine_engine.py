import joblib
import numpy as np
from pydantic import BaseModel, Field
from functools import lru_cache
from settings import settings

class WineRequest(BaseModel):
    fixed_acidity: float = Field(..., example=7.4)
    volatile_acidity: float = Field(..., example=0.7)
    citric_acid: float = Field(..., example=0.0)
    residual_sugar: float = Field(..., example=1.9)
    chlorides: float = Field(..., example=0.076)
    free_sulfur_dioxide: float = Field(..., example=11.0)
    total_sulfur_dioxide: float = Field(..., example=34.0)
    density: float = Field(..., example=0.9978)
    pH: float = Field(..., example=3.51)
    sulphates: float = Field(..., example=0.56)
    alcohol: float = Field(..., example=9.4)

class PredictionResponse(BaseModel):
    prediction: int
    quality_label: str
    confidence: float
    recommendation: str

@lru_cache(maxsize=1)
def load_model():
    path = settings.models_dir / settings.model_filename
    print(f"DEBUG: Intentando cargar modelo desde: {path}")
    return joblib.load(path)

@lru_cache(maxsize=1)
def load_scaler():
    path = settings.models_dir / settings.scaler_filename
    return joblib.load(path)

def predict_wine_quality(data: WineRequest) -> PredictionResponse:
    scaler = load_scaler()
    model = load_model()
    
    features = np.array([[ 
        data.fixed_acidity, data.volatile_acidity, data.citric_acid,
        data.residual_sugar, data.chlorides, data.free_sulfur_dioxide,
        data.total_sulfur_dioxide, data.density, data.pH,
        data.sulphates, data.alcohol
    ]])
    
    features_scaled = scaler.transform(features)
    
    prediction = int(model.predict(features_scaled)[0])
    probs = model.predict_proba(features_scaled)[0]
    confidence = float(np.max(probs)) * 100
    
    if prediction >= 7:
        label, rec = "Premium", "Calidad excepcional. Recomendado para reserva."
    elif prediction >= 5:
        label, rec = "Estándar", "Vino equilibrado. Apto para consumo masivo."
    else:
        label, rec = "Básica", "Calidad por debajo del promedio. Uso sugerido: cocina."
        
    return PredictionResponse(
        prediction=prediction,
        quality_label=label,
        confidence=round(confidence, 2),
        recommendation=rec
    )