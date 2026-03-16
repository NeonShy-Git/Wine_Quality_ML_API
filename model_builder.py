import joblib
import os
from pathlib import Path
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from settings import settings

def train():
    print("🚀 Iniciando descarga de datos de UCI...")
    wine = fetch_ucirepo(id=186) 
    X = wine.data.features
    y = wine.data.targets.values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("🧠 Entrenando HistGradientBoostingClassifier...")
    model = HistGradientBoostingClassifier(
        max_iter=100, 
        learning_rate=0.1, 
        max_depth=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    models_dir = Path(settings.models_dir)
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(scaler, models_dir / settings.scaler_filename)
    joblib.dump(model, models_dir / settings.model_filename)
    
    print(f"✅ Artefactos guardados exitosamente en {models_dir}")

if __name__ == "__main__":
    train()