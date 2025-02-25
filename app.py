from fastapi import FastAPI, HTTPException, Query
import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Charger le modèle
MODEL_PATH = "model.joblib"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle : {str(e)}")

# Initialisation de FastAPI
app = FastAPI()


class RetrainRequest(BaseModel):
    learning_rate: float
    n_estimators: int
    max_depth: int


# Route GET pour la prédiction avec des paramètres dans l'URL
@app.get("/predict")
async def predict_get(
    f1: float = Query(..., description="Feature 1"),
    f2: float = Query(..., description="Feature 2"),
    f3: float = Query(..., description="Feature 3"),
    f4: float = Query(..., description="Feature 4"),
    #    f5: float = Query(..., description="Feature 5"),
    #    f6: float = Query(..., description="Feature 6"),
    #    f7: float = Query(..., description="Feature 7"),
    #    f8: float = Query(..., description="Feature 8")
):
    try:
        # Transformer les données en tableau NumPy
        data = np.array([[f1, f2, f3, f4]])  # , f5, f6, f7, f8]])
        prediction = model.predict(data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Définition du format des données d'entrée
class PredictionInput(BaseModel):
    features: list[float]


# Route de prédiction
@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Transformer les données en tableau NumPy
        data = np.array([input_data.features])
        prediction = model.predict(data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Route pour réentrainer le modèle
@app.post("/retrain/")
def retrain(params: RetrainRequest):
    try:
        global model  # Pour mettre à jour le modèle global

        # Charger les données (exemple avec Iris)
        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )

        # Créer et configurer un nouveau modèle avec les paramètres fournis
        new_model = GradientBoostingClassifier(
            learning_rate=params.learning_rate,
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
        )

        # Réentrainer le modèle avec les données d'entraînement
        new_model.fit(X_train, y_train)

        # Sauvegarder le modèle réentrainé dans un fichier
        joblib.dump(new_model, MODEL_PATH)

        # Recharger le modèle après entrainement
        model = joblib.load(MODEL_PATH)

        # Retourner un message de succès
        return {"message": "Modèle réentrainé avec succès"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de réentrainement : {e}")
