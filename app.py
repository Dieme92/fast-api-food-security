import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    # Charger les modèles sauvegardés en dictionnaire
    rf_bundle = joblib.load(os.path.join(BASE_DIR, "modele_food_insecurity_rf.pkl"))
    xgb_bundle = joblib.load(os.path.join(BASE_DIR, "modele_food_insecurity_xgb.pkl"))

    rf_model = rf_bundle["model"]
    xgb_model = xgb_bundle["model"]
    features_rf = rf_bundle["features"]
    features_xgb = xgb_bundle["features"]

except Exception as e:
    print("Erreur lors du chargement des modèles :", e)
    rf_model, xgb_model = None, None
    features_rf, features_xgb = [], []

# Schéma d'entrée basé sur les colonnes sauvegardées
class InputData(BaseModel):
    q606_1_avoir_faim_mais_ne_pas_manger: int
    q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent: int
    q604_manger_moins_que_ce_que_vous_auriez_du: int
    q603_sauter_un_repas: int
    q601_ne_pas_manger_nourriture_saine_nutritive: int
    modele: str = "rf_model"

@app.get("/health")
def health_check():
    return {"status": "API opérationnelle ✅"}

@app.post("/predict")
def predict(data: InputData):
    try:
        if rf_model is None or xgb_model is None:
            raise RuntimeError("Les modèles n'ont pas été chargés correctement.")

        input_df = pd.DataFrame([data.dict()])

        if data.modele == "xgb_model":
            model = xgb_model
            expected_features = features_xgb
        else:
            model = rf_model
            expected_features = features_rf

        # Prédiction
        proba = model.predict_proba(input_df[expected_features])[0]
        seuil_severe = 0.4
        prediction_binaire = int(proba[1] > seuil_severe)

        if input_df[expected_features].sum().sum() == 0:
            niveau = "aucune"
            profil = "neutre"
        else:
            niveau = "sévère" if prediction_binaire == 1 else "modérée"
            profil = "critique" if prediction_binaire == 1 else "intermédiaire"

        return JSONResponse(content={
            "prediction": prediction_binaire,
            "niveau": niveau,
            "profil": profil,
            "score": round(float(proba[1]), 4),
            "probabilités": {
                "classe_0": round(float(proba[0]), 4),
                "classe_1": round(float(proba[1]), 4)
            },
            "modele_utilisé": data.modele
        })

    except Exception as e:
        return JSONResponse(content={
            "error": "Une erreur est survenue",
            "details": str(e)
        }, status_code=500)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
