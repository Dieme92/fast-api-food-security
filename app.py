import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import uvicorn

# ✅ Initialisation de l'application
app = FastAPI()

# ✅ Charger les modèles avec un chemin relatif
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    rf_model = joblib.load(os.path.join(BASE_DIR, "modele_food_insecurity.pkl"))
    xgb_model = joblib.load(os.path.join(BASE_DIR, "modele_food_insecurity_xgb1.pkl"))
except Exception as e:
    print("Erreur lors du chargement des modèles :", e)
    rf_model, xgb_model = None, None

# ✅ Fonction robuste pour récupérer les colonnes
def get_feature_names(model, fallback=None):
    if model is None:
        return []
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return fallback if fallback else []

# ✅ Colonnes attendues par défaut
default_features = [
    "q606_1_avoir_faim_mais_ne_pas_manger",
    "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent",
    "q604_manger_moins_que_ce_que_vous_auriez_du",
    "q603_sauter_un_repas",
    "q601_ne_pas_manger_nourriture_saine_nutritive"
]

features_rf = get_feature_names(rf_model, default_features)
features_xgb = get_feature_names(xgb_model, default_features)

# ✅ Schéma d'entrée
class InputData(BaseModel):
    q606_1_avoir_faim_mais_ne_pas_manger: int
    q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent: int
    q604_manger_moins_que_ce_que_vous_auriez_du: int
    q603_sauter_un_repas: int
    q601_ne_pas_manger_nourriture_saine_nutritive: int
    modele: str = "rf_model"   # valeur par défaut ("rf_model" ou "xgb_model")

# ✅ Endpoint de santé
@app.get("/health")
def health_check():
    return {"status": "API opérationnelle ✅"}

# ✅ Endpoint de prédiction
@app.post("/predict")
def predict(data: InputData):
    try:
        if rf_model is None or xgb_model is None:
            raise RuntimeError("Les modèles n'ont pas été chargés correctement.")

        input_df = pd.DataFrame([data.dict()])

        # Choisir le modèle
        if data.modele == "xgb_model":
            model = xgb_model
            expected_features = features_xgb
        else:
            model = rf_model
            expected_features = features_rf

        # Vérification des colonnes
        if not all(col in input_df.columns for col in expected_features):
            raise ValueError(
                f"Les colonnes envoyées ne correspondent pas au modèle {data.modele}.\n"
                f"Attendu : {expected_features}\n"
                f"Reçu : {list(input_df.columns)}"
            )

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

# ✅ Bloc pour lancer l'application en local ou sur Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render fournit PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
