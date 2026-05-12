from fastapi import FastAPI
from pydantic import BaseModel, conint
import joblib
import pandas as pd

# Charger les bundles (modèle + features)
bundle_rf = joblib.load("modele_food_insecurity_best.pkl")   # RandomForest
bundle_xgb = joblib.load("modele_food_insecurity_xgb.pkl")   # XGBoost

models = {
    "rf": bundle_rf["model"],
    "xgb": bundle_xgb["model"]
}

features = bundle_rf["features"]  # même liste de variables pour les deux

# Définition du schéma d'entrée
class InputData(BaseModel):
    modele: str = "rf"  # choix par défaut : RandomForest
    q600_inquiets_de_ne_pas_avoir_suffisamment_de_nourriture: conint(ge=0, le=7)
    q601_ne_pas_manger_nourriture_saine_nutritive: conint(ge=0, le=7)
    q602_manger_nourriture_peu_variee: conint(ge=0, le=7)
    q603_sauter_un_repas: conint(ge=0, le=7)
    q604_manger_moins_que_ce_que_vous_auriez_du: conint(ge=0, le=7)
    q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent: conint(ge=0, le=7)
    q606_1_avoir_faim_mais_ne_pas_manger: conint(ge=0, le=7)
    q607_1_passer_toute_une_journee_sans_manger: conint(ge=0, le=7)

app = FastAPI()

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "features_count": len(features)
    }

@app.post("/predict")
def predict(data: InputData):
    input_dict = data.dict()

    # Sélection du modèle
    modele = input_dict.pop("modele")
    if modele not in models:
        return {"error": f"Modèle '{modele}' non reconnu. Choisissez 'rf' ou 'xgb'."}

    model = models[modele]

    # Construire le dictionnaire complet avec toutes les features
    full_dict = {col: 0 for col in features}
    full_dict.update(input_dict)

    # Construire le DataFrame avec les colonnes dans le bon ordre
    df = pd.DataFrame([full_dict])[features]

    # Prédiction
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0]

    mapping = {0: "Insécurité alimentaire modérée", 1: "Insécurité alimentaire sévère"}

    return {
        "modele_utilise": modele,
        "prediction": mapping[prediction],
        "probabilities": {
            "moderee": round(float(proba[0]), 3),
            "severe": round(float(proba[1]), 3)
        },
        "features_used": features
    }
