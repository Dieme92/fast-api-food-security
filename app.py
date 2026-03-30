from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Charger le bundle (modèle + features)
bundle = joblib.load("modele_food_insecurity_best.pkl")
model = bundle["model"]
selected_features = bundle["features"]

# Définition du schéma d'entrée
class InputData(BaseModel):
    q600_inquiets_de_ne_pas_avoir_suffisamment_de_nourriture: int
    q601_ne_pas_manger_nourriture_saine_nutritive: int
    q602_manger_nourriture_peu_variee: int
    q603_sauter_un_repas: int
    q604_manger_moins_que_ce_que_vous_auriez_du: int
    q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent: int
    q606_1_avoir_faim_mais_ne_pas_manger: int
    q607_1_passer_toute_une_journee_sans_manger: int

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    input_dict = data.dict()

    # Ajouter toutes les colonnes attendues par le modèle avec valeur par défaut = 0
    full_dict = {col: 0 for col in selected_features}
    full_dict.update(input_dict)

    # Construire le DataFrame avec les colonnes dans le bon ordre
    df = pd.DataFrame([full_dict])[selected_features]

    # Prédiction
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0]

    mapping = {0: "Insécurité alimentaire modérée", 1: "Insécurité alimentaire sévère"}

    return {
        "prediction": mapping[prediction],
        "probability_moderee": round(float(proba[0]), 3),
        "probability_severe": round(float(proba[1]), 3)
    }
