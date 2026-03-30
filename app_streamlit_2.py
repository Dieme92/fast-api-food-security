import streamlit as st
import joblib
import pandas as pd

# Charger le bundle (modèle + features)
bundle = joblib.load("modele_food_insecurity_best.pkl")
model = bundle["model"]
selected_features = bundle["features"]

st.title("📊 Prédiction de l'insécurité alimentaire modérée  ou sévère ")

st.markdown("Entrez les fréquences observées sur les 7 derniers jours :")

# Formulaire Streamlit
q600 = st.slider("q600 - Inquiets de ne pas avoir suffisamment de nourriture", 0, 3, 0)
q601 = st.slider("q601 - Ne pas manger nourriture saine/nutritive", 0, 3, 0)
q602 = st.slider("q602 - Manger nourriture peu variée", 0, 3, 0)
q603 = st.slider("q603 - Sauter un repas", 0, 3, 0)
q604 = st.slider("q604 - Manger moins que ce que vous auriez dû", 0, 3, 0)
q605 = st.slider("q605 - Ne plus avoir de nourriture faute d'argent", 0, 3, 0)
q606 = st.slider("q606 - Avoir faim mais ne pas manger", 0, 3, 0)
q607 = st.slider("q607 - Passer toute une journée sans manger", 0, 3, 0)

if st.button("Prédire"):
    # Construire dictionnaire avec valeurs saisies
    input_dict = {
        "q600_inquiets_de_ne_pas_avoir_suffisamment_de_nourriture": q600,
        "q601_ne_pas_manger_nourriture_saine_nutritive": q601,
        "q602_manger_nourriture_peu_variee": q602,
        "q603_sauter_un_repas": q603,
        "q604_manger_moins_que_ce_que_vous_auriez_du": q604,
        "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent": q605,
        "q606_1_avoir_faim_mais_ne_pas_manger": q606,
        "q607_1_passer_toute_une_journee_sans_manger": q607
    }

    # Ajouter colonnes manquantes avec valeur par défaut
    full_dict = {col: 0 for col in selected_features}
    full_dict.update(input_dict)

    # Construire DataFrame avec colonnes dans le bon ordre
    df = pd.DataFrame([full_dict])[selected_features]

    # Prédiction
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    mapping = {
        0: "Insécurité alimentaire modérée",
        1: "Insécurité alimentaire sévère"
    }

    st.success(f"🎯 Résultat : {mapping[prediction]}")
    st.info(f"📊 Probabilité associée : {round(float(proba), 3)}")
