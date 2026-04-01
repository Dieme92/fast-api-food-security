import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le bundle (modèle + features)
bundle = joblib.load("modele_food_insecurity_best.pkl")
model = bundle["model"]
selected_features = bundle["features"]

# Charger ton dataset encodé avec la colonne insécurité_alimentaire déjà sauvegardée
data = pd.read_csv("data_encoded_1.csv")

# --- Sidebar pour exploration ---
st.sidebar.title("🔎 Exploration des données")

show_raw = st.sidebar.checkbox("📄 Données brutes")
show_corr = st.sidebar.checkbox("📊 Matrice de corrélation")

variables = st.sidebar.multiselect(
    "📈 Choisir une ou plusieurs variables :", 
    selected_features
)
show_curves = st.sidebar.button("Afficher les distributions")

show_importances = st.sidebar.button("🌟 Top 5 variables importantes (RandomForest)")

# --- Exploration ---
st.title("📊 Prédiction de l'insécurité alimentaire")

if show_raw:
    st.subheader("📄 Données brutes")
    st.write(data.head(20))  # affiche les 20 premières lignes

if show_corr:
    st.subheader("📊 Matrice de corrélation")
    corr = data[selected_features + ["insécurité_alimentaire"]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)

if show_curves and variables:
    st.subheader("📈 Distribution des variables choisies")
    n = len(variables)
    rows = (n // 2) + (n % 2)  # nombre de lignes nécessaires
    idx = 0

    for r in range(rows):
        cols = st.columns(2)  # deux colonnes par ligne
        for c in range(2):
            if idx < n:
                var = variables[idx]
                with cols[c]:
                    fig, ax = plt.subplots(figsize=(3.5, 3))  # petit format
                    sns.histplot(data[var], bins=10, kde=True, ax=ax)
                    ax.set_title(var, fontsize=9)
                    st.pyplot(fig)
                idx += 1

if show_importances:
    st.subheader("🌟 Top 5 variables importantes (RandomForest)")
    importances = model.feature_importances_
    top_idx = importances.argsort()[-5:]
    top_features = [selected_features[i] for i in top_idx]
    top_values = importances[top_idx]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=top_values, y=top_features, ax=ax, color="blue")
    ax.set_title("Top 5 Features - RandomForest")
    st.pyplot(fig)

# --- Section prédiction ---
st.header("🎯 Prédiction")

st.markdown("Entrez les fréquences observées sur les 7 derniers jours (0 = jamais, 1 = 1-2 jours, 2 = 3-4 jours, 3 = 5-7 jours) :")

# Champs de saisie
inputs = {}
for col in [
    "q600_inquiets_de_ne_pas_avoir_suffisamment_de_nourriture",
    "q601_ne_pas_manger_nourriture_saine_nutritive",
    "q602_manger_nourriture_peu_variee",
    "q603_sauter_un_repas",
    "q604_manger_moins_que_ce_que_vous_auriez_du",
    "q605_1_ne_plus_avoir_de_nourriture_pas_suffisamment_d_argent",
    "q606_1_avoir_faim_mais_ne_pas_manger",
    "q607_1_passer_toute_une_journee_sans_manger"
]:
    inputs[col] = st.number_input(col, min_value=0, max_value=3, step=1)

if st.button("Prédire"):
    # Vérifier si toutes les variables valent zéro
    if all(value == 0 for value in inputs.values()):
        st.success("Résultat : Sécurité alimentaire")
        st.info("📊 Probabilité modérée : 0.0")
        st.info("📊 Probabilité sévère : 0.0")
    else:
        # Ajouter colonnes manquantes avec valeur par défaut
        full_dict = {col: 0 for col in selected_features}
        full_dict.update(inputs)

        df = pd.DataFrame([full_dict])[selected_features]

        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0]

        mapping = {0: "Insécurité alimentaire modérée", 1: "Insécurité alimentaire sévère"}

        st.success(f"Résultat : {mapping[prediction]}")
        st.info(f"📊 Probabilité modérée : {round(float(proba[0]), 3)}")
        st.info(f"📊 Probabilité sévère : {round(float(proba[1]), 3)}")
