import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib

###########################################################
# ✅ Chargement des modèles
###########################################################
@st.cache_resource
def load_models():
    rf_model = joblib.load("modele_food_insecurity.pkl")
    xgb_model = joblib.load("modele_food_insecurity_xgb1.pkl")
    return {"RandomForest": rf_model, "XGBoost": xgb_model}

models = load_models()
rf_model = models["RandomForest"]
xgb_model = models["XGBoost"]

###########################################################
# ✅ Chargement des données
###########################################################
@st.cache_data
def load_data():
    df = pd.read_csv("data_encoded_1.csv")
    return df

df = load_data()
df_sample = df.sample(100)

if st.sidebar.checkbox("Afficher les données brutes", False):
    st.subheader("Jeu de données 'data_encoded_1.csv' : Echantillon de 100 observateurs")
    st.write(df_sample)

st.title("📊 Analyse exploratoire du dataset")
st.write("Voici quelques statistiques descriptives sur les réponses des participants.")

st.subheader("📌 Statistiques descriptives")
st.dataframe(df.describe().round(2))

###########################################################
# ✅ Variables utilisées par les modèles
###########################################################
features_rf = list(rf_model.feature_names_in_)
features_xgb = list(xgb_model.feature_names_in_)

st.write("Colonnes attendues par RandomForest :", features_rf)
st.write("Colonnes attendues par XGBoost :", features_xgb)

###########################################################
# ✅ Fonction pour générer automatiquement le payload
###########################################################
def build_payload(user_inputs: dict, model_name="rf_model"):
    if model_name == "xgb_model":
        expected_features = features_xgb
    else:
        expected_features = features_rf

    payload = {col: user_inputs.get(col, 0) for col in expected_features}
    payload["modele"] = model_name
    return payload

###########################################################
# 🔹 Matrice de corrélation
###########################################################
st.subheader("📈 Matrice de corrélation des variables")
fig, ax = plt.subplots(figsize=(20, 10))
corr = df[features_rf].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

###########################################################
# 🔹 Histogrammes des variables
###########################################################
st.sidebar.subheader("📊 Sélection des variables à afficher")
vars_selectionnees = st.sidebar.multiselect("Choisissez les variables :", features_rf)
couleurs = sns.color_palette("husl", len(vars_selectionnees))

if vars_selectionnees:
    cols = st.columns(2)  # colonnes fixes
    for i, (var, couleur) in enumerate(zip(vars_selectionnees, couleurs)):
        col = cols[i % 2]
        with col:
            st.subheader(f"Histogramme : {var}")
            fig, ax = plt.subplots()
            sns.histplot(df[var], bins=10, kde=True, color=couleur, ax=ax)
            ax.set_title(f"Distribution de : {var}")
            st.pyplot(fig)

###########################################################
# 🔹 Formulaire de prédiction
###########################################################
st.title("🧠 Prédiction d'insécurité alimentaire")

modele_selectionne = st.radio("Choisissez le modèle :", ["rf_model", "xgb_model"])

user_inputs = {}
expected_features = features_xgb if modele_selectionne == "xgb_model" else features_rf

for col in expected_features:
    user_inputs[col] = st.number_input(f"{col}", min_value=0, max_value=10, value=0)

if st.button("🔍 Lancer la prédiction"):
    payload = build_payload(user_inputs, model_name=modele_selectionne)

    try:
        # ⚠️ Remplace par ton URL publique Render
        API_URL = "https://fast-api-food-security-dieme.onrender.com/predict"
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()

        niveau = result.get("niveau", "inconnu")
        score = result.get("score", 0.00)
        profil = result.get("profil", "inconnu")
        probabilites = result.get("probabilités", {})

        if niveau == "sévère":
            st.error("🔴 Niveau d'insécurité alimentaire : **sévère**")
        elif niveau == "modérée":
            st.warning("🟠 Niveau d'insécurité alimentaire : **modérée**")
        elif niveau == "aucune":
            st.success("🟢 Aucun signe d'insécurité alimentaire")
        else:
            st.info("ℹ️ Niveau inconnu")

        st.write("### 🔎 Score de risque")
        st.progress(score)
        st.write(f"Profil détecté : **{profil.capitalize()}**")
        st.write(f"Modèle utilisé : **{modele_selectionne}**")

        if probabilites:
            st.write("### 📊 Répartition des probabilités")
            fig, ax = plt.subplots()
            labels = ["Modérée", "Sévère"]
            sizes = [probabilites.get("classe_0", 0.0), probabilites.get("classe_1", 0.0)]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                   colors=['#4CAF50', '#FF9800'])
            ax.axis('equal')
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Erreur lors de la requête : {e}")
        if 'response' in locals():
            st.text(f"Réponse brute : {response.text}")
