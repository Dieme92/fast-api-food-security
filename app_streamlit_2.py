import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib

@st.cache_resource
def load_models():
    rf_bundle = joblib.load("modele_food_insecurity_rf.pkl")
    xgb_bundle = joblib.load("modele_food_insecurity_xgb.pkl")
    return {
        "RandomForest": rf_bundle["model"],
        "XGBoost": xgb_bundle["model"],
        "features": rf_bundle["features"]
    }

models = load_models()
rf_model = models["RandomForest"]
xgb_model = models["XGBoost"]
selected_features = models["features"]

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
st.subheader("📌 Statistiques descriptives")
st.dataframe(df.describe().round(2))

st.write("Colonnes utilisées par les modèles :", selected_features)

def build_payload(user_inputs: dict, model_name="rf_model"):
    payload = {col: user_inputs.get(col, 0) for col in selected_features}
    payload["modele"] = model_name
    return payload

st.subheader("📈 Matrice de corrélation des variables")
fig, ax = plt.subplots(figsize=(20, 10))
corr = df[selected_features].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.sidebar.subheader("📊 Sélection des variables à afficher")
vars_selectionnees = st.sidebar.multiselect("Choisissez les variables :", selected_features)
couleurs = sns.color_palette("husl", len(vars_selectionnees))

if vars_selectionnees:
    cols = st.columns(2)
    for i, (var, couleur) in enumerate(zip(vars_selectionnees, couleurs)):
        with cols[i % 2]:
            st.subheader(f"Histogramme : {var}")
            fig, ax = plt.subplots()
            sns.histplot(df[var], bins=10, kde=True, color=couleur, ax=ax)
            ax.set_title(f"Distribution de : {var}")
            st.pyplot(fig)

st.title("🧠 Prédiction d'insécurité alimentaire")
modele_selectionne = st.radio("Choisissez le modèle :", ["rf_model", "xgb_model"])

user_inputs = {col: st.number_input(f"{col}", min_value=0, max_value=10, value=0) for col in selected_features}

if st.button("🔍 Lancer la prédiction"):
    payload = build_payload(user_inputs, model_name=modele_selectionne)
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
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
