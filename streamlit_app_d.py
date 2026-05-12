import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

# Charger les bundles (modèle + features)
bundle_rf = joblib.load("modele_food_insecurity_best.pkl")
bundle_xgb = joblib.load("modele_food_insecurity_xgb.pkl")

models = {
    "RandomForest": bundle_rf["model"],
    "XGBoost": bundle_xgb["model"]
}
selected_features = bundle_rf["features"]

# Charger dataset encodé
data = pd.read_csv("data_encoded_1.csv")
y = data["insécurité_alimentaire"].replace({1:0, 2:1}).astype(int)
X = data[selected_features]

# Séparer train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Sidebar exploration ---
st.sidebar.title("🔎 Exploration des données")
show_raw = st.sidebar.checkbox("📄 Données brutes")
show_corr = st.sidebar.checkbox("📊 Matrice de corrélation")
variables = st.sidebar.multiselect("📈 Choisir les variables :", selected_features)
show_curves = st.sidebar.button("Afficher les distributions")
show_importances = st.sidebar.button("🌟 Top 5 des variables importantes (RandomForest)")

# --- Exploration ---
st.title("📊 Projet de prédiction de l'insécurité alimentaire modérée ou sévère")

if show_raw:
    st.subheader("📄 Données brutes")
    st.write(data.head(20))

if show_corr:
    st.subheader("📊 Matrice de corrélation")
    corr = data[selected_features + ["insécurité_alimentaire"]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig)

if show_curves and variables:
    st.subheader("📈 Distribution des variables choisies")
    n = len(variables)
    rows = (n // 2) + (n % 2)
    idx = 0
    for r in range(rows):
        cols = st.columns(2)
        for c in range(2):
            if idx < n:
                var = variables[idx]
                with cols[c]:
                    fig, ax = plt.subplots(figsize=(3.5, 3))
                    sns.histplot(data[var], bins=10, kde=True, ax=ax)
                    ax.set_title(var, fontsize=9)
                    st.pyplot(fig)
                idx += 1

if show_importances:
    st.subheader("🌟 Top 5 variables importantes (RandomForest)")
    importances = models["RandomForest"].feature_importances_
    top_idx = importances.argsort()[-5:]
    top_features = [selected_features[i] for i in top_idx]
    top_values = importances[top_idx]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=top_values, y=top_features, ax=ax, color="blue")
    ax.set_title("Top 5 Features - RandomForest")
    st.pyplot(fig)

# --- Comparaison automatique ---
st.header("📊 Comparaison RandomForest vs XGBoost (train vs test)")

results = []
fig_cm, axes_cm = plt.subplots(2, 2, figsize=(10, 8))
fig_roc, ax_roc = plt.subplots(figsize=(6, 4))

for j, (name, model) in enumerate(models.items()):
    y_pred_train = model.predict(X_train)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    cm_train = confusion_matrix(y_train, y_pred_train, labels=[0,1])
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=axes_cm[0, j],
                xticklabels=["Modérée", "Sévère"], yticklabels=["Modérée", "Sévère"])
    axes_cm[0, j].set_title(f"{name} - Train")

    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    cm_test = confusion_matrix(y_test, y_pred_test, labels=[0,1])
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Reds", ax=axes_cm[1, j],
                xticklabels=["Modérée", "Sévère"], yticklabels=["Modérée", "Sévère"])
    axes_cm[1, j].set_title(f"{name} - Test")

    fpr, tpr, _ = roc_curve(y_test, y_proba_test, pos_label=1)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, lw=2, label=f"{name} (AUC test={roc_auc:.3f})")

    results.append({
        "Modèle": name,
        "Accuracy (test)": accuracy_score(y_test, y_pred_test),
        "Recall (Sévère test)": recall_score(y_test, y_pred_test, pos_label=1),
        "Précision (Sévère test)": precision_score(y_test, y_pred_test, pos_label=1),
        "F1-score (Sévère test)": f1_score(y_test, y_pred_test, pos_label=1),
        "AUC (test)": roc_auc
    })

results_df = pd.DataFrame(results)
st.dataframe(results_df, use_container_width=True)

st.subheader("Matrice de confusion (train vs test)")
st.pyplot(fig_cm)

ax_roc.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
ax_roc.set_title("Courbes ROC comparées (jeu de test)")
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)

best_model_row = results_df.loc[results_df["AUC (test)"].idxmax()]
best_model_name = best_model_row["Modèle"]
st.success(f"🏆 Meilleur modèle : {best_model_name} (AUC test = {best_model_row['AUC (test)']:.4f})")

# --- Section prédiction ---
st.header("🎯 Prédiction de l'insécurité alimentaire")
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
    full_dict = {col: 0 for col in selected_features}
    full_dict.update(inputs)
    df = pd.DataFrame([full_dict])[selected_features]

    # Règle métier : si toutes les variables sont à 0
    if df.sum(axis=1).iloc[0] == 0:
        st.success("Résultat : Pas d'insécurité alimentaire")
    else:
        model = models[best_model_name]
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0]

        mapping = {0: "Insécurité alimentaire modérée", 1: "Insécurité alimentaire sévère"}

        st.success(f"Résultat ({best_model_name}) : {mapping[prediction]}")
        st.info(f"📊 Probabilité modérée : {round(float(proba[0]), 3)}")
        st.info(f"📊 Probabilité sévère : {round(float(proba[1]), 3)}")
