import pandas as pd
import numpy as np
import streamlit as st
import joblib

data = pd.read_csv("Expresso_churn_dataset.csv")

print(data.head())

print(data.info())

# Générer un rapport de profilage pour explorer les données plus en détail
from pandas_profiling import ProfileReport

profile = ProfileReport(data)
profile.to_file("expresso_churn_data_report.html")

# Diviser les données en fonctionnalités et étiquettes
X = data.drop("CHURN", axis=1)  # Remplacer "target_column" par le nom de votre colonne cible
y = data["CHURN"]

# Diviser les données en ensembles d'entraînement et de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un classificateur (par exemple, RandomForestClassifier)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = classifier.predict(X_test)

# Évaluer les performances du modèle
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Sauvegarder le modèle entraîné dans un fichier
joblib.dump(classifier, "modele_ml.pkl")

# Charger le modèle entraîné
model = joblib.load("modele_ml.pkl") 
# Fonction pour faire des prédictions
def predict(data):
    prediction = model.predict(data)
    return prediction

# Interface utilisateur Streamlit
def main():
    st.title("Application de prédiction")

    # Ajouter des champs de saisie pour les fonctionnalités
    feature1 = st.number_input("MONTANT")
    feature2 = st.number_input("FREQUENCE_RECH")
    # Ajouter d'autres champs de saisie pour les autres fonctionnalités

    # Bouton de prédiction
    if st.button("Faire la prédiction"):
        # Rassembler les données d'entrée dans un DataFrame
        input_data = pd.DataFrame([[feature1, feature2]], columns=["MONTANT", "FREQUENCE_RECH"])  # Ajouter d'autres colonnes si nécessaire

        # Faire la prédiction
        prediction = predict(input_data)

        # Afficher le résultat de la prédiction
        st.write("Résultat de la prédiction:", prediction)

if __name__ == "__main__":
    main()
