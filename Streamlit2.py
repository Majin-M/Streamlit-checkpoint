import streamlit as st
import pandas as pd

data = pd.read_csv("Financial_inclusion_dataset.csv.csv")

print(data.head())

print(data.info())

from pandas_profiling import ProfileReport

profile = ProfileReport(data)
profile.to_file("Financial_inclusion_report.html")

categorical_columns = [col for col in data.select_dtypes(include=['object']).columns if col not in ['cellphone_access', 'bank_account']]

data_encoded = pd.get_dummies(data, columns=categorical_columns)

# Ajouter les colonnes booléennes encodées manuellement
data_encoded['cellphone_access'] = data['cellphone_access'].map({'Yes': 1, 'No': 0})
data_encoded['bank_account'] = data['bank_account'].map({'Yes': 1, 'No': 0})

# Séparation des caractéristiques et de la cible
X = data_encoded.drop('bank_account', axis=1)
y = data_encoded['bank_account']

# Création et entraînement du modèle
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)

from sklearn.metrics import accuracy_score
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

# Titre de l'application
st.title('Prediction ouverture de compte bancaire')

# Champs de saisie pour les caractéristiques
country = st.selectbox('Pays', data['country'].unique())
year = st.slider('Année', int(data['year'].min()), int(data['year'].max()), int(data['year'].median()))
# Ajoutez d'autres champs de saisie pour les caractéristiques restantes

# Bouton de validation
if st.button('Prédire'):
    # Encodage des caractéristiques
    features = pd.DataFrame({
        'country': [country],
        'year': [year],
        # Ajoutez d'autres caractéristiques ici
    })
    features_encoded = pd.get_dummies(features, drop_first=True)

    # Prédiction avec le modèle
    prediction = model.predict(features_encoded)
    
    # Affichage de la prédiction
    st.write('La prédiction de l\'ouverture de compte bancaire est:', prediction)