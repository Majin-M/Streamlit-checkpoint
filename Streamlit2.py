import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("C:\\Users\\marcs\\OneDrive\\Bureau\\Financial_inclusion_dataset.csv")

print(data.head())

print(data.info())


categorical_columns = [col for col in data.select_dtypes(include=['object']).columns if col not in ['cellphone_access', 'bank_account']]

data_encoded = pd.get_dummies(data, columns=categorical_columns)

# Ajouter les colonnes booléennes encodées manuellement
data_encoded['cellphone_access'] = data['cellphone_access'].map({'Yes': 1, 'No': 0})
data_encoded['bank_account'] = data['bank_account'].map({'Yes': 1, 'No': 0})


# Séparation des caractéristiques et de la cible
X = data_encoded.drop('bank_account', axis=1)
y = data_encoded['bank_account']

# Division les données en un ensemble d'entraînement et un ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création et entraînement du modèle
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

#Evaluation des performances du modèle
accuracy = accuracy_score(y_test, y_pred)
st.write("Exactitude du modèle :", accuracy)


# Titre de l'application
st.title('Prediction ouverture de compte bancaire')

# Champs de saisie pour les caractéristiques
country = st.selectbox('Pays', data_encoded['country'].unique())
year = st.slider('Année', int(data_encoded['year'].min()), int(data_encoded['year'].max()), int(data_encoded['year'].median()))


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
