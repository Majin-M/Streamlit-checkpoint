
import streamlit as st
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fonction de prétraitement des données
def preprocess(text):
    # Tokeniser le texte en phrases
    sentences = sent_tokenize(text)
    
    # Tokeniser chaque phrase en mots et supprimer les mots vides
    stop_words = set(stopwords.words("french"))
    preprocessed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        filtered_words = [word for word in words if word not in stop_words and word.isalnum()]
        preprocessed_sentences.append(" ".join(filtered_words))
    
    return preprocessed_sentences

# Définir la fonction de similarité
def get_most_relevant_sentence(query, preprocessed_sentences):
    # Transformer la requête en vecteur tf-idf
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences + [query])
    
    # Calculer la similarité cosinus entre le vecteur tf-idf de la requête et chaque vecteur tf-idf de phrase
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    # Obtenir l'index de la phrase la plus similaire
    most_similar_idx = similarities.argmax()
    
    return preprocessed_sentences[most_similar_idx]

# Définir la fonction du chatbot
def chatbot(query, preprocessed_sentences):
    most_relevant_sentence = get_most_relevant_sentence(query, preprocessed_sentences)
    # Vous pouvez retourner la phrase la plus pertinente comme réponse pour plus de simplicité
    return most_relevant_sentence

# Créer une application Streamlit
def main():
    st.title("Chatbot sur l'Intelligence Artificielle")
    st.write("Bienvenue ! Posez-moi toutes vos questions sur l'Intelligence Artificielle.")

    fichier_texte = "votre_fichier_texte.txt"  # Remplacez par le chemin de votre fichier texte
    with open(fichier_texte, "r", encoding="utf-8") as file:
        donnees_texte = file.read()

    phrases_pretraitees = preprocess(donnees_texte)

    question_utilisateur = st.text_input("Posez une question :")
    if question_utilisateur:
        reponse = chatbot(question_utilisateur, phrases_pretraitees)
        st.text_area("Réponse :", value=reponse, height=200)

if __name__ == "__main__":
    main()
