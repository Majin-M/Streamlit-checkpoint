
import streamlit as st
import nltk
import epub2txt

from epub2txt import epub2txt
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


# Définir la fonction du chatbot
def chatbot(query, preprocessed_sentences):
    if not preprocessed_sentences:
        return "Désolé, je ne dispose pas de données pour répondre à votre question."

    most_relevant_sentence = get_most_relevant_sentence(query, preprocessed_sentences)
    return most_relevant_sentence
# Définir la fonction de similarité
def get_most_relevant_sentence(query, preprocessed_sentences):
    if not preprocessed_sentences:
        return ""

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences + [query])

    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]

    most_similar_idx = similarities.argmax()
    
    return preprocessed_sentences[most_similar_idx]


def extract_text_from_epub(epub_file):
    return epub2txt(epub_file)

# Créer une application 
# 
# Streamlit
def main():
    st.title("Chatbot sur la seconde guerre mondiale")
    st.write("Bienvenue ! Posez-moi toutes vos questions sur la seconde guerre mondiale.")
    
    epub_file = "Encyclopédie de la Seconde .epub"
    texte_extrait = extract_text_from_epub(epub_file)  # Modifier cette ligne
    phrases_pretraitees = preprocess(texte_extrait)

    question_utilisateur = st.text_input("Posez une question :")
    if question_utilisateur:
        reponse = chatbot(question_utilisateur, phrases_pretraitees)
        st.text_area("Réponse :", value=reponse, height=200)

if __name__ == "__main__":
    main()
