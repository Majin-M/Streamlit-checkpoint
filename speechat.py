import streamlit as st
import nltk
import speech_recognition as sr

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Téléchargement des ressources nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_data(file_path):
    # Extraction du texte du fichier
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Suppression de la ponctuation et des mots vides, lemmatisation
    stop_words = set(stopwords.words('french'))
    lemmatizer = WordNetLemmatizer()
    preprocessed_sentences = []
    
    # Séparation du texte en phrases
    sentences = sent_tokenize(text)
    
    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        preprocessed_sentence = ' '.join(lemmatized_tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return preprocessed_sentences


# Fonction pour le chatbot
def chatbot(query, preprocessed_sentences):
    if not preprocessed_sentences:
        return "Désolé, je ne dispose pas de données pour répondre à votre question."

    most_relevant_sentence = get_most_relevant_sentence(query, preprocessed_sentences)
    return most_relevant_sentence

# Fonction pour obtenir la phrase la plus pertinente
def get_most_relevant_sentence(query, preprocessed_sentences):
    if not preprocessed_sentences:
        return "Désolé, je ne dispose pas de données pour répondre à votre question."

    vectorizer = CountVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences + [query])

    query_vector = tfidf_matrix[-1]
    sentence_vectors = tfidf_matrix[:-1]

    jaccard_similarities = [len(set(query_vector.indices) & set(vector.indices)) / len(set(query_vector.indices) | set(vector.indices)) for vector in sentence_vectors]

    most_similar_idx = jaccard_similarities.index(max(jaccard_similarities))

    return preprocessed_sentences[most_similar_idx]


# Fonction pour transcrire la parole en texte
def transcribe_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Parlez quelque chose...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language="fr-FR") # Reconnaissance vocale en français
        return text
    except sr.UnknownValueError:
        st.write("Impossible de comprendre l'audio.")
    except sr.RequestError:
        st.write("Erreur de service de reconnaissance vocale.")
        

# Création de l'application Streamlit
def main():
    st.title("Chatbot avec entrée de texte et de parole")
    st.write("Bienvenue ! Posez-moi toutes vos questions sur la seconde guerre mondiale.")
    
    texte_extrait = "Seconde guerre mondiale.txt"
    
    phrases_pretraitees = preprocess_data(texte_extrait)
    print(phrases_pretraitees)

    input_option = st.radio("Choisissez l'option d'entrée:", ("Texte", "Parole"))

    if input_option == "Texte":
        user_input = st.text_input("Entrez votre texte ici:")
        if st.button("Envoyer"):
            response = chatbot(user_input, phrases_pretraitees)
            st.write("Réponse du Chatbot:", response)
    elif input_option == "Parole":
        if st.button("Démarrer l'enregistrement"):
            user_input = transcribe_speech()
            if user_input:
                response = chatbot(user_input, phrases_pretraitees)
                st.write("Réponse du Chatbot:", response)

if __name__ == "__main__":
    main()
