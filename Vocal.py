import streamlit as st
import speech_recognition as sr

class SpeechRecognitionApp:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.selected_api = "Google"  # API par défaut est Google Speech Recognition

    def transcribe_speech(self, audio):
        try:
            if self.selected_api == "Google":
                return self.recognizer.recognize_google(audio)
            else:
                raise ValueError("API sélectionnée invalide")
        except sr.UnknownValueError:
            return "La parole n'a pas pu être reconnue"
        except sr.RequestError as e:
            return f"Erreur lors de la récupération des résultats de l'API : {e}"

    def save_transcription_to_file(self, text, filename):
        with open(filename, "w") as file:
            file.write(text)

    def recognize_and_save(self, audio, filename):
        text = self.transcribe_speech(audio)
        if text:
            self.save_transcription_to_file(text, filename)
            st.success(f"Le texte transcrit a été enregistré dans {filename}")
        else:
            st.error("Impossible de transcrire la parole")


    def choose_language(self):
        language = st.radio("Sélectionnez la langue dans laquelle vous parlez :", ["Anglais", "Français"])
        if language == "Français":
            self.recognizer.energy_threshold = 4000  # Ajuster le seuil d'énergie pour le français
        else:
            self.recognizer.energy_threshold = 300  # Ajuster le seuil d'énergie pour l'anglais

    def start_recognition(self, filename):
        with sr.Microphone() as source:
            st.write("En écoute...")
            audio = self.recognizer.listen(source)

        self.recognize_and_save(audio, filename)

def main():
    st.title("Speech Recognition App")
    app = SpeechRecognitionApp()
    
    app.choose_language()
    filename = st.text_input("Entrez le nom du fichier pour enregistrer la transcription : ")
    if st.button("Commencer la reconnaissance vocale"):
        app.start_recognition(filename)

if __name__ == "__main__":
    main()
