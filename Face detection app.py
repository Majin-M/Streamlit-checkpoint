import streamlit as st
import cv2
import numpy as np

# Ajouter des instructions pour guider l'utilisateur
st.write("## Bienvenue dans l'application de détection de visages")
st.write("Utilisez le panneau de gauche pour ajuster les paramètres et voir les résultats en direct.")

# Charger le modèle de détection de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fonction pour détecter les visages et dessiner des rectangles autour d'eux
def detect_faces(image, scaleFactor, minNeighbors, color):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    return image, len(faces)

# Fonction pour sauvegarder l'image avec les visages détectés
def save_image_with_faces(image, filename):
    cv2.imwrite(filename, image)

# Charger une image depuis le fichier
uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lire l'image à partir du fichier
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Ajouter des contrôles pour ajuster les paramètres
    scaleFactor = st.slider("scaleFactor", 1.1, 2.0, 1.2, 0.1)
    minNeighbors = st.slider("minNeighbors", 1, 10, 5)
    # Convertir la couleur sélectionnée par l'utilisateur au format BGR
    color_hex = st.color_picker("Couleur du rectangle", "#FF0000")
    color_bgr = (int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16))


    # Détecter les visages et afficher l'image résultante
    result_image, num_faces = detect_faces(image, scaleFactor, minNeighbors, color_bgr)
    st.image(result_image, channels="BGR")

    # Bouton pour sauvegarder l'image avec les visages détectés
    if st.button("Enregistrer l'image avec les visages"):
        filename = "faces_detected_" + str(num_faces) + ".jpg"
        save_image_with_faces(result_image, filename)
        st.success("L'image avec les visages détectés a été enregistrée en tant que " + filename)
