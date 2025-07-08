import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import cv2
from datetime import datetime

# === Configurazione ===
MODEL_PATH = "models/mobilenet_NOME_v1.h5"  # Percorso del modello salvato
IMG_SIZE = (224, 224)  # Dimensione dell'immagine di input
CLASS_NAMES = [
    "aqualy", "calcolatrice_casio", "bicchiere", "iphone13", "mouse_wireless",
    "pennarello_giotto", "persona", "webcam_box"
]  # Classi del tuo dataset

# === Cattura immagine dalla webcam e salvala ===


def capture_image_from_webcam(save_dir="dataset/test"):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Impossibile accedere alla webcam.")
    print("Premi SPAZIO per scattare la foto, ESC per uscire.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Webcam - Premi SPAZIO per scattare", frame)
        key = cv2.waitKey(1)
        if key % 256 == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError("Cattura annullata.")
        elif key % 256 == 32:  # SPACE
            filename = f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            img_path = os.path.join(save_dir, filename)
            cv2.imwrite(img_path, frame)
            cap.release()
            cv2.destroyAllWindows()
            print(f"Immagine salvata in {img_path}")
            return img_path


# Controlla che il modello esista
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modello non trovato: {MODEL_PATH}")

# Caricamento del modello salvato
model = load_model(MODEL_PATH)

# Funzione per fare inferenza su una singola immagine


def predict_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_class_idx]
    predicted_prob = predictions[0][predicted_class_idx]
    return predicted_class, predicted_prob


# === Cattura immagine dalla webcam, salva e inferisci ===
try:
    IMG_PATH = capture_image_from_webcam()
    predicted_class, predicted_prob = predict_image(IMG_PATH)
    print(f"Classe predetta: {predicted_class}")
    print(f"Confidenza della predizione: {predicted_prob:.4f}")
except Exception as e:
    print(f"Errore: {e}")
