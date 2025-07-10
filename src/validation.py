import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import platform
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# === Configurazione ===
VALIDATION_DIR = "dataset/validation"  # Percorso del dataset di validazione
IMG_SIZE = (224, 224)  # Dimensione delle immagini di input
BATCH_SIZE = 1  # Dimensione del batch
MODEL_PATH = "models/mobilenet_NOME_v1.h5"  # Percorso del modello salvato

# === Caricamento dataset di validazione ===
validation_ds = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,  # Cartella contenente le immagini di validazione
    labels="inferred",  # Le etichette sono inferite dai nomi delle sottocartelle
    label_mode="categorical",  # Le etichette sono one-hot encoded
    batch_size=BATCH_SIZE,  # Dimensione del batch
    image_size=IMG_SIZE,  # Ridimensionamento delle immagini
    shuffle=False  # Non mescolare, per valutare correttamente
)
class_names = validation_ds.class_names

# === Controllo e caricamento del modello salvato ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modello non trovato: {MODEL_PATH}")
model = load_model(MODEL_PATH)  # Carica il modello pre-addestrato

# === Valutazione del modello sul dataset di validazione ===
loss, accuracy = model.evaluate(validation_ds, verbose=0)
print(f"Loss sul dataset di validazione: {loss:.4f}")
print(f"Accuratezza sul dataset di validazione: {accuracy:.4f}")

# === Predizioni e metriche avanzate ===
y_true = []
y_pred = []

for batch in validation_ds:
    images, labels = batch
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
