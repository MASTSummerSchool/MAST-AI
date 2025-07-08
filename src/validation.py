import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import platform

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

# === Controllo e caricamento del modello salvato ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modello non trovato: {MODEL_PATH}")
model = load_model(MODEL_PATH)  # Carica il modello pre-addestrato

# === (Opzionale) Se vuoi ricompilare il modello per ulteriori valutazioni/addestramenti ===
if platform.machine() in ["arm64", "arm"]:
    from tensorflow.keras.optimizers.legacy import Adam
    optimizer = Adam(learning_rate=1e-3)
else:
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy', metrics=['accuracy'])

# === Valutazione del modello sul dataset di validazione ===
loss, accuracy = model.evaluate(validation_ds)

# === Stampa dei risultati ===
print(f"Loss sul dataset di validazione: {loss}")
print(f"Accuratezza sul dataset di validazione: {accuracy}")
