from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras import layers, models, optimizers
import platform

# === Configurazione iniziale ===
DATA_DIR = "dataset/train"  # Percorso del dataset di addestramento
IMG_SIZE = (224, 224)  # Dimensione delle immagini di input
BATCH_SIZE = 32  # Numero di immagini per batch
EPOCHS = 10  # Numero di epoche per l'addestramento
# Percorso dove verrà salvato il modello finale
# Estensione .h5 per compatibilità TF 2.11
MODEL_PATH = "models/mobilenet_NOME_v3.h5"

# === Caricamento dataset ===
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,  # Cartella contenente le immagini
    labels="inferred",  # Le etichette sono inferite dai nomi delle sottocartelle
    label_mode="categorical",  # Le etichette sono one-hot encoded
    batch_size=BATCH_SIZE,  # Dimensione del batch
    image_size=IMG_SIZE,  # Ridimensionamento delle immagini
    shuffle=True  # Mescola il dataset
)

num_classes = len(train_ds.class_names)  # Numero di classi nel dataset

# === Data Augmentation ===
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),  # Flip orizzontale casuale
    layers.RandomRotation(0.2),  # Rotazione casuale tra -20% e +20%
    layers.RandomZoom(0.2),  # Zoom casuale tra -20% e +20%
    layers.RandomContrast(0.2),  # Contrasto casuale tra -20% e +20%
])

# === Costruzione del modello base ===
inputs = layers.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
base_model = MobileNetV3Small(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
base_model.trainable = False
x = base_model(x, training=False)
x = layers.Dense(192, activation='relu')(x)  # Miglior valore trovato
x = layers.Dropout(0.3)(x)                   # Miglior valore trovato
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs)

# === Selezione ottimizzatore in base all'architettura ===
if platform.machine() in ["arm64", "arm"]:
    from tensorflow.keras.optimizers.legacy import Adam
    optimizer = Adam(learning_rate=0.0005)   # Miglior valore trovato
else:
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=0.0005)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === Addestramento testa (feature extraction) ===

early_stopping = EarlyStopping(
    monitor="loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=None,  # aggiungi val_ds se disponibile
    callbacks=[early_stopping]
)

# === Salvataggio del modello ===
model.save(MODEL_PATH, save_format="h5")
print(f"Modello salvato in {MODEL_PATH}")
