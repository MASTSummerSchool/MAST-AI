import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras import layers, models, optimizers

# === Configurazione iniziale ===
DATA_DIR = "dataset/train"  # Percorso del dataset di addestramento
IMG_SIZE = (224, 224)  # Dimensione delle immagini di input
BATCH_SIZE = 256  # Numero di immagini per batch
EPOCHS = 10  # Numero di epoche per l'addestramento
# Percorso dove verrà salvato il modello finale
MODEL_PATH = "mobilenet_NOME_v1.keras"

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

# === Costruzione del modello ===
inputs = layers.Input(shape=IMG_SIZE + (3,))  # Layer di input per immagini RGB
x = data_augmentation(inputs)  # Applica data augmentation
x = tf.keras.applications.mobilenet_v3.preprocess_input(
    x)  # Preprocessa l'input per MobileNetV3
base_model = MobileNetV3Small(
    input_shape=IMG_SIZE + (3,),  # Dimensione delle immagini in input
    include_top=False,  # Esclude il classificatore finale
    weights='imagenet',  # Usa i pesi pre-addestrati su ImageNet
    pooling='avg'  # Usa la media globale delle caratteristiche
)
base_model.trainable = False  # Congela i pesi del modello base
x = base_model(x, training=False)  # Passa l'input attraverso il modello base
# Aggiungi un layer denso con 128 neuroni e ReLU
x = layers.Dense(128, activation='relu')(x)
# Aggiungi un layer di Dropout per prevenire overfitting
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(
    x)  # Layer di output con attivazione softmax
model = models.Model(inputs, outputs)  # Crea il modello finale

# === Compilazione del modello ===
model.compile(
    # Ottimizzatore Adam con un tasso di apprendimento di 0.001
    optimizer=optimizers.Adam(learning_rate=1e-3),
    # Funzione di perdita per la classificazione multi-classe
    loss='categorical_crossentropy',
    # Metriche da monitorare durante l'addestramento (accuratezza)
    metrics=['accuracy']
)

# === Addestramento del modello ===
model.fit(
    train_ds,  # Dataset di addestramento
    epochs=EPOCHS  # Numero di epoche
)

# === Salvataggio del modello ===
model.save(MODEL_PATH)  # Salva il modello addestrato nel percorso specificato
print(f"Modello salvato in {MODEL_PATH}")  # Messaggio di conferma
