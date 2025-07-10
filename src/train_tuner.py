import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras import layers, models
import platform
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping

# === Configurazione iniziale ===
TRAIN_DIR = "dataset/train"  # Percorso del dataset di addestramento
VAL_DIR = "dataset/validation"  # Percorso del dataset di validazione
IMG_SIZE = (224, 224)  # Dimensione delle immagini di input
BATCH_SIZE = 32  # Numero di immagini per batch
EPOCHS = 8  # Numero di epoche per l'addestramento
# Percorso dove verrà salvato il modello finale
# Estensione .h5 per compatibilità TF 2.11
MODEL_PATH = "models/mobilenet_NOME_v1.h5"

# === Caricamento dataset ===
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,  # Cartella contenente le immagini
    labels="inferred",  # Le etichette sono inferite dai nomi delle sottocartelle
    label_mode="categorical",  # Le etichette sono one-hot encoded
    batch_size=BATCH_SIZE,  # Dimensione del batch
    image_size=IMG_SIZE,  # Ridimensionamento delle immagini
    shuffle=True  # Mescola il dataset
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,  # Cartella contenente le immagini di validazione
    labels="inferred",  # Le etichette sono inferite dai nomi delle sottocartelle
    label_mode="categorical",  # Le etichette sono one-hot encoded
    batch_size=BATCH_SIZE,  # Dimensione del batch
    image_size=IMG_SIZE,  # Ridimensionamento delle immagini
    shuffle=False  # Non mescolare, per valutare correttamente
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


def build_model(hp):
    # Layer di input per immagini RGB
    inputs = layers.Input(shape=IMG_SIZE + (3,))
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
    # Passa l'input attraverso il modello base
    x = base_model(x, training=False)
    # Hyperparam: numero neuroni
    x = layers.Dense(
        hp.Int("dense_units", min_value=64, max_value=256, step=32),
        activation='relu'
    )(x)
    # Hyperparam: dropout
    x = layers.Dropout(hp.Float("dropout", 0.3, 0.7, step=0.1))(x)
    outputs = layers.Dense(num_classes, activation='softmax')(
        x)  # Layer di output con attivazione softmax
    model = models.Model(inputs, outputs)  # Crea il modello finale

    # Hyperparam: learning rate
    lr = hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4, 5e-5])
    # Ottimizzatore legacy su Mac ARM
    if platform.machine() in ["arm64", "arm"]:
        from tensorflow.keras.optimizers.legacy import Adam
        optimizer = Adam(learning_rate=lr)
    else:
        from tensorflow.keras.optimizers import Adam
        optimizer = Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=8,
    executions_per_trial=1,
    directory="models/tuner_results",
    project_name="mobilenetv3_tuning"
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

tuner.search(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

best_model = tuner.get_best_models(num_models=1)[0]
best_model.save(MODEL_PATH, save_format="h5")
print(f"Modello salvato in {MODEL_PATH}")
