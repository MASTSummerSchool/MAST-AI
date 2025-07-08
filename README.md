# MAST-AI

Progetto di classificazione immagini con MobileNetV3, TensorFlow e Keras.

## Requisiti

- Python 3.9+
- vedi [`requirements.txt`](requirements.txt)

## Installazione

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Addestramento

```bash
python src/train.py
```

## Validazione

```bash
python src/validation.py
```

## Inferenza da webcam

```bash
python src/inference.py
```

## Struttura delle cartelle

- `src/` — Codice sorgente
- `models/` — Modelli salvati (`.h5`)
- `dataset/train/` — Immagini di training (sottocartelle per classe)
- `dataset/validation/` — Immagini di validazione (sottocartelle per classe)
- `dataset/test/` — Immagini di test/webcam

## Note

- Su Mac M1/M2 viene usato automaticamente l’ottimizzatore legacy per TensorFlow.
- Per la webcam serve una webcam funzionante e permessi di accesso.

---
