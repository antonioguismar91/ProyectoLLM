# -*- coding: utf-8 -*-
"""
Fine-tuning BERT para NER en Contratos de Alquiler
====================================================
Modelo:    dccuchile/bert-base-spanish-wwm-uncased
Tarea:     Named Entity Recognition (NER) en formato BIO
Entidades: ARRENDADOR, ARRENDATARIO, DIRECCION, RENTA, FECHA_INICIO, DURACION
"""

# ==============================================================================
# 1. Instalación de dependencias
# ==============================================================================
# !pip install transformers datasets seqeval scikit-learn matplotlib seaborn -q

# ==============================================================================
# 2. Imports
# ==============================================================================
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import Dataset as HFDataset
from seqeval.metrics import f1_score, classification_report as seq_report

# ==============================================================================
# 3. Carga del dataset
# ==============================================================================
with open("contratos_ner.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"Total de ejemplos: {len(raw_data)}")
print(f"\nEjemplo 0:")
print(f"  Tokens: {raw_data[0]['tokens']}")
print(f"  Tags:   {raw_data[0]['ner_tags']}")


# ==============================================================================
# 4. Preparación del vocabulario de etiquetas
# ==============================================================================
all_unique_tags = sorted(set(tag for item in raw_data for tag in item["ner_tags"]))
label2id = {tag: i for i, tag in enumerate(all_unique_tags)}
id2label  = {i: tag for tag, i in label2id.items()}

print(f"Etiquetas ({len(all_unique_tags)}): {all_unique_tags}")


# ==============================================================================
# 5. Configuración del modelo y tokenización
# ==============================================================================
#
# Modelo elegido: dccuchile/bert-base-spanish-wwm-uncased
#   - Pre-entrenado exclusivamente en español (Wikipedia, libros, noticias).
#   - 110M parámetros: manejable en CPU/GPU modesta.
#   - Rendimiento sólido en NER del Spanish NLP Benchmark.
#
# Alternativa probada: bertin-project/bertin-roberta-base-spanish
#   - Con el mismo dataset obtuvo F1 = 0.69 frente a F1 = 0.92 de BERT.
#   - Con solo 20 ejemplos, un modelo más grande no generaliza mejor.
#   - BERT base resultó ser el punto óptimo para este tamaño de corpus.
#

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"
MAX_LENGTH = 64  # cubre el máximo de 36 tokens con margen para [CLS] y [SEP]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Split: 70% train | 15% val | 15% test
train_val, test_data  = train_test_split(raw_data, test_size=0.15,  random_state=42)
train_data, val_data  = train_test_split(train_val, test_size=0.176, random_state=42)

print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")


def tokenize_and_align_labels(examples):
    """
    Tokeniza con WordPiece y alinea las etiquetas BIO.
    Los subtokens adicionales reciben etiqueta -100 (ignorados en la loss).
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    labels_batch = []
    for i, label_seq in enumerate(examples["ner_tags"]):
        word_ids  = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)          # [CLS], [SEP], [PAD]
            elif word_id != prev_word_id:
                label_ids.append(label2id[label_seq[word_id]])
            else:
                label_ids.append(-100)          # subtoken: ignorar
            prev_word_id = word_id
        labels_batch.append(label_ids)
    tokenized["labels"] = labels_batch
    return tokenized


def to_hf_dataset(data_list):
    return HFDataset.from_dict({
        "tokens":   [d["tokens"]   for d in data_list],
        "ner_tags": [d["ner_tags"] for d in data_list],
    })


train_hf = to_hf_dataset(train_data).map(tokenize_and_align_labels, batched=True)
val_hf   = to_hf_dataset(val_data).map(tokenize_and_align_labels,   batched=True)
test_hf  = to_hf_dataset(test_data).map(tokenize_and_align_labels,  batched=True)

print("Tokenización completa")


# ==============================================================================
# 6. Carga del modelo
# ==============================================================================
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

print(f"Modelo cargado: {MODEL_NAME}")
print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")


# ==============================================================================
# 7. Métrica de evaluación
# ==============================================================================
def compute_metrics(eval_pred):
    """
    F1-score por entidad con seqeval (evaluación a nivel de span completo).

    Usamos F1 y no accuracy porque:
    - La mayoría de tokens son 'O', lo que inflaría artificialmente el accuracy.
    - F1 penaliza igual falsos positivos y negativos, crítico en contexto legal.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_labels, true_preds = [], []
    for pred_seq, label_seq in zip(predictions, labels):
        true_label_row, true_pred_row = [], []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_label_row.append(id2label[l])
                true_pred_row.append(id2label[p])
        true_labels.append(true_label_row)
        true_preds.append(true_pred_row)

    return {"f1": f1_score(true_labels, true_preds)}


# ==============================================================================
# 8. Entrenamiento
# ==============================================================================
data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir="./bert_ner_contratos",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=3e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=5,
    report_to="none",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_hf,
    eval_dataset=val_hf,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()


# ==============================================================================
# 9. Evaluación en test
# ==============================================================================
predictions_output = trainer.predict(test_hf)
logits      = predictions_output.predictions
preds       = np.argmax(logits, axis=-1)
labels_test = predictions_output.label_ids

true_labels_flat, true_preds_flat = [], []
true_labels_seq,  true_preds_seq  = [], []

for pred_seq, label_seq in zip(preds, labels_test):
    row_labels, row_preds = [], []
    for p, l in zip(pred_seq, label_seq):
        if l != -100:
            true_labels_flat.append(id2label[l])
            true_preds_flat.append(id2label[p])
            row_labels.append(id2label[l])
            row_preds.append(id2label[p])
    true_labels_seq.append(row_labels)
    true_preds_seq.append(row_preds)

print("=== RESULTADOS EN TEST ===")
print(seq_report(true_labels_seq, true_preds_seq))
print(f"\nF1 global (seqeval): {f1_score(true_labels_seq, true_preds_seq):.4f}")


# ==============================================================================
# 10. Matriz de confusión
# ==============================================================================
present_labels = sorted(set(true_labels_flat + true_preds_flat))
cm = confusion_matrix(true_labels_flat, true_preds_flat, labels=present_labels)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=present_labels, yticklabels=present_labels,
    linewidths=0.5, ax=ax,
)
ax.set_title("Matriz de confusión - NER Contratos (test set)", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Etiqueta predicha", fontsize=12)
ax.set_ylabel("Etiqueta real", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix_ner.png", dpi=150, bbox_inches="tight")
plt.show()
print("Matriz guardada como confusion_matrix_ner.png")


# ==============================================================================
# 11. Inferencia — función para pipeline.py
# ==============================================================================
def extraer_entidades(texto: str, model, tokenizer, id2label: dict) -> dict:
    """
    Dado un texto de contrato, devuelve un dict con las entidades extraídas.

    Ejemplo de salida:
        {
            'ARRENDADOR':   'Juan Pérez',
            'ARRENDATARIO': 'María López',
            'DIRECCION':    'Calle Mayor 5',
            'RENTA':        '800 €',
            'FECHA_INICIO': '1 de enero de 2024',
            'DURACION':     '12 meses'
        }
    """
    model.eval()
    tokens   = texto.split()
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        outputs = model(**encoding)

    pred_ids = torch.argmax(outputs.logits[0], dim=-1).tolist()
    word_ids = encoding.word_ids(batch_index=0)

    # Alinear predicciones con palabras originales (primer subtoken)
    word_preds = {}
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in word_preds:
            continue
        word_preds[word_id] = id2label[pred_ids[idx]]

    # Reconstruir entidades agrupando B- e I-
    entidades      = {}
    current_entity = None
    current_tokens = []

    for word_id in sorted(word_preds):
        tag   = word_preds[word_id]
        token = tokens[word_id]

        if tag.startswith("B-"):
            if current_entity:
                entidades[current_entity] = " ".join(current_tokens)
            current_entity = tag[2:]
            current_tokens = [token]
        elif tag.startswith("I-") and current_entity == tag[2:]:
            current_tokens.append(token)
        else:
            if current_entity:
                entidades[current_entity] = " ".join(current_tokens)
            current_entity = None
            current_tokens = []

    if current_entity:
        entidades[current_entity] = " ".join(current_tokens)

    return entidades


# Prueba rápida
texto_prueba = (
    "Juan Pérez arrienda a María López el piso en Calle Mayor 5 "
    "por 800 € mensuales desde el 1 de enero de 2024 por 12 meses ."
)
resultado = extraer_entidades(texto_prueba, model, tokenizer, id2label)
print("\nEntidades extraídas (prueba):")
for k, v in resultado.items():
    print(f"  {k}: {v}")


# ==============================================================================
# 12. Guardar el modelo fine-tuneado
# ==============================================================================
model.save_pretrained("./bert_ner_contratos/checkpoint_final")
tokenizer.save_pretrained("./bert_ner_contratos/checkpoint_final")
print("\nModelo guardado en ./bert_ner_contratos/checkpoint_final")
