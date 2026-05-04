# pipeline.py
# Caso 1: Sector Legal - Análisis de Contratos
# Autores: Antonio Guisado, Lucas Tallafet, Alejandro Tirado

# ==============================================================
# IMPORTS
# ==============================================================
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)

# ==============================================================
# CONFIGURACIÓN
# ==============================================================
ENCODER_PATH = "./src/representacion/bert_ner_contratos/checkpoint_final"
DECODER_PATH = "./src/generacion/modelo_gpt2_contratos"
MAX_LENGTH_ENCODER = 64
MAX_LENGTH_DECODER = 180

# ==============================================================
# CARGA DE MODELOS
# ==============================================================
print("Cargando modelos...")

# Modelo de Lucas (encoder - NER)
encoder_tokenizer = AutoTokenizer.from_pretrained(ENCODER_PATH)
encoder_model = AutoModelForTokenClassification.from_pretrained(ENCODER_PATH)
id2label = encoder_model.config.id2label

# Modelo de Alejandro (decoder - GPT2)
decoder_tokenizer = GPT2Tokenizer.from_pretrained(DECODER_PATH)
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
decoder_model = GPT2LMHeadModel.from_pretrained(DECODER_PATH)

print("Modelos cargados correctamente.\n")

# ==============================================================
# FUNCIÓN ENCODER: extraer entidades (código de Lucas)
# ==============================================================
def extraer_entidades(texto: str) -> dict:
    encoder_model.eval()
    tokens = texto.split()
    encoding = encoder_tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH_ENCODER,
    )
    with torch.no_grad():
        outputs = encoder_model(**encoding)

    pred_ids = torch.argmax(outputs.logits[0], dim=-1).tolist()
    word_ids = encoding.word_ids(batch_index=0)

    word_preds = {}
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in word_preds:
            continue
        word_preds[word_id] = id2label[pred_ids[idx]]

    entidades = {}
    current_entity = None
    current_tokens = []

    for word_id in sorted(word_preds):
        tag = word_preds[word_id]
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


# ==============================================================
# FUNCIÓN DECODER: generar contrato (código de Alejandro)
# ==============================================================
def crear_prompt(entidades: dict) -> str:
    return f"""Genera un contrato de alquiler con los siguientes datos:

arrendador: {entidades.get('ARRENDADOR', '')}
arrendatario: {entidades.get('ARRENDATARIO', '')}
direccion: {entidades.get('DIRECCION', '')}
renta: {entidades.get('RENTA', '')}
fecha_inicio: {entidades.get('FECHA_INICIO', '')}
duracion: {entidades.get('DURACION', '')}

Contrato:
"""

def generar_contrato(entidades: dict) -> str:
    prompt = crear_prompt(entidades)
    inputs = decoder_tokenizer(prompt, return_tensors="pt")
    outputs = decoder_model.generate(
        **inputs,
        max_length=MAX_LENGTH_DECODER,
        do_sample=True,
        temperature=0.6,
        top_p=0.85,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
    )
    return decoder_tokenizer.decode(outputs[0], skip_special_tokens=True)


# ==============================================================
# PIPELINE COMPLETO
# ==============================================================
def run_pipeline(texto_contrato: str):
    print("=" * 60)
    print("PIPELINE - ANÁLISIS Y GENERACIÓN DE CONTRATOS")
    print("=" * 60)

    print("\n📄 TEXTO DE ENTRADA:")
    print(texto_contrato)

    print("\n🔍 ENTIDADES EXTRAÍDAS (Encoder - BERT):")
    entidades = extraer_entidades(texto_contrato)
    for k, v in entidades.items():
        print(f"  {k}: {v}")

    print("\n📝 BORRADOR GENERADO (Decoder - GPT2):")
    borrador = generar_contrato(entidades)
    print(borrador)

    print("\n" + "=" * 60)
    return entidades, borrador


# ==============================================================
# EJEMPLOS DE PRUEBA
# ==============================================================
if __name__ == "__main__":

    contrato1 = (
        "Juan Pérez arrienda a María López el piso en Calle Mayor 5 "
        "por 800 € mensuales desde el 1 de enero de 2024 por 12 meses ."
    )

    contrato2 = (
        "Carlos Ruiz como arrendador cede a Ana García la vivienda "
        "sita en Avenida Constitución 12 con una renta de 650 € "
        "al mes a partir del 1 de marzo de 2024 durante 6 meses ."
    )

    contrato3 = (
        "Pedro Sánchez arrienda a Laura Martínez el local en "
        "Plaza España 3 por 1200 € desde el 15 de febrero "
        "de 2024 por 24 meses ."
    )

    print("\n📌 EJEMPLO 1")
    run_pipeline(contrato1)

    print("\n📌 EJEMPLO 2")
    run_pipeline(contrato2)

    print("\n📌 EJEMPLO 3")
    run_pipeline(contrato3)