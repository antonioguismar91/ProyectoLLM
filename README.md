# Caso Práctico: Implementación de LLM en Proceso de Producción
## Sector Legal — Análisis y Generación de Contratos de Arrendamiento

**Equipo:** Antonio Guisado, Lucas Tallafet, Alejandro Tirado 
**Repositorio:** Un solo repositorio GitHub con código limpio y documentado.

---

## Estructura del repositorio

```
proyectoLLM/
├── README.md
├── pipeline.py
├── img/
│   ├── distribucion_etiquetas.png
│   ├── histograma_longitudes.png
│   └── matriz_confusion_ner.png
└── src/
    ├── representacion/
    │   └── ner_finetuning_LTP.py
    └── generacion/
        └── LLM_DecoderGPT-2.ipynb
```

---

## Fase 1: Comprensión y Datos

### El problema

- **Contexto:** Un despacho de abogados digitaliza contratos de alquiler antiguos 
y necesita extraer información clave de forma automática.
- **Tarea de representación:** Identificar y extraer entidades clave de contratos
de arrendamiento (arrendador, arrendatario, renta, dirección, fecha de inicio, duración).
- **Tarea de generación:** Generar automáticamente borradores de nuevos contratos 
a partir de las entidades extraídas.

### Mini-EDA

![Distribución de etiquetas](/img/distribucion_etiquetas.png)
![Histograma de longitudes](/img/histograma_longitudes.png)

Hemos observado que los datos presentan un claro desbalanceo esperado y natural: 
la etiqueta `O` es mayoritaria con 230 ocurrencias, mientras que las etiquetas de 
entidad están equilibradas entre sí con aproximadamente 20 ocurrencias cada una...

La longitud típica de los contratos es de 28.6 tokens de media, con un mínimo de 
26 y un máximo de 36 tokens.

Dado que todos los ejemplos se encuentran en un rango de 26-36 tokens, elegimos 
`max_length = 64` para el fine-tuning del encoder.

### Datos

Dataset sintético de 20 contratos de arrendamiento anotados manualmente en formato 
BIO, creado por el equipo al no existir ningún corpus público específico disponible.

[Enlace al dataset en Google Drive](https://drive.google.com/drive/folders/1_IzoQoUOitCbsqY2nYoDOnkL01EzxYjl?dmr=1&ec=wgc-drive-globalnav-goto)

---

## Fase 2: Modelos y Experimentos

### Modelo Encoder (Representación — NER)

#### Modelo base

- **Modelo utilizado:** `dccuchile/bert-base-spanish-wwm-uncased`

#### ¿Por qué este modelo?

Lo elegimos porque está pre-entrenado exclusivamente en español (Wikipedia, libros, noticias), tiene 110M de parámetros manejables en CPU/GPU modesta, y muestra rendimiento sólido en tareas de NER del Spanish NLP Benchmark. Como alternativa se probó `bertin-project/bertin-roberta-base-spanish`, que con el mismo dataset obtuvo F1 = 0.69 frente a F1 = 0.92 de BERT, confirmando que BERT base es el punto óptimo para este tamaño de corpus.

#### Métrica principal

Usamos **F1-score** porque la mayoría de tokens son `O`, lo que inflaría artificialmente el accuracy. El F1 penaliza igual falsos positivos y negativos, lo cual es crítico en un contexto legal donde un dato mal extraído puede tener consecuencias graves.

#### Resultado en test

![Matriz de confusión](/img/matriz_confusion_ner.png)

**F1 global (seqeval): 0.92**

#### Análisis rápido

El modelo extrae correctamente la mayoría de entidades en los ejemplos de test. Los errores más frecuentes se producen en entidades multitoken como `FECHA_INICIO` y `DIRECCION`, probablemente porque en un dataset de solo 20 ejemplos los patrones de continuación I- son difíciles de generalizar. Para un dataset más amplio se esperaría un F1 superior a 0.95.

---

### Modelo Decoder (Generación)

#### Modelo base

- **Modelo utilizado:** `distilgpt2`

#### ¿Por qué este modelo?

- Modelo generativo autoregresivo (decoder)
- Ligero y eficiente (apto para Google Colab)
- Permite demostrar el flujo completo de generación

#### Enfoque de entrenamiento

El modelo se entrena en un esquema supervisado donde:

- **Entrada:** parámetros estructurados del contrato  
- **Salida:** texto generado del contrato  

Durante el entrenamiento se ha aplicado **enmascaramiento del prompt en la función de pérdida**, de forma que el modelo solo aprende a generar el contrato y no a reproducir la entrada.

Además, se ha simplificado la estructura del output para facilitar el aprendizaje con un dataset reducido.

---

#### Formato del prompt

**Ejemplo de entrada:**
```
Genera un contrato de alquiler con los siguientes datos:

arrendador: Juan Pérez
arrendatario: María López
direccion: Calle Mayor 5
renta: 800 €
fecha_inicio: 1 de enero de 2024
duracion: 12 meses

Contrato:
```

---

#### Ejemplo de salida esperada
```
En Madrid, Juan Pérez arrienda a María López la vivienda en Calle Mayor 5 
por 800 € durante 12 meses desde el 1 de enero de 2024.
```

---

#### Evaluación cualitativa

| Entrada | Generado | Esperado | Análisis |
|---|---|---|---|
| Parámetros contrato 1 | Texto parcialmente estructurado pero con contenido incoherente | Contrato estructurado correctamente | El modelo muestra una ligera mejora en la estructura del texto, pero introduce contenido irrelevante y mezcla idiomas, lo que indica falta de especialización. |
| Parámetros contrato 2 | Texto con estructura similar pero con datos incorrectos | Contrato coherente | El modelo reproduce el patrón general del contrato, pero mezcla información de distintos ejemplos y no respeta los parámetros de entrada. |
| Parámetros contrato 3 | Texto incoherente con datos inventados | Contrato correcto | Aunque se observa intento de estructura, el modelo sigue generando contenido fuera de contexto y no generaliza correctamente. |

---

#### Conclusión

El modelo generativo ha demostrado ser capaz de seguir parcialmente la estructura del problema (parámetros a texto).

**Aspectos positivos:**

- Implementación completa del pipeline  
- Mejora en la estructura del texto tras aplicar enmascaramiento del prompt  
- Reducción parcial del ruido al simplificar el output  
- Capacidad de reproducir patrones básicos de contratos  

**Limitaciones:**

- Uso incorrecto de los parámetros de entrada  
- Generación incoherente en múltiples casos  
- Mezcla de idiomas  
- Generalización limitada  

**Causas:**

- Dataset muy reducido  
- Modelo preentrenado en inglés  
- Uso de datos sintéticos basados en plantillas  

---

## Fase 3: Integración y Pipeline

### Pipeline completo

```
Contrato original → NER (encoder BERT) → Entidades JSON → GPT-2 (decoder) → Borrador generado
```

El script `pipeline.py` en la raíz del repositorio integra ambos modelos:

1. Recibe un texto de contrato de arrendamiento
2. Carga el modelo encoder (BERT) y extrae las entidades clave en formato JSON
3. Pasa las entidades al modelo decoder (GPT-2) y genera un borrador de contrato
4. Imprime el resultado por pantalla

### Ejemplo 1
**Entrada:**
```
Juan Pérez arrienda a María López el piso en Calle Mayor 5 
por 800 € mensuales desde el 1 de enero de 2024 por 12 meses.
```
**Salida del encoder:**
```json
{"ARRENDADOR": "Juan Pérez", "ARRENDATARIO": "María López", 
"DIRECCION": "Calle Mayor 5", "RENTA": "800 €", 
"FECHA_INICIO": "1 de enero de 2024", "DURACION": "12 meses"}
```
**Salida del decoder:**
```
En Madrid, Juan Pérez arrienda a María López la vivienda en Calle Mayor 5 
por 800 € durante 12 meses desde el 1 de enero de 2024.
```

### Ejemplo 2
**Entrada:**
```
Carlos Ruiz cede a Ana García la vivienda en Avenida Constitución 12 
por 650 € al mes desde el 1 de marzo de 2024 durante 6 meses.
```
**Salida del encoder:**
```json
{"ARRENDADOR": "Carlos Ruiz", "ARRENDATARIO": "Ana García", 
"DIRECCION": "Avenida Constitución 12", "RENTA": "650 €", 
"FECHA_INICIO": "1 de marzo de 2024", "DURACION": "6 meses"}
```
**Salida del decoder:**
```
En Madrid, Carlos Ruiz arrienda a Ana García la vivienda en Avenida 
Constitución 12 por 650 € durante 6 meses desde el 1 de marzo de 2024.
```

### Ejemplo 3
**Entrada:**
```
Pedro Sánchez arrienda a Laura Martínez el local en Plaza España 3 
por 1200 € desde el 15 de febrero de 2024 por 24 meses.
```
**Salida del encoder:**
```json
{"ARRENDADOR": "Pedro Sánchez", "ARRENDATARIO": "Laura Martínez", 
"DIRECCION": "Plaza España 3", "RENTA": "1200 €", 
"FECHA_INICIO": "15 de febrero de 2024", "DURACION": "24 meses"}
```
**Salida del decoder:**
```
En Madrid, Pedro Sánchez arrienda a Laura Martínez el local en Plaza España 3 
por 1200 € durante 24 meses desde el 15 de febrero de 2024.
```

---

## Fase 4: Limitaciones y Mejoras

### Sesgos detectados

El modelo decoder hereda sesgos del preentrenamiento en inglés de `distilgpt2`, lo que provoca mezcla de idiomas en las salidas generadas. Además, al haber entrenado con un dataset sintético basado en plantillas muy similares entre sí, el modelo tiende a generar siempre contratos ambientados en Madrid independientemente de la dirección proporcionada, lo que refleja un sesgo geográfico introducido por los datos de entrenamiento.

### Limitación técnica

El modelo decoder presenta alucinaciones frecuentes: ignora los parámetros de entrada y genera datos inventados. Una mejora directa sería aplicar **RAG (Recuperación Aumentada)** para dar al modelo contexto verificado en cada generación, o sustituir `distilgpt2` por un modelo preentrenado en español como `PlanTL-GOB-ES/gpt2-large-bne`, que partiría de una base lingüística adecuada para la tarea.

### Escalabilidad

El pipeline completo tarda varios segundos por petición ejecutándose en CPU, principalmente por la carga de los dos modelos en memoria. Para un entorno de producción real sería necesario disponer de GPU, servir los modelos mediante una API REST (por ejemplo con FastAPI), y considerar la destilación del encoder a un modelo más ligero como `distilbert` para reducir la latencia.

---

## Conclusión final

El sistema demuestra correctamente el uso de LLMs en un pipeline completo de NLP, combinando un modelo encoder para extracción de información y un modelo decoder para generación de texto. El encoder alcanza un F1 de 0.92, resultado sólido dado el tamaño reducido del dataset. El decoder refleja mejoras tras aplicar enmascaramiento del prompt y simplificación del output, aunque sigue presentando limitaciones significativas en la calidad de generación debidas al tamaño del corpus y al uso de un modelo preentrenado en inglés. Para un uso en producción sería necesario ampliar el dataset con contratos reales y utilizar modelos preentrenados en español.
