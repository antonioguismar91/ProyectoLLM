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

## Modelo Decoder (Generación)

### Modelo base

- **Modelo utilizado:** `distilgpt2`

### ¿Por qué este modelo?

- Modelo generativo autoregresivo (decoder)
- Ligero y eficiente (apto para Google Colab)
- Permite demostrar el flujo completo de generación

### Enfoque de entrenamiento

El modelo se entrena en un esquema supervisado donde:

- **Entrada:** parámetros estructurados del contrato  
- **Salida:** texto generado del contrato  

Durante el entrenamiento se ha aplicado **enmascaramiento del prompt en la función de pérdida**, de forma que el modelo solo aprende a generar el contrato y no a reproducir la entrada.

Además, se ha simplificado la estructura del output para facilitar el aprendizaje con un dataset reducido.

---

### Formato del prompt

**Ejemplo de entrada:**
Genera un contrato de alquiler con los siguientes datos:

arrendador: Juan Pérez
arrendatario: María López
direccion: Calle Mayor 5
renta: 800 €
fecha_inicio: 1 de enero de 2024
duracion: 12 meses

Contrato:

---

### Ejemplo de salida esperada
En Madrid, Juan Pérez arrienda a María López la vivienda en Calle Mayor 5 por 800 € durante 12 meses desde el 1 de enero de 2024.

---

### Evaluación cualitativa

- **Entrada:** Parámetros contrato 1  
  **Generado:** Texto parcialmente estructurado pero con contenido incoherente  
  **Esperado:** Contrato estructurado correctamente  
  **Análisis:** El modelo muestra una ligera mejora en la estructura del texto, pero introduce contenido irrelevante y mezcla idiomas, lo que indica falta de especialización.  

- **Entrada:** Parámetros contrato 2  
  **Generado:** Texto con estructura similar pero con datos incorrectos  
  **Esperado:** Contrato coherente  
  **Análisis:** El modelo reproduce el patrón general del contrato, pero mezcla información de distintos ejemplos y no respeta los parámetros de entrada.  

- **Entrada:** Parámetros contrato 3  
  **Generado:** Texto incoherente con datos inventados  
  **Esperado:** Contrato correcto  
  **Análisis:** Aunque se observa intento de estructura, el modelo sigue generando contenido fuera de contexto y no generaliza correctamente.  

---

### Conclusión

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

### Pipeline completo
Contrato original → NER (encoder) → Parámetros → GPT-2 (decoder) → Contrato generado

---

### Limitaciones y mejoras

**Sesgos:**

El modelo hereda sesgos del preentrenamiento en inglés, lo que provoca mezcla de idiomas.

**Limitaciones técnicas:**

- Bajo rendimiento en generación  
- Falta de coherencia  
- Dependencia de pocos datos  

**Mejoras propuestas:**

- Ampliar dataset con contratos reales  
- Usar modelos preentrenados en español  
- Aplicar técnicas RAG (Recuperación Aumentada)  
- Aplicar enmascaramiento del prompt (ya implementado parcialmente)  
- Afinar parámetros de generación  

---

### Ejemplos de uso del pipeline

**Entrada:**

Texto de contrato de alquiler  

**Salida del encoder:**

- ARRENDADOR: Juan Pérez  
- ARRENDATARIO: María López  

**Salida del decoder:**

Contrato generado en lenguaje natural  

---

### Conclusión final

El sistema demuestra correctamente el uso de LLMs en un pipeline completo de NLP y refleja mejoras tras la aplicación de técnicas como el enmascaramiento del prompt y la simplificación del output.

No obstante, sigue presentando limitaciones significativas en la calidad de generación debido al tamaño reducido del dataset, por lo que sería necesario mejorar tanto los datos como el modelo para un uso en producción.
