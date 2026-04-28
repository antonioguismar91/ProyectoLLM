# pipeline.py
# Caso 1: Sector Legal - Análisis de Contratos
# Autores: [Tu nombre], Lucas, Alejandro

# ============================================================
# IMPORTS
# ============================================================
# from src.representacion.predict import extract_entities
# from src.generacion.generate import generate_contract

# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def run_pipeline(texto_contrato):
    
    print("=" * 60)
    print("PIPELINE - ANÁLISIS Y GENERACIÓN DE CONTRATOS")
    print("=" * 60)
    
    # PASO 1: Mostrar texto de entrada
    print("\n📄 TEXTO DE ENTRADA:")
    print(texto_contrato)
    
    # PASO 2: Encoder → Extracción de entidades (modelo de Lucas)
    print("\n🔍 ENTIDADES EXTRAÍDAS:")
    # entidades = extract_entities(texto_contrato)  # descomentar cuando Lucas tenga su modelo
    entidades = {
        "ARRENDADOR": "pendiente modelo Lucas",
        "ARRENDATARIO": "pendiente modelo Lucas",
        "RENTA": "pendiente modelo Lucas",
        "DIRECCION": "pendiente modelo Lucas",
        "FECHA_INICIO": "pendiente modelo Lucas",
        "DURACION": "pendiente modelo Lucas"
    }
    print(entidades)
    
    # PASO 3: Decoder → Generación de borrador (modelo de Alejandro)
    print("\n📝 BORRADOR GENERADO:")
    # borrador = generate_contract(entidades)  # descomentar cuando Alejandro tenga su modelo
    borrador = "pendiente modelo Alejandro"
    print(borrador)
    
    print("\n" + "=" * 60)
    
    return entidades, borrador


# ============================================================
# EJEMPLOS DE PRUEBA
# ============================================================

if __name__ == "__main__":

    # Ejemplo 1
    contrato1 = """
    Juan Pérez arrienda a María López el piso en Calle Mayor 5 
    por 800 € mensuales desde el 1 de enero de 2024 por 12 meses.
    """

    # Ejemplo 2
    contrato2 = """
    Carlos Ruiz como arrendador cede a Ana García la vivienda 
    sita en Avenida Constitución 12, con una renta de 650 € 
    al mes a partir del 1 de marzo de 2024 durante 6 meses.
    """

    # Ejemplo 3
    contrato3 = """
    Pedro Sánchez arrienda a Laura Martínez el local en 
    Plaza España 3 por 1200 € desde el 15 de febrero 
    de 2024 por 24 meses.
    """

    print("\n📌 EJEMPLO 1")
    run_pipeline(contrato1)

    print("\n📌 EJEMPLO 2")
    run_pipeline(contrato2)

    print("\n📌 EJEMPLO 3")
    run_pipeline(contrato3)