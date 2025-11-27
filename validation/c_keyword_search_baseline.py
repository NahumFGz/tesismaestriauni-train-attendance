#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Línea base de recuperación usando keyword search simple.
Genera el mismo formato que contexto_qdrant.json para comparación.
"""

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def normalizar_texto(texto: str) -> str:
    """Normaliza texto para búsqueda: minúsculas, sin acentos, solo alphanuméricas"""
    # Convertir a minúsculas
    texto = texto.lower()
    # Remover acentos
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn"
    )
    # Solo mantener letras, números y espacios
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    # Normalizar espacios
    texto = " ".join(texto.split())
    return texto


def extraer_keywords(query: str) -> List[str]:
    """Extrae keywords relevantes de la consulta"""
    query_norm = normalizar_texto(query)

    # Palabras comunes a filtrar
    stop_words = {
        "el",
        "la",
        "los",
        "las",
        "de",
        "del",
        "en",
        "y",
        "a",
        "que",
        "fue",
        "cual",
        "cuales",
        "dame",
        "documentos",
        "asistencia",
        "congreso",
        "fueron",
        "cuantas",
        "hay",
        "tienes",
        "disponibles",
        "me",
        "puedes",
        "dar",
        "mostrar",
        "ver",
        "buscar",
    }

    # Dividir en palabras y filtrar stop words
    palabras = [p for p in query_norm.split() if p not in stop_words and len(p) > 2]

    return palabras


def buscar_keywords_en_documento(doc: Dict[str, Any], keywords: List[str]) -> float:
    """Calcula score de coincidencia de keywords en un documento"""
    if not keywords:
        return 0.0

    # Campos donde buscar (texto completo del documento)
    texto_completo = (
        f"{doc.get('fecha_larga', '')} {doc.get('legislatura', '')} "
        f"{doc.get('periodo_congreso', '')} {doc.get('periodo_anual', '')} "
        f"{doc.get('fecha_corta', '')} {doc.get('hora', '')}"
    )

    texto_norm = normalizar_texto(texto_completo)

    score = 0.0
    for keyword in keywords:
        if keyword in texto_norm:
            # Peso mayor si la keyword aparece como palabra completa
            if f" {keyword} " in f" {texto_norm} ":
                score += 2.0
            else:
                score += 1.0

    return score / len(keywords)  # Normalizar por número de keywords


def keyword_search(documentos: List[Dict[str, Any]], query: str, top_k: int = 10) -> List[str]:
    """
    Realiza búsqueda por keywords y retorna contextos en formato string
    """
    keywords = extraer_keywords(query)

    if not keywords:
        return []

    # Calcular scores para todos los documentos
    doc_scores = []
    for doc in documentos:
        score = buscar_keywords_en_documento(doc, keywords)
        if score > 0:  # Solo considerar documentos con alguna coincidencia
            doc_scores.append((doc, score))

    # Ordenar por score descendente
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    # Tomar los top_k documentos y convertir a formato de contexto
    contextos = []
    for doc, score in doc_scores[:top_k]:
        contexto = (
            f"Asistencia del {doc['fecha_larga']} – {doc['legislatura']}.\n"
            f"Congreso {doc['periodo_congreso']} | "
            f"Periodo anual {doc['periodo_anual']}.\n"
            f"URL: {doc['url']}"
        )
        contextos.append(contexto)

    return contextos


def main():
    """Función principal que genera el archivo keyword_search.json"""

    # Rutas de archivos
    DATA_PATH = Path("../data/attendance_docs_enriched.json")
    PREGUNTAS_PATH = Path("./testset/preguntas_contexto_esperado.json")
    OUTPUT_DIR = Path("./testset")
    OUTPUT_PATH = OUTPUT_DIR / "keyword_search.json"

    print("🔄 Cargando datos de asistencia...")

    # Cargar documentos de asistencia
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            documentos = json.load(f)
        print(f"✅ Cargados {len(documentos)} documentos de asistencia")
    except Exception as e:
        print(f"❌ Error cargando documentos: {e}")
        return

    # Cargar preguntas de prueba
    try:
        with open(PREGUNTAS_PATH, "r", encoding="utf-8") as f:
            preguntas_contexto = json.load(f)
        print(f"✅ Cargadas preguntas de {len(preguntas_contexto)} tipos")
    except Exception as e:
        print(f"❌ Error cargando preguntas: {e}")
        return

    print("🔍 Generando contextos con keyword search...")

    # Generar resultado en el mismo formato que contexto_qdrant.json
    resultado_final = {}

    total_preguntas = 0
    for tipo, items in preguntas_contexto.items():
        print(f"  📋 Procesando tipo: {tipo} ({len(items)} preguntas)")
        resultado_final[tipo] = []

        for item in items:
            query = item.get("query", "").strip()
            if not query:
                continue

            # Realizar búsqueda por keywords
            contextos = keyword_search(documentos, query, top_k=10)

            resultado_final[tipo].append({"query": query, "context": contextos})

            total_preguntas += 1

    # Crear directorio de salida si no existe
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Guardar resultado
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(resultado_final, f, ensure_ascii=False, indent=2)

        print(f"✅ Archivo guardado en: {OUTPUT_PATH}")
        print(f"📊 Total de preguntas procesadas: {total_preguntas}")

        # Mostrar estadísticas por tipo
        for tipo, items in resultado_final.items():
            total_contextos = sum(len(item["context"]) for item in items)
            promedio_contextos = total_contextos / len(items) if items else 0
            print(
                f"  📈 {tipo}: {len(items)} preguntas, "
                f"{promedio_contextos:.1f} contextos promedio"
            )

    except Exception as e:
        print(f"❌ Error guardando archivo: {e}")


if __name__ == "__main__":
    main()
