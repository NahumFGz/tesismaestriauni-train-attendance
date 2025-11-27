#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Línea base de recuperación usando TF-IDF con similitud coseno.
Genera el mismo formato que contexto_qdrant.json para comparación.
"""

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


def preprocesar_query(query: str) -> str:
    """Preprocesa una consulta para TF-IDF"""
    return normalizar_texto(query)


def crear_corpus_documentos(documentos: List[Dict[str, Any]]) -> List[str]:
    """Crea el corpus de texto para TF-IDF a partir de los documentos"""
    corpus = []
    for doc in documentos:
        # Combinar campos relevantes del documento
        texto_completo = (
            f"{doc.get('fecha_larga', '')} {doc.get('legislatura', '')} "
            f"{doc.get('periodo_congreso', '')} {doc.get('periodo_anual', '')} "
            f"{doc.get('fecha_corta', '')} {doc.get('hora', '')} "
            f"{doc.get('sesion', '')}"
        )
        texto_normalizado = normalizar_texto(texto_completo)
        corpus.append(texto_normalizado)

    return corpus


class TFIDFRetriever:
    """Clase para recuperación de documentos usando TF-IDF"""

    def __init__(self, documentos: List[Dict[str, Any]]):
        self.documentos = documentos
        self.corpus = crear_corpus_documentos(documentos)

        # Configurar vectorizador TF-IDF
        # Usar palabras comunes en español como stop words
        spanish_stop_words = {
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
            "es",
            "son",
            "por",
            "para",
            "con",
            "su",
            "sus",
            "se",
            "le",
            "lo",
            "un",
            "una",
            "unos",
            "unas",
            "al",
        }

        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Limitar vocabulario
            min_df=2,  # Mínimo 2 documentos para incluir término
            max_df=0.8,  # Máximo 80% de documentos
            stop_words=list(spanish_stop_words),
            ngram_range=(1, 2),  # Unigramas y bigramas
            lowercase=True,
            token_pattern=r"\b[a-z][a-z0-9]+\b",  # Solo tokens que empiecen con letra
        )

        print("🔄 Entrenando vectorizador TF-IDF...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        print(
            f"✅ Vectorizador entrenado. Vocabulario: {len(self.vectorizer.vocabulary_)} términos"
        )

    def search(self, query: str, top_k: int = 10) -> List[str]:
        """
        Busca documentos relevantes usando TF-IDF y similitud coseno
        """
        # Preprocesar query
        query_normalizada = preprocesar_query(query)

        if not query_normalizada.strip():
            return []

        # Vectorizar query
        query_vector = self.vectorizer.transform([query_normalizada])

        # Calcular similitud coseno
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Obtener índices de documentos más similares
        top_indices = similarities.argsort()[-top_k:][::-1]  # Ordenar descendente

        # Filtrar documentos con similitud > 0
        contextos = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Solo documentos con alguna similitud
                doc = self.documentos[idx]
                contexto = (
                    f"Asistencia del {doc['fecha_larga']} – {doc['legislatura']}.\n"
                    f"Congreso {doc['periodo_congreso']} | "
                    f"Periodo anual {doc['periodo_anual']}.\n"
                    f"URL: {doc['url']}"
                )
                contextos.append(contexto)

        return contextos


def main():
    """Función principal que genera el archivo tf_idf.json"""

    # Rutas de archivos
    DATA_PATH = Path("../data/attendance_docs_enriched.json")
    PREGUNTAS_PATH = Path("./testset/preguntas_contexto_esperado.json")
    OUTPUT_DIR = Path("./testset")
    OUTPUT_PATH = OUTPUT_DIR / "tf_idf.json"

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

    # Inicializar retriever TF-IDF
    try:
        retriever = TFIDFRetriever(documentos)
    except Exception as e:
        print(f"❌ Error inicializando TF-IDF retriever: {e}")
        return

    print("🔍 Generando contextos con TF-IDF...")

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

            # Realizar búsqueda con TF-IDF
            contextos = retriever.search(query, top_k=10)

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

        # Mostrar información del vocabulario TF-IDF
        print(f"\n📚 Información del modelo TF-IDF:")
        print(f"  🔤 Tamaño del vocabulario: {len(retriever.vectorizer.vocabulary_)}")
        print(f"  📄 Documentos procesados: {retriever.tfidf_matrix.shape[0]}")
        print(f"  🎯 Características por documento: {retriever.tfidf_matrix.shape[1]}")

    except Exception as e:
        print(f"❌ Error guardando archivo: {e}")


if __name__ == "__main__":
    main()
