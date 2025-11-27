#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Línea base de recuperación usando BM25 (Best Matching 25).
Genera el mismo formato que contexto_qdrant.json para comparación.
BM25 es una función de ranking probabilística basada en TF-IDF mejorada.
"""

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
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


def tokenizar(texto: str) -> List[str]:
    """Tokeniza el texto en palabras individuales"""
    texto_normalizado = normalizar_texto(texto)

    # Stop words en español
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
        "como",
        "este",
        "esta",
        "estos",
        "estas",
        "ese",
        "esa",
        "esos",
        "esas",
        "aquel",
        "aquella",
        "aquellos",
        "aquellas",
        "todo",
        "toda",
        "todos",
        "todas",
        "otro",
        "otra",
        "otros",
        "otras",
    }

    # Tokenizar y filtrar
    tokens = [
        token for token in texto_normalizado.split() if len(token) > 2 and token not in stop_words
    ]

    return tokens


def crear_corpus_documentos(documentos: List[Dict[str, Any]]) -> List[str]:
    """Crea el corpus de texto para BM25 a partir de los documentos"""
    corpus = []
    for doc in documentos:
        # Combinar campos relevantes del documento
        texto_completo = (
            f"{doc.get('fecha_larga', '')} {doc.get('legislatura', '')} "
            f"{doc.get('periodo_congreso', '')} {doc.get('periodo_anual', '')} "
            f"{doc.get('fecha_corta', '')} {doc.get('hora', '')} "
            f"{doc.get('sesion', '')}"
        )
        corpus.append(texto_completo)

    return corpus


class BM25Retriever:
    """
    Implementación de BM25 (Best Matching 25) para recuperación de documentos.
    BM25 es una función de ranking probabilística que mejora TF-IDF.
    """

    def __init__(self, documentos: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        """
        Inicializa el retriever BM25

        Args:
            documentos: Lista de documentos
            k1: Parámetro que controla la saturación de frecuencia de términos (típicamente 1.2-2.0)
            b: Parámetro que controla el efecto de la longitud del documento (0-1)
        """
        self.documentos = documentos
        self.corpus = crear_corpus_documentos(documentos)
        self.k1 = k1
        self.b = b

        print("🔄 Inicializando BM25...")

        # Tokenizar todos los documentos
        self.doc_tokens = [tokenizar(doc) for doc in self.corpus]

        # Calcular estadísticas del corpus
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)  # Longitud promedio
        self.N = len(self.documentos)  # Número total de documentos

        # Construir vocabulario y calcular document frequency (df)
        self.vocabulary = set()
        self.doc_freq = defaultdict(int)  # Frecuencia de documento para cada término

        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.vocabulary.add(token)
                self.doc_freq[token] += 1

        print(f"✅ BM25 inicializado:")
        print(f"  📄 Documentos: {self.N}")
        print(f"  🔤 Vocabulario: {len(self.vocabulary)} términos únicos")
        print(f"  📏 Longitud promedio de documento: {self.avgdl:.1f} tokens")
        print(f"  ⚙️  Parámetros: k1={self.k1}, b={self.b}")

    def _calculate_idf(self, term: str) -> float:
        """Calcula el IDF (Inverse Document Frequency) para un término"""
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0

        # IDF de BM25: log((N - df + 0.5) / (df + 0.5))
        idf = math.log((self.N - df + 0.5) / (df + 0.5))
        return max(0.0, idf)  # Asegurar que no sea negativo

    def _calculate_bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calcula el score BM25 para un documento dado una query"""
        doc_tokens = self.doc_tokens[doc_idx]
        doc_length = self.doc_lengths[doc_idx]

        # Contar frecuencias de términos en el documento
        term_freq = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in self.vocabulary:
                continue

            tf = term_freq.get(term, 0)
            if tf == 0:
                continue

            # IDF del término
            idf = self._calculate_idf(term)

            # Componente TF de BM25
            tf_component = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
            )

            # Score BM25 = IDF * TF_component
            score += idf * tf_component

        return score

    def search(self, query: str, top_k: int = 10) -> List[str]:
        """
        Busca documentos relevantes usando BM25

        Args:
            query: Consulta de búsqueda
            top_k: Número máximo de documentos a retornar

        Returns:
            Lista de contextos formateados
        """
        # Tokenizar query
        query_tokens = tokenizar(query)

        if not query_tokens:
            return []

        # Calcular scores BM25 para todos los documentos
        doc_scores = []
        for doc_idx in range(self.N):
            score = self._calculate_bm25_score(query_tokens, doc_idx)
            if score > 0:  # Solo considerar documentos con score positivo
                doc_scores.append((doc_idx, score))

        # Ordenar por score descendente
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Tomar los top_k documentos y convertir a formato de contexto
        contextos = []
        for doc_idx, score in doc_scores[:top_k]:
            doc = self.documentos[doc_idx]
            contexto = (
                f"Asistencia del {doc['fecha_larga']} – {doc['legislatura']}.\n"
                f"Congreso {doc['periodo_congreso']} | "
                f"Periodo anual {doc['periodo_anual']}.\n"
                f"URL: {doc['url']}"
            )
            contextos.append(contexto)

        return contextos


def main():
    """Función principal que genera el archivo bm25.json"""

    # Rutas de archivos
    DATA_PATH = Path("../data/attendance_docs_enriched.json")
    PREGUNTAS_PATH = Path("./testset/preguntas_contexto_esperado.json")
    OUTPUT_DIR = Path("./testset")
    OUTPUT_PATH = OUTPUT_DIR / "bm25.json"

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

    # Inicializar retriever BM25
    try:
        retriever = BM25Retriever(documentos, k1=1.5, b=0.75)
    except Exception as e:
        print(f"❌ Error inicializando BM25 retriever: {e}")
        return

    print("🔍 Generando contextos con BM25...")

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

            # Realizar búsqueda con BM25
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

        # Mostrar información del modelo BM25
        print(f"\n📚 Información del modelo BM25:")
        print(f"  🔤 Tamaño del vocabulario: {len(retriever.vocabulary)}")
        print(f"  📄 Documentos procesados: {retriever.N}")
        print(f"  📏 Longitud promedio: {retriever.avgdl:.1f} tokens")
        print(f"  ⚙️  Parámetros: k1={retriever.k1}, b={retriever.b}")

    except Exception as e:
        print(f"❌ Error guardando archivo: {e}")


if __name__ == "__main__":
    main()
