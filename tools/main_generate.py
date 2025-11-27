from dotenv import load_dotenv

load_dotenv()

import os
import time
from typing import List, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import START, StateGraph
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient


# ── Embeddings y vector store con reintentos ─────────────────────────────────
def init_vector_store_with_retries(max_retries: int = 5, delay: int = 3) -> QdrantVectorStore:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"🔁 Intento {attempt} de conexión a Qdrant...")
            qdrant_client = QdrantClient(
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
            )
            vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name="attendance_docs",
                embedding=embeddings,
            )
            print("✅ Conexión a Qdrant exitosa")
            return vector_store
        except Exception as e:
            print(f"⚠️ Fallo al conectar a Qdrant: {e}")
            last_exception = e
            if attempt < max_retries:
                time.sleep(delay)

    raise RuntimeError(
        f"No se pudo conectar a Qdrant después de {max_retries} intentos"
    ) from last_exception


vector_store = init_vector_store_with_retries()

PROMPT = PromptTemplate.from_template(
    """Responde únicamente con la información exacta proporcionada en el contexto. 
No inventes ni completes información que no esté explícitamente presente. 
Incluye solo el asunto (si lo hay), la fecha o legislatura correspondiente y la URL del documento.

{context}

Pregunta: {question}

Respuesta:"""
)

llm = init_chat_model("gpt-4o-mini", model_provider="openai")


class RAGState(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: RAGState) -> RAGState:
    docs = vector_store.similarity_search(state["question"])
    return {"question": state["question"], "context": docs}


def generate(state: RAGState) -> RAGState:
    ctx = "\n\n".join(d.page_content for d in state["context"])
    messages = PROMPT.invoke({"question": state["question"], "context": ctx})
    response = llm.invoke(messages)
    return {"question": state["question"], "context": state["context"], "answer": response.content}


# Definir el grafo RAG
rag_graph = (
    StateGraph(RAGState)
    .add_node("retrieve", retrieve)
    .add_node("generate", generate)
    .add_edge(START, "retrieve")
    .add_edge("retrieve", "generate")
    .set_entry_point("retrieve")
    .set_finish_point("generate")
    .compile()
)

# ── MCP server ────────────────────────────────────────────────────────────────

mcp = FastMCP("AsistenciaParlamentariaRAG", host="0.0.0.0", port=8000)


class RAGResponse(TypedDict):
    answer: str


@mcp.tool()
async def consultar_asistencia(pregunta: str) -> RAGResponse:
    """Busca y retorna información sobre la asistencia parlamentaria según la consulta proporcionada.
    Esta herramienta permite consultar registros de asistencia parlamentaria, devolviendo:
    - La fecha o legislatura específica
    - La URL del documento oficial con la imagen del registro de asistencia

    Ejemplo de uso:
    - "¿Cuál fue la asistencia del 15 de marzo de 2024?"
    """
    try:
        result = await rag_graph.ainvoke({"question": pregunta})
        return {"answer": result["answer"]}
    except Exception as e:
        return {"answer": f"Ocurrió un error al buscar información: {str(e)}"}


class RangoFechasResponse(TypedDict):
    rango: str


@mcp.tool()
async def obtener_rango_asistencia() -> RangoFechasResponse:
    """Retorna el rango de fechas disponible para consultar asistencias parlamentarias."""
    return {
        "rango": "La información de asistencias está disponible desde enero de 2009 hasta marzo de 2025"
    }


# ── Ejecutar servidor MCP (comentado para pruebas) ────────────────────────────
# if __name__ == "__main__":
#     mcp.run(transport="streamable-http")

#! Para probar los tools
if __name__ == "__main__":
    import asyncio

    async def test_tools():
        pregunta = "¿Cuál fue la asistencia del octubre del 2022?"
        resultado = await consultar_asistencia(pregunta)
        print("🔍 Resultado:", resultado)

        resultado_rango = await obtener_rango_asistencia()
        print("🔍 Resultado:", resultado_rango)

    asyncio.run(test_tools())
