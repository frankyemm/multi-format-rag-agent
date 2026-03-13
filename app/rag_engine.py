"""
RAG Engine — Core pipeline for chunking, embedding, retrieval, and LLM generation.

Uses:
- RecursiveCharacterTextSplitter for chunking
- sentence-transformers (all-MiniLM-L6-v2, 384-dim) for local embeddings (CPU-only for compatibility)
- ChromaDB for persistent vector storage
- OpenAI GPT-4o-mini for answer generation
"""

import hashlib
import re
from typing import Optional

import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from app.config import (
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)
from app.prompt_builder import build_prompt


# ---------------------------------------------------------------------------
# Lazy singletons (initialized on first use)
# ---------------------------------------------------------------------------

_embedding_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.ClientAPI] = None
_openai_client: Optional[OpenAI] = None

COLLECTION_NAME = "documents"


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        # device="cpu" is forced to avoid CUDA kernel errors on incompatible GPUs
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _embedding_model


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_collection() -> chromadb.Collection:
    """Get or create the main document collection."""
    client = _get_chroma_client()
    # Adding robust HNSW parameters to avoid "ef/M too small" errors on tiny collections
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 128,
            "hnsw:M": 16
        },
    )


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split text into overlapping chunks for embedding.

    Args:
        text: Full extracted text from a document.
        source: Filename or identifier for citation.

    Returns:
        List of dicts with 'text', 'source', and 'id' keys.
    """
    chunks: list[dict] = []
    if not text.strip():
        return chunks

    # Simple recursive splitting by paragraphs, then by size
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 <= CHUNK_SIZE:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
        else:
            if current_chunk:
                chunks.append(_make_chunk(current_chunk, source))
            if len(para) > CHUNK_SIZE:
                for sub in _split_long_text(para):
                    chunks.append(_make_chunk(sub, source))
                current_chunk = ""
            else:
                current_chunk = para

    if current_chunk.strip():
        chunks.append(_make_chunk(current_chunk, source))

    return _add_overlap(chunks)


def semantic_chunk_text(text: str, source: str, threshold: float = 0.6) -> list[dict]:
    """
    Adaptive (Semantic) Chunking implementation.
    
    Instead of fixed character limits, this method:
    1. Splits text into sentences.
    2. Groups sentences that are semantically similar.
    3. Starts a new chunk when the 'topic' changes (similarity drops).
    """
    if not text.strip():
        return []

    # 1. Split into sentences (simple regex)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= 1:
        return [_make_chunk(text, source)]

    # 2. Get embeddings for each sentence
    embeddings = _get_embedding_model().encode(sentences)
    
    # 3. Calculate similarity between neighbors
    chunks = []
    current_sentences = [sentences[0]]
    
    for i in range(len(sentences) - 1):
        # Cosine similarity
        vec1 = embeddings[i]
        vec2 = embeddings[i+1]
        sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # If similarity is low, we found a 'breakpoint' (theme shift)
        # Also check for max chunk size to avoid infinite growth
        current_text = " ".join(current_sentences)
        if sim < threshold or len(current_text) > CHUNK_SIZE:
             chunks.append(_make_chunk(current_text, source))
             current_sentences = [sentences[i+1]]
        else:
             current_sentences.append(sentences[i+1])
             
    if current_sentences:
        chunks.append(_make_chunk(" ".join(current_sentences), source))
        
    return chunks


def _make_chunk(text: str, source: str) -> dict:
    chunk_id = hashlib.md5(f"{source}:{text[:100]}".encode()).hexdigest()
    return {"id": chunk_id, "text": text.strip(), "source": source}


def _split_long_text(text: str) -> list[str]:
    pieces: list[str] = []
    words = text.split()
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= CHUNK_SIZE:
            current = f"{current} {word}" if current else word
        else:
            if current:
                pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    return pieces


def _add_overlap(chunks: list[dict]) -> list[dict]:
    if len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_text = chunks[i - 1]["text"]
        overlap = prev_text[-CHUNK_OVERLAP:] if len(prev_text) > CHUNK_OVERLAP else prev_text
        new_text = f"{overlap}\n\n{chunks[i]['text']}"
        result.append({
            "id": chunks[i]["id"],
            "text": new_text,
            "source": chunks[i]["source"],
        })
    return result


# ---------------------------------------------------------------------------
# Embedding & storage
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def store_chunks(chunks: list[dict], doc_id: str) -> int:
    if not chunks:
        return 0

    collection = get_collection()
    texts = [c["text"] for c in chunks]
    # Include the source filename in the text to be embedded so semantic search can find it by name
    embed_inputs = [f"Archivo: {c['source']}\nContenido: {c['text']}" for c in chunks]
    embeddings = embed_texts(embed_inputs)

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": c["source"], "doc_id": doc_id} for c in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(chunks)


def delete_document(doc_id: str) -> int:
    collection = get_collection()
    results = collection.get(where={"doc_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])
    return len(results["ids"])


# ---------------------------------------------------------------------------
# Query & generation
# ---------------------------------------------------------------------------

def query(question: str) -> dict:
    collection = get_collection()

    if collection.count() == 0:
        return {
            "answer": "No hay documentos cargados. Por favor, sube al menos un documento para poder responder tus preguntas.",
            "sources": [],
            "context_chunks": [],
        }

    q_embedding = embed_texts([question])[0]

    # Retrieve top-K relevant chunks with defensive error handling
    try:
        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=min(8, collection.count()),  # Increased K to 8 for better coverage
            include=["documents", "metadatas", "distances"],
        )
    except RuntimeError as e:
        # Graceful fallback: if HNSW fails on a small collection, just get all docs raw
        if "RuntimeError" in str(e) or "ef or M" in str(e):
            results = collection.get(include=["documents", "metadatas"], limit=10)
            # Reformat get results to match query results structure temporarily
            results = {
                "documents": [results["documents"]],
                "metadatas": [results["metadatas"]]
            }
        else:
            raise e

    context_chunks: list[dict] = []
    sources: set[str] = set()
    
    # Check if we have results
    if results["documents"] and results["documents"][0]:
        for doc_text, metadata in zip(results["documents"][0], results["metadatas"][0]):
            src = metadata.get("source", "Desconocido")
            sources.add(src)
            # Add explicit source header to the text chunk to help LLM distinguish
            context_chunks.append({
                "text": f"--- FUENTE: {src} ---\n{doc_text}",
                "source": src
            })

    prompt = build_prompt(question, context_chunks)
    answer = _call_openai(prompt)

    return {
        "answer": answer,
        "sources": sorted(sources),
        "context_chunks": [{"text": c["text"][:200] + "...", "source": c["source"]} for c in context_chunks],
    }


def _call_openai(prompt: str) -> str:
    """Call OpenAI API to generate an answer using gpt-4o-mini."""
    if not OPENAI_API_KEY:
        return "⚠️ Error: No se ha configurado la clave de API de OpenAI. Agrega OPENAI_API_KEY en el archivo .env"

    client = _get_openai_client()

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un asistente técnico que responde basándose exclusivamente en documentos."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error al generar respuesta: {str(e)}"
