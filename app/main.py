"""
FastAPI application — REST API for the RAG AI Data Analyst.

Endpoints:
    POST /upload      — Upload and process documents
    POST /ask          — Ask a question about uploaded documents
    GET  /documents    — List uploaded documents
    DELETE /documents/{doc_id} — Remove a document and its vectors
"""

import uuid
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.config import UPLOAD_DIR, STATIC_DIR
from app.document_loader import load_document, SUPPORTED_EXTENSIONS
from app.rag_engine import semantic_chunk_text, store_chunks, delete_document, query

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG AI Data Analyst",
    description="Analista de Datos IA con RAG — No-Hallucination Protocol",
    version="1.0.0",
)

# Document registry (persisted as JSON)
REGISTRY_PATH = UPLOAD_DIR / "_registry.json"


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def _save_registry(registry: dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# API Models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document, extract text, chunk, embed, and store."""
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado: '{ext}'. Extensiones válidas: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # Save file to disk
    doc_id = str(uuid.uuid4())
    safe_name = f"{doc_id}{ext}"
    file_path = UPLOAD_DIR / safe_name

    content = await file.read()
    file_path.write_bytes(content)

    try:
        # Extract text
        text = load_document(str(file_path))
        if not text.strip():
            raise HTTPException(status_code=400, detail="El documento no contiene texto extraíble.")

        # Semantic (Adaptive) Chunking
        chunks = semantic_chunk_text(text, file.filename)
        num_chunks = store_chunks(chunks, doc_id)

        # Update registry
        registry = _load_registry()
        registry[doc_id] = {
            "filename": file.filename,
            "extension": ext,
            "size_bytes": len(content),
            "num_chunks": num_chunks,
            "file_path": str(file_path),
        }
        _save_registry(registry)

        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "num_chunks": num_chunks,
            "message": f"Documento '{file.filename}' procesado exitosamente con {num_chunks} fragmentos.",
        }

    except HTTPException:
        raise
    except Exception as e:
        # Clean up on failure
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error procesando documento: {str(e)}")


@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """Ask a question about the uploaded documents."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")

    result = query(req.question)
    return AskResponse(answer=result["answer"], sources=result["sources"])


@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    registry = _load_registry()
    docs = []
    for doc_id, info in registry.items():
        docs.append({
            "doc_id": doc_id,
            "filename": info["filename"],
            "extension": info["extension"],
            "size_bytes": info["size_bytes"],
            "num_chunks": info["num_chunks"],
        })
    return {"documents": docs}


@app.delete("/documents/{doc_id}")
async def remove_document(doc_id: str):
    """Remove a document and all its vectors from the store."""
    registry = _load_registry()
    if doc_id not in registry:
        raise HTTPException(status_code=404, detail="Documento no encontrado.")

    info = registry[doc_id]

    # Delete vectors
    deleted = delete_document(doc_id)

    # Delete file
    file_path = Path(info["file_path"])
    file_path.unlink(missing_ok=True)

    # Update registry
    del registry[doc_id]
    _save_registry(registry)

    return {
        "message": f"Documento '{info['filename']}' eliminado ({deleted} fragmentos removidos).",
        "doc_id": doc_id,
    }


# ---------------------------------------------------------------------------
# Static files & SPA fallback
# ---------------------------------------------------------------------------

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    """Serve the main HTML page."""
    return FileResponse(str(STATIC_DIR / "index.html"))
