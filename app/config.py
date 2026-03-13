"""Application configuration loaded from environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"
STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# RAG settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K = 15
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim vectors, local, fast
