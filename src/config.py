import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # LLM
    ollama_model: str        = os.getenv("OLLAMA_MODEL", "llama3.2")
    ollama_base_url: str     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    embedding_model: str     = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    chroma_persist_dir: str  = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name: str     = os.getenv("COLLECTION_NAME", "rag_collection")

    chunk_size: int          = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int       = int(os.getenv("CHUNK_OVERLAP", "50"))

    retrieval_k: int         = int(os.getenv("RETRIEVAL_K", "3"))

    data_dir: str            = os.getenv("DATA_DIR", "./data")

config = Config()