import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    ollama_model: str        = os.getenv("OLLAMA_MODEL", "llama3.2")
    ollama_base_url: str     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    llm_provider: str        = os.getenv("LLM_PROVIDER", "ollama")
    eval_llm_provider: str   = os.getenv("EVAL_LLM_PROVIDER", "gemini")

    embedding_model: str     = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    chroma_persist_dir: str = os.getenv(
    "CHROMA_PERSIST_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db"))

    collection_name: str     = os.getenv("COLLECTION_NAME", "rag_collection")

    chunk_size: int          = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int       = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_n_chunks: int        = int(os.getenv("TOP_N_CHUNKS", "5"))

    retrieval_k: int         = int(os.getenv("RETRIEVAL_K", "10"))
    multi_query_variants: int    = int(os.getenv("MULTI_QUERY_VARIANTS", "3"))

    data_dir: str = os.getenv(
    "DATA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
)

    rerank_model: str        = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    google_api_key: str = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str        = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


config = Config()