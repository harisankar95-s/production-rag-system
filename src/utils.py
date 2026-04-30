import logging
from src.config import config
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


def get_embedding_model():
    logger.info(f'loading embedding model {config.embedding_model}')
    embedding_model = HuggingFaceEmbeddings(model_name= config.embedding_model)
    logger.info(f'{config.embedding_model} model loaded')
    return embedding_model

def print_retrieved_chunks(docs):
    print("\n========== Retrieved Chunks ==========")
    for i, doc in enumerate(docs):
        print(f"\n--- Chunk {i+1} | Page {doc.metadata.get('page', 'N/A')} ---")
        print(doc.page_content[:500])
        print("-" * 40)
    print("======================================\n")

def get_rerank_model():
    logger.info(f"Loading rerank model: {config.rerank_model}")
    model = CrossEncoder(config.rerank_model)
    logger.info("Rerank model loaded")
    return model

