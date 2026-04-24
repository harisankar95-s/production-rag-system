from langchain_chroma import Chroma
import logging
from src.config import config


logger = logging.getLogger(__name__)

def load_vectorstore(embedding_model):
    logger.info(f"Loading {config.collection_name} vector store")
    vector_store = Chroma(embedding_function= embedding_model,collection_name=config.collection_name,persist_directory=config.chroma_persist_dir)
    logger.info(f"Loaded vector store ")
    return vector_store

def get_retriever(vector_store):
    logger.info("Initiating retrieving")
    retriever = vector_store.as_retriever(search_kwargs={"k": config.retrieval_k})
    logger.info("Retrieval completed ")
    return retriever

