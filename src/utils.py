import logging
from src.config import config
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def get_embedding_model():
    logger.info(f'loading embedding model {config.embedding_model}')
    embedding_model = HuggingFaceEmbeddings(model_name= config.embedding_model)
    logger.info(f'{config.embedding_model} model loaded')
    return embedding_model

