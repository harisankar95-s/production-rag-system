import logging
from src.config import config
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM

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

def get_llm():
    logger.info(f"Loading LLM provider: {config.llm_provider}")
    if config.llm_provider == "ollama":
        llm = OllamaLLM(model=config.ollama_model, base_url=config.ollama_base_url)
    elif config.llm_provider == "gemini":
        llm = ChatGoogleGenerativeAI(model=config.gemini_model, google_api_key=config.google_api_key)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
    logger.info(f"LLM loaded: {config.llm_provider}")
    return llm


def get_eval_llm():
    logger.info(f"Loading eval LLM provider: {config.eval_llm_provider}")
    if config.eval_llm_provider == "ollama":
        llm = OllamaLLM(model=config.ollama_model, base_url=config.ollama_base_url)
    elif config.eval_llm_provider == "gemini":
        llm = ChatGoogleGenerativeAI(model=config.gemini_model, google_api_key=config.google_api_key)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.eval_llm_provider}")
    logger.info(f"Eval LLM loaded: {config.eval_llm_provider}")
    return llm



