from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
import logging
from src.config import config

logger = logging.getLogger(__name__)

def build_pipeline(retriever):
    logger.info("Loading LLM")
    llm = OllamaLLM(model= config.ollama_model)
    logger.info("LLM loaded")
    qa_chain = RetrievalQA.from_chain_type(retriever = retriever,llm = llm,return_source_documents=True)
    logger.info("qa_chain created")
    return qa_chain

def ask(qa_chain, question):
    logger.info(f"Passing question - {question}- to llm")
    answer = qa_chain.invoke(question)
    logger.info(f"Received llm output")
    return answer['result'],answer["source_documents"]


