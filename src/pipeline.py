from langchain_ollama import OllamaLLM
import logging
from src.config import config
from src.retriever import rerank

logger = logging.getLogger(__name__)


def build_pipeline():
    logger.info(f"Loading LLM: {config.ollama_model}")
    llm = OllamaLLM(model=config.ollama_model, base_url=config.ollama_base_url)
    logger.info("LLM loaded")
    return llm


def ask(llm, retriever, rerank_model, question):
    logger.info(f"Question: {question}")
    chunks = retriever.invoke(question)
    logger.info(f"Retrieved {len(chunks)} chunks")
    reranked_chunks = rerank(rerank_model, question, chunks)
    logger.info(f"Reranked to {len(reranked_chunks)} chunks")
    context = "\n\n".join([doc.page_content for doc in reranked_chunks])
    prompt = f"""Answer the question based only on the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""
    
    # Step 5: generate
    answer = llm.invoke(prompt)
    logger.info("Received LLM output")
    
    return answer, reranked_chunks