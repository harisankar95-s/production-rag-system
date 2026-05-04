import logging
from src.config import config
from src.rag.retriever import rerank
from src.rag.prompts import RAG_PROMPT

logger = logging.getLogger(__name__)

def ask(llm, retriever, rerank_model, question):
    logger.info(f"Question: {question}")
    chunks = retriever.invoke(question)
    logger.info(f"Retrieved {len(chunks)} chunks")
    reranked_chunks = rerank(rerank_model, question, chunks)
    logger.info(f"Reranked to {len(reranked_chunks)} chunks")
    context = "\n\n".join([f"Page {doc.metadata.get('page', 'N/A')}:\n{doc.page_content}" for doc in reranked_chunks])
    prompt = RAG_PROMPT.format(context=context, question=question)
    
    answer = llm.invoke(prompt)
    if hasattr(answer, 'content'):
        answer = answer.content
    logger.info("Received LLM output")
    
    return answer, reranked_chunks