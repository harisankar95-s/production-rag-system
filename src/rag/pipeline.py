import logging
from src.config import config
from src.rag.retriever import rerank
from src.rag.prompts import RAG_PROMPT
from src.utils.utils import extract_content
from src.utils.timer import timer

logger = logging.getLogger(__name__)

def ask(llm, retriever, rerank_model, question):
    logger.info(f"Question: {question}")
    with timer("multi_query_retrieval"):
        chunks = retriever.invoke(question)
    logger.info(f"Retrieved {len(chunks)} chunks")
    with timer("reranking"):
        reranked_chunks,top_score = rerank(rerank_model, question, chunks)
    logger.info(f"Reranked to {len(reranked_chunks)} chunks")
    context = "\n\n".join([f"Page {doc.metadata.get('page', 'N/A')}:\n{doc.page_content}" for doc in reranked_chunks])
    prompt = RAG_PROMPT.format(context=context, question=question)
    with timer("rag_generation"):
        answer = llm.invoke(prompt)
    if hasattr(answer, 'content'):
        answer = extract_content(answer)
    
    return answer, reranked_chunks,top_score

