import os
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langfuse.langchain import CallbackHandler

from src.rag.retriever import load_vectorstore, get_multi_query_retriever
from src.multi_agent.supervisor import build_supervisor
from src.utils.rag_components import init_rag_components
from src.utils.cache import SemanticCache
from src.utils.guardrails import sanitize_input
from src.utils.utils import get_embedding_model, get_rerank_model, get_llm, load_json
from src.config import config

embedding_model = get_embedding_model()
rerank_model = get_rerank_model()
vector_store = load_vectorstore(embedding_model)
llm = get_llm()
retriever = get_multi_query_retriever(vector_store, llm)

init_rag_components(llm, retriever, rerank_model)
doc_details = load_json(os.path.join(config.data_dir, 'document_summaries.json'))

checkpointer = MemorySaver()
graph = build_supervisor(llm, doc_details, checkpointer=checkpointer)
cache = SemanticCache(embedding_model)

def get_answer(question: str,thread_id: str) -> str:
    run_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
    try:
        cached = cache.get(question)
        if cached:
            return cached

        if sanitize_input(question) is None:
            return "I can't process that request."

        langfuse_handler = CallbackHandler()

        result = graph.invoke(
            {"query": question, "messages": [HumanMessage(content=question)]},
            {**run_config, "callbacks": [langfuse_handler]}
        )

        answer = None
        route = result.get("route", "unknown")

        if result.get("final_output"):
            answer = result["final_output"]
        elif result.get("answer"):
            answer = result["answer"]
        elif result.get("web_results"):
            answer = result["web_results"]

        if answer:
            cache.set(question, answer, route)
            return answer

        return "I could not find an answer."

    except Exception as e:
        return f"Error: {e}"