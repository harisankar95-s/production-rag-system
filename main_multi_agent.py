from src.rag.retriever import load_vectorstore, get_multi_query_retriever
from src.multi_agent.supervisor import build_supervisor
from src.utils.rag_components import init_rag_components
from src.config import config
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

import logging
from src.utils.utils import get_embedding_model, get_rerank_model, get_llm, load_json
import os
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)


if __name__ == "__main__":
    embedding_model = get_embedding_model()
    rerank_model = get_rerank_model()
    vector_store = load_vectorstore(embedding_model)
    llm = get_llm()
    retriever = get_multi_query_retriever(vector_store, llm)

    init_rag_components(llm, retriever, rerank_model)
    doc_details = load_json(os.path.join(config.data_dir, 'document_summaries.json'))

    checkpointer = MemorySaver()
    graph = build_supervisor(llm, doc_details, checkpointer=checkpointer)

    thread_id = str(uuid.uuid4())
    run_config = {"configurable": {"thread_id": thread_id}}

    def get_response(question):
        result = graph.invoke(
            {"query": question, "messages": [HumanMessage(content=question)]},
            run_config
        )
        if llm.using_fallback:
            print("Note: Response generated using local fallback model due to API unavailability. Quality may vary.")
        if result.get("final_output"):
            print("Agent:", result["final_output"])
        elif result.get("answer"):
            print("Agent:", result["answer"])
        elif result.get("web_results"):
            print("Agent:", result["web_results"])

    while True:
        question = input("You: ")
        if not question.strip():
            continue
        if question.lower() in ["exit", "quit"]:
            print("Ending session.")
            break
        get_response(question)