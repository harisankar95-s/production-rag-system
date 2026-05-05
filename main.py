from src.rag.ingestion import ingest
from src.rag.retriever import load_vectorstore, get_multi_query_retriever
from src.agent.graph import build_graph
from src.agent.tools import init_rag_components
from src.config import config
from langchain_core.messages import HumanMessage

import logging
from src.utils import get_embedding_model, get_rerank_model, get_llm,load_json
import os
from langchain_core.messages import AIMessage

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


    graph = build_graph(llm,doc_details)

    # questions = [
    #     "Who won the IPL 2025? and what is the difference between supervised and unsupervised learning?",
    #     "What is gradient descent and who is the current CEO of Google?",
    #     "Explain overfitting in machine learning and what is the latest iPhone model?",
    #     "What is the moral of the hare and tortoise story and who won the 2024 US election?",
    # ]


    # for message, metadata in graph.stream(
    #     {"messages": [HumanMessage(content=questions)]},
    #     stream_mode="messages"
    # ):
    #     if (isinstance(message, AIMessage) and 
    #         message.content and 
    #         not message.tool_calls and
    #         metadata.get("langgraph_node") == "agent"):
    #         content = message.content
    #         if isinstance(content, list):
    #             print(content[0].get('text', ''), end="", flush=True)
    #         else:
    #             print(content, end="", flush=True)
    # print()

    question = "Who won the IPL 2025? and what is the difference between supervised and unsupervised learning?"

    for message, metadata in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="messages"
    ):
        if (isinstance(message, AIMessage) and 
            message.content and 
            not message.tool_calls and
            metadata.get("langgraph_node") == "agent"):
            content = message.content
            if isinstance(content, list):
                print(content[0].get('text', ''), end="", flush=True)
            else:
                print(content, end="", flush=True)
    print()