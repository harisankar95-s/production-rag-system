from src.ingestion import ingest
from src.retriever import load_vectorstore, get_multi_query_retriever
from src.config import config
from src.agent import init_rag_components, build_graph
from langchain_core.messages import HumanMessage

import logging
from src.utils import get_embedding_model, get_rerank_model, get_llm

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

    graph = build_graph(llm)

    # question = "What is the latest version of LangGraph released in 2025?"

    # response = graph.invoke({"messages": [HumanMessage(content=question)]})
    # final = response['messages'][-1].content
    # if isinstance(final, list):
    #     print(final[0]['text'])
    # else:
    #     print(final)
    questions = [
        "What is the moral of the story about the hare and the tortoise?",
        "What is gradient descent?",
        "What is the latest version of Python released in 2025?",
        "Who won the IPL 2025?",
        "What is overfitting in machine learning?",
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"{'='*60}")
        response = graph.invoke({"messages": [HumanMessage(content=question)]})
        final = response['messages'][-1].content
        if isinstance(final, list):
            print(final[0]['text'])
        else:
            print(final)