from src.rag.ingestion import ingest
from src.rag.retriever import load_vectorstore, get_multi_query_retriever
from src.agent.graph import build_graph
from src.agent.tools import init_rag_components
from src.config import config
from langchain_core.messages import HumanMessage

import logging
from src.utils import get_embedding_model, get_rerank_model, get_llm,load_json
import os

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

    questions = [
        "Who won the IPL 2025? and what is difference betwwen supervised and unsupervised learning?",

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