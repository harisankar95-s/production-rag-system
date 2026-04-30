from src.ingestion import ingest
from src.retriever import load_vectorstore,get_retriever,get_hybrid_retriever
from src.pipeline import build_pipeline,ask
from src.config import config

import logging
from src.utils import get_embedding_model,get_rerank_model
from src.utils import print_retrieved_chunks

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
    retriever = get_hybrid_retriever(vector_store)
    llm = build_pipeline()

    question = "What is the difference between supervised and unsupervised learning?"

    answer,relavent_chunks  = ask(llm, retriever, rerank_model, question)

    print(answer)
    print_retrieved_chunks(relavent_chunks)


