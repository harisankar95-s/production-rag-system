from src.ingestion import ingest
from src.retriever import load_vectorstore,get_retriever
from src.pipeline import build_pipeline,ask
from src.config import config

import logging
from src.utils import get_embedding_model
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
    vector_store = load_vectorstore(embedding_model)
    retriever = get_retriever(vector_store)
    qa_pipeline = build_pipeline(retriever)

    question = "What is the difference between supervised and unsupervised learning?"

    answer,relavent_chunks  = ask(qa_pipeline,question)

    print(answer)
    print_retrieved_chunks(relavent_chunks)


