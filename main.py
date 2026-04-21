from src.ingestion import ingest
from src.retriever import load_vectorstore,get_retriever
from src.pipeline import build_pipeline,ask
from src.config import config

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


if __name__ == "__main__":
    ingest(f"{config.data_dir}/sample.txt")

    vector_store = load_vectorstore()
    retriever = get_retriever(vector_store)
    qa_pipeline = build_pipeline(retriever)

    question = "What was the Quit India Movement?"

    answer = ask(qa_pipeline,question)

    print(answer)

