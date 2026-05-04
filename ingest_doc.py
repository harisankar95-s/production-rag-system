import os
import logging
from src.ingestion import ingest
from src.utils import get_embedding_model,get_llm
from src.config import config

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

if __name__ == "__main__":
    embedding_model = get_embedding_model()
    llm = get_llm()
    ingest(config.data_dir, embedding_model,llm)