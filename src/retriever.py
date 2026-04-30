from langchain_chroma import Chroma
import logging
from src.config import config
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class HybridRetriever(BaseRetriever):
    bm25_retriever: any
    vector_retriever: any
    k: int = 6

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        bm25_results = self.bm25_retriever.invoke(query)
        vector_results = self.vector_retriever.invoke(query)
        all_results = bm25_results + vector_results

        seen = set()
        unique_docs = []
        for doc in all_results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        return unique_docs[:self.k]

def load_vectorstore(embedding_model):
    logger.info(f"Loading {config.collection_name} vector store")
    vector_store = Chroma(embedding_function= embedding_model,collection_name=config.collection_name,persist_directory=config.chroma_persist_dir)
    logger.info(f"Loaded vector store ")
    return vector_store

def get_retriever(vector_store):
    logger.info("Initiating retrieving")
    retriever = vector_store.as_retriever(search_kwargs={"k": config.retrieval_k})
    logger.info("Retrieval completed ")
    return retriever

def get_bm25_retriever(vector_store):
    logger.info("Initiating BM25 retriever")
    data = vector_store.get()
    documents = data['documents']
    meta_data = data['metadatas']
    logger.info("Creating document objects from text chunks")
    docs = [Document(page_content=text, metadata=meta)
    for text, meta in zip(documents, meta_data)]
    logger.info("Creating BM25 retriever")
    bm25_retriever = BM25Retriever.from_documents(docs,k =config.retrieval_k)
    return  bm25_retriever

def get_hybrid_retriever(vectorstore):
    logger.info("Building hybrid retriever")
    bm25_retriever = get_bm25_retriever(vectorstore)
    vector_retriever = get_retriever(vectorstore)
    logger.info("Hybrid retriever ready")
    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        k=config.retrieval_k
    )





    

