from langchain_chroma import Chroma
import logging
from src.config import config
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from src.utils.utils import extract_content
from src.utils.timer import timer

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
    with timer("bm25_retriever_init"):
         bm25_retriever = get_bm25_retriever(vectorstore)
    vector_retriever = get_retriever(vectorstore)
    logger.info("Hybrid retriever ready")
    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever,
        k=config.retrieval_k
    )
def rerank(model, query, chunks):
    logger.info(f"Re-ranking {len(chunks)} chunks")
    pairs = [(query, chunk.page_content) for chunk in chunks]
    scores = model.predict(pairs)
    chunk_score_pairs = list(zip(chunks, scores))
    sorted_pairs = sorted(chunk_score_pairs, key=lambda x: x[1], reverse=True)
    top_score = sorted_pairs[0][1]  # highest score after sorting
    reranked_chunks = [chunk for chunk, score in sorted_pairs]
    logger.info(f"Re-ranking complete, returning top {config.top_n_chunks} chunks")
    return reranked_chunks[:config.top_n_chunks],top_score

class MultiQueryHybridRetriever(BaseRetriever):
    hybrid_retriever: HybridRetriever
    llm: any
    n_variants: int = 3

    def _generate_variants(self, query: str) -> list:
        prompt = f"""Generate {self.n_variants} different versions of this question \
to improve document retrieval. Each version should approach the same \
information need from a different angle.
Return only the questions, one per line, no numbering or explanation.

Original question: {query}"""
        response = self.llm.invoke(prompt)
        lines = extract_content(response).strip().split('\n')
        variants = [l.strip() for l in lines if l.strip()]
        logger.info(f"Generated {len(variants)} query variants")
        return variants[:self.n_variants]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ):
        variants = self._generate_variants(query)
        all_queries = [query] + variants

        seen = set()
        unique_docs = []
        for q in all_queries:
            docs = self.hybrid_retriever.invoke(q)
            for doc in docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    unique_docs.append(doc)

        logger.info(f"Multi-query retrieved {len(unique_docs)} unique chunks from {len(all_queries)} queries")
        return unique_docs
    
def get_multi_query_retriever(vectorstore, llm):
    logger.info("Building multi-query hybrid retriever")
    hybrid = get_hybrid_retriever(vectorstore)
    return MultiQueryHybridRetriever(
        hybrid_retriever=hybrid,
        llm=llm,
        n_variants=config.multi_query_variants
    )






    

