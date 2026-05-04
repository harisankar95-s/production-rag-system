from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from src.rag.pipeline import ask

_rag_components = {}

def init_rag_components(llm, retriever, rerank_model):
    _rag_components['llm'] = llm
    _rag_components['retriever'] = retriever
    _rag_components['rerank_model'] = rerank_model

@tool
def rag_search(query: str) -> str:
    """Search the uploaded documents to answer questions about their content.
    Use this when the question is about information that could be in the documents."""
    answer, chunks = ask(
        _rag_components['llm'],
        _rag_components['retriever'],
        _rag_components['rerank_model'],
        query
    )
    return answer

@tool
def web_search(query: str) -> str:
    """Search the web for current information not available in documents.
    Use this for recent events, general knowledge, or anything not document-specific."""
    search = DuckDuckGoSearchRun()
    return search.run(query)