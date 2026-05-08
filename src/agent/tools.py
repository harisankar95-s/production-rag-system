from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from src.rag.pipeline import ask
from pydantic import BaseModel
from typing import List

class RAGSearchResult(BaseModel):
    answer: str
    source: str
    pages: List[int]
    confidence: str
    
    def __str__(self):
        return f"Answer: {self.answer}\nSource: {self.source}\nPages: {self.pages}\nConfidence: {self.confidence}"

_rag_components = {}

def init_rag_components(llm, retriever, rerank_model):
    _rag_components['llm'] = llm
    _rag_components['retriever'] = retriever
    _rag_components['rerank_model'] = rerank_model

@tool
def rag_search(query: str) -> RAGSearchResult:
    """Use this tool when the question is about topics covered in the document 
summaries provided in the system prompt. Check the document summaries first — 
if the question matches topics described there, use this tool.
Do not use this tool for current events, recent news, or topics clearly 
not mentioned in any document summary."""
    try :
        answer, chunks,top_score = ask(
            _rag_components['llm'],
            _rag_components['retriever'],
            _rag_components['rerank_model'],
            query
        )
        pages = sorted(set([
            chunk.metadata.get('page') 
            for chunk in chunks 
            if chunk.metadata.get('page') is not None
        ]))
        if top_score > 5:
            confidence = "high"
        elif top_score > 0:
            confidence = "medium"
        else:
            confidence = "low"
        result = RAGSearchResult(
            answer=answer,
            source=chunks[0].metadata.get('source', 'unknown'),
            pages=pages,
            confidence= confidence
        )
        return result
    except Exception as e:
        return f"RAG search failed: {str(e)}. Try web search instead."


@tool
def web_search(query: str) -> str:
    """Use this tool for questions about current events, recent news, live data,
    or information that would not be in any uploaded document.
    Use this as a fallback when the document summaries suggest the answer 
    is not in the available documents."""
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        return f"Web search is currently unavailable: {str(e)}. Answer from your own knowledge if possible, or inform the user that current information cannot be retrieved."