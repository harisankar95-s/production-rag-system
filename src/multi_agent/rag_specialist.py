from src.rag.pipeline import ask
from src.utils.rag_components import _rag_components
from src.multi_agent.state import MultiAgentState
from langgraph.graph import StateGraph, START, END

def rag_node(state: MultiAgentState):
    query = state['query']
    answer, chunks, top_score = ask(
        _rag_components['llm'],
        _rag_components['retriever'],
        _rag_components['rerank_model'],
        query)
    return {"answer": answer, "chunks": chunks}

def build_rag_specialist():
    graph = StateGraph(MultiAgentState)
    graph.add_node("rag", rag_node)
    graph.add_edge(START, "rag")
    graph.add_edge("rag", END)
    return graph.compile()