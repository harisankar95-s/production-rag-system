from src.multi_agent.state import MultiAgentState
from langgraph.graph import StateGraph, START, END
from src.utils.search_components import web_search_run

def web_search_node(state: MultiAgentState):
    query = state['query']
    answer = web_search_run(query)
    return {"web_results": answer}  

def build_web_specialist():
    graph = StateGraph(MultiAgentState)
    graph.add_node("web_search", web_search_node)
    graph.add_edge(START, "web_search")
    graph.add_edge("web_search", END)
    return graph.compile()
