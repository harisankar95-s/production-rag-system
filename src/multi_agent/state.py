from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages

class MultiAgentState(TypedDict):
    query: str
    answer: str
    chunks: List
    web_results: str
    verdict: str
    messages: Annotated[list, add_messages]
    route: str
    rag_query: str
    web_query: str
    final_output: str 