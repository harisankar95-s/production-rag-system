from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage,SystemMessage
from src.multi_agent.state import MultiAgentState
from src.multi_agent.rag_specialist import build_rag_specialist
from src.multi_agent.web_specialist import build_web_specialist
from src.multi_agent.validator import create_validator_node
from src.multi_agent.prompts import SUPERVISOR_PROMPT,SYNTHESIS_PROMPT
import json
from src.utils.utils import extract_content
from langgraph.types import Send
from src.utils.timer import timer, reset_timings, print_breakdown

def create_supervisor_node(llm, doc_summaries):
    def supervisor_node(state: MultiAgentState):
        query = state['query']
        prompt = SUPERVISOR_PROMPT.format(doc_summaries=doc_summaries)
        reset_timings()
        with timer("supervisor_llm_call"):
            response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=query)])
        try:
            if isinstance(response.content, list):
                content_text = response.content[0].get('text', '')
            else:
                content_text = response.content
            result = json.loads(content_text)
            route = result.get("route", "rag")
            rag_query = result.get("rag_query", query)
            web_query = result.get("web_query", query)
        except Exception as e:
            print(f"PARSING FAILED: {e}")
            route = "rag"
            rag_query = query
            web_query = query
        return {
    "route": route,
    "rag_query": rag_query,
    "web_query": web_query,
    "answer": "",        
    "web_results": "",   
    "chunks": [],        
    "verdict": "",       
    "final_output": ""   
}
    return supervisor_node
    
def call_rag_specialist(state: MultiAgentState):
    rag_specialist = build_rag_specialist()
    result = rag_specialist.invoke({"query": state["rag_query"]})
    return {"answer": result["answer"], "chunks": result["chunks"]}

def call_web_specialist(state: MultiAgentState):
    web_specialist = build_web_specialist()
    result = web_specialist.invoke({"query": state["web_query"]})
    return {"web_results": result["web_results"]}

def should_route(state: MultiAgentState):
    route = state["route"]
    if route == "rag":
        return [Send("rag", state)]
    elif route == "web":
        return [Send("web", state)]
    elif route == "both":
        return [Send("rag", state), Send("web", state)]
    
def create_final_node(llm):
    def final_node(state: MultiAgentState):
        answer = state.get("answer", "")
        web_results = state.get("web_results", "")
        verdict = state.get("verdict", "supported")
        rag_query = state.get("rag_query", "")
        web_query = state.get("web_query", "")

        if not answer and not web_results:
            print_breakdown()
            return {"final_output": "Could not retrieve information from any source."}

        if not web_results:
            if verdict == "not_supported":
                print_breakdown()
                return {"final_output": f"Could not find a grounded answer for '{rag_query}' in the documents."}
            print_breakdown()
            return {"final_output": answer}

        if not answer:
            print_breakdown()
            return {"final_output": f"From web search: {web_results}"}

        if verdict == "not_supported":
            print_breakdown()
            return {"final_output": (
                f"RAG could not find a grounded answer for '{rag_query}' in the documents.\n\n"
                f"Web search result for '{web_query}':\n{web_results}"
            )}

        prompt = SYNTHESIS_PROMPT.format(answer=answer, web_results=web_results)
        with timer("final_node_llm_call"):
            response = llm.invoke(prompt)
        if isinstance(response.content, list):
            final_text = response.content[0].get('text', '')
        else:
            final_text = response.content
        print_breakdown()
        return {"final_output": final_text}

    return final_node

def merge_node(state: MultiAgentState):
    return {}
def should_validate(state: MultiAgentState):
    if state["route"] == "web":
        return "final"
    return "validator"
    
def build_supervisor(llm, doc_summaries, checkpointer=None):
    supervisor_node = create_supervisor_node(llm, doc_summaries)
    validator_node = create_validator_node(llm)
    final_node = create_final_node(llm)
    
    graph = StateGraph(MultiAgentState)
    
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag", call_rag_specialist)
    graph.add_node("web", call_web_specialist)
    graph.add_node("merge", merge_node)
    graph.add_node("validator", validator_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", should_route)
    graph.add_edge("rag", "merge")
    graph.add_edge("web", "merge")
    graph.add_conditional_edges("merge", should_validate, {
    "validator": "validator",
    "final": "final"})
    graph.add_edge("validator", "final")
    graph.add_edge("final", END)
    return graph.compile(checkpointer=checkpointer)