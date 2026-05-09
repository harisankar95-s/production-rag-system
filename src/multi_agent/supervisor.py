from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage,SystemMessage
from src.multi_agent.state import MultiAgentState
from src.multi_agent.rag_specialist import build_rag_specialist
from src.multi_agent.web_specialist import build_web_specialist
from src.multi_agent.validator import create_validator_node
from src.multi_agent.prompts import SUPERVISOR_PROMPT,SYNTHESIS_PROMPT
import json
from src.utils.utils import extract_content

def create_supervisor_node(llm, doc_summaries):
    def supervisor_node(state: MultiAgentState):
        query = state['query']
        prompt = SUPERVISOR_PROMPT.format(doc_summaries=doc_summaries)
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
        return {"route": route, "rag_query": rag_query, "web_query": web_query}
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
    print(f"ROUTING TO: {route}")
    if route == "rag":
        return "rag"
    elif route == "web":
        return "web"
    else:
        return "rag"

def should_continue(state: MultiAgentState):
    verdict = state["verdict"]
    route = state["route"]
    print(f"VERDICT: {verdict}")
    if route == "both":
        return "web"  
    if verdict == "supported":
        return "end"
    else:
        return "web"
    
def create_final_node(llm):
    def final_node(state: MultiAgentState):
        answer = state.get("answer", "")
        web_results = state.get("web_results", "")
        verdict = state.get("verdict", "supported")

        if verdict == "not_supported":
            answer = ""

        if answer and web_results:
            prompt = SYNTHESIS_PROMPT.format(answer=answer, web_results=web_results)
            response = llm.invoke(prompt)
            final_answer = extract_content(response)
        elif answer:
            final_answer = answer
        else:
            final_answer = web_results

        return {"answer": final_answer}
    return final_node
    
def build_supervisor(llm, doc_summaries, checkpointer=None):
    supervisor_node = create_supervisor_node(llm, doc_summaries)
    validator_node = create_validator_node(llm)
    final_node = create_final_node(llm)
    
    graph = StateGraph(MultiAgentState)
    
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag", call_rag_specialist)
    graph.add_node("web", call_web_specialist)
    graph.add_node("validator", validator_node)
    graph.add_node("final", final_node)
    
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", should_route, {
        "rag": "rag",
        "web": "web"
    })
    graph.add_edge("rag", "validator")
    graph.add_conditional_edges("validator", should_continue, {
        "end": "final",
        "web": "web"
    })
    graph.add_edge("web", "final")
    graph.add_edge("final", END)
    return graph.compile(checkpointer=checkpointer)