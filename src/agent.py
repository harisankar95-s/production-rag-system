from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,SystemMessage

from src.pipeline import ask
from langchain_community.tools import DuckDuckGoSearchRun


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

def create_agent(llm, summaries):
    llm_with_tools = llm.bind_tools([rag_search, web_search])
    
    doc_context = "\n".join([f"- {name}: {summary}" 
                             for name, summary in summaries.items()])
    
    system_prompt = f"""You are a helpful document assistant.

The user has the following documents available:
{doc_context}

Use rag_search for questions that can be answered from these documents.
Use web_search for current events or information not in these documents.
If unsure, prefer rag_search first."""

    def agent_node(state: MessagesState):
        messages = [SystemMessage(content=system_prompt)] + state['messages']
        response = llm_with_tools.invoke(messages)
        return {'messages': [response]}
    
    return agent_node

def should_continue(state: MessagesState):
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return 'tools'
    return END

def build_graph(llm,summaries):
    agent_node = create_agent(llm,summaries)
    tools = [rag_search, web_search]
    
    graph = StateGraph(MessagesState)
    
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    
    return graph.compile()