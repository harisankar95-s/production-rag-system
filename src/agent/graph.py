from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from src.agent.tools import rag_search, web_search
from src.agent.prompts import build_system_prompt
import logging

logger = logging.getLogger(__name__)

def create_agent(llm, summaries):
    llm_with_tools = llm.bind_tools([rag_search, web_search])
    system_prompt = build_system_prompt(summaries)

    def agent_node(state: MessagesState):
        logger.info(f"Agent node — messages in state: {len(state['messages'])}")
        messages = [SystemMessage(content=system_prompt)] + state['messages']
        response = llm_with_tools.invoke(messages)
        logger.info(f"Agent node — LLM response has tool calls: {bool(response.tool_calls)}")
        return {'messages': [response]}

    return agent_node

def should_continue(state: MessagesState):
    last_message = state['messages'][-1]
    decision = 'tools' if last_message.tool_calls else 'END'
    logger.info(f"Conditional edge — decision: {decision}, total messages: {len(state['messages'])}")
    if last_message.tool_calls:
        return 'tools'
    return END

def build_graph(llm, summaries):
    agent_node = create_agent(llm, summaries)
    tools = [rag_search, web_search]

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()