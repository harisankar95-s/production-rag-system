from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage,HumanMessage,RemoveMessage
from src.agent.tools import rag_search, web_search
from src.agent.prompts import build_system_prompt,create_summerize_prompt
from src.agent.memory import get_checkpointer
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


def create_summarize_node(llm):
    def summarize_node(state: MessagesState):
        logger.info(f"Summarize node — compressing first 6 of {len(state['messages'])} messages")
        messages = state['messages'][:6]
        system_summarize_prompt = create_summerize_prompt(messages)
        print("PROMPT LENGTH:", len(system_summarize_prompt))
        print("PROMPT PREVIEW:", system_summarize_prompt[:500])
        response = llm.invoke([HumanMessage(content=system_summarize_prompt)])
        messages_to_remove = [RemoveMessage(id=m.id) for m in state['messages'][:6]]
        summary_message = HumanMessage(content=f"Summary: {response.content}")
        logger.info(f"Summarize node — done, messages after: {len(state['messages']) - 6 + 1}")
        return {"messages": messages_to_remove + [summary_message]}
    return summarize_node

def should_continue(state: MessagesState):
    last_message = state['messages'][-1]
    decision = 'tools' if last_message.tool_calls else 'END'
    logger.info(f"Conditional edge — decision: {decision}, total messages: {len(state['messages'])}")
    if last_message.tool_calls:
        return 'tools'
    return END

def should_summarize(state: MessagesState):
    total_messages = len(state['messages'])
    logger.info(f"Should summarize — total messages: {total_messages}")
    if total_messages >= 10:
        logger.info("Summarization triggered")
        return 'summarize'
    else:
        return 'continue'

def build_graph(llm, summaries):
    agent_node = create_agent(llm, summaries)
    summerize_node = create_summarize_node(llm)
    tools = [rag_search, web_search]

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node('summarize',summerize_node)
    graph.add_conditional_edges(START, should_summarize, {
    "summarize": "summarize",
    "continue": "agent"})
    graph.add_edge('summarize', "agent")
    graph.add_conditional_edges("agent", should_continue)
    checkpointer = get_checkpointer()
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
