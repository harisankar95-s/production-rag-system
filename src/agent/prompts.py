def build_system_prompt(summaries: dict) -> str:
    doc_context = "\n".join([f"- {name}: {summary}" 
                             for name, summary in summaries.items()])
    return f"""You are a helpful document assistant.

The user has the following documents available:
{doc_context}

Use rag_search for questions that can be answered from these documents.
Use web_search for current events or information not in these documents.
If unsure, prefer rag_search first.
When using information from rag_search results, always preserve 
the page citations exactly as they appear in the tool output."""

def get_message_text(m):
    if isinstance(m.content, list):
        return " ".join([
            item.get('text', '') 
            for item in m.content 
            if isinstance(item, dict)
        ])
    return m.content

def create_summerize_prompt(messages):
    messages_text = "\n".join([
        f"{m.__class__.__name__}: {get_message_text(m)}" 
        for m in messages 
        if m.content
    ])
    system_summerize_prompt = f"""Summarize the following conversation messages into a single concise paragraph.
Preserve what the user asked, what information was retrieved, and what was answered.
Be concise — this summary replaces multiple messages to save context space.

Messages:
{messages_text}"""
    return system_summerize_prompt