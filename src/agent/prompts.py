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