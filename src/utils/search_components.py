from langchain_community.tools import DuckDuckGoSearchRun

def web_search_run(query: str) -> str:
    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as e:
        return f"Web search is currently unavailable: {str(e)}. Answer from your own knowledge if possible, or inform the user that current information cannot be retrieved."