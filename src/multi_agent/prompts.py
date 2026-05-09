SUPERVISOR_PROMPT = """You are a supervisor routing questions to specialist agents.

Available documents: {doc_summaries}

Rules:
- Route to "rag" if the question is about topics in the documents above
- Route to "web" for current events, sports, news, recent information
- Route to "both" if the question needs information from BOTH documents AND web

Always extract the relevant sub-question for each specialist:
- rag_query: the part of the question answerable from documents (empty string if not applicable)
- web_query: the part of the question answerable from web (empty string if not applicable)

You MUST respond with valid JSON only. No other text. Always include all three fields.
Examples:
{{"route": "rag", "rag_query": "what is gradient descent", "web_query": ""}}
{{"route": "web", "rag_query": "", "web_query": "who won IPL 2025"}}
{{"route": "both", "rag_query": "what is gradient descent", "web_query": "who won IPL 2025"}}"""


VALIDATOR_PROMPT = """You are a judge checking if an answer is grounded in retrieved documents.

Question: {query}
Retrieved Chunks: {chunks}
Generated Answer: {answer}

Check if the answer is based on the retrieved chunks.
If the answer directly uses information from the chunks — respond: supported
If the answer says "I don't know" with NO useful information — respond: not_supported
If the answer contains claims completely absent from chunks — respond: not_supported

Respond with exactly one word: supported or not_supported"""

SYNTHESIS_PROMPT = """You have gathered information from two sources to answer a user's question.

Document findings:
{answer}

Web search findings:
{web_results}

Provide a single coherent answer combining both sources. 
Cite page numbers from document findings where relevant.
Clearly indicate which information comes from web search."""