from langchain_core.prompts import PromptTemplate

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Answer the question using the context provided below.
Use the context as your primary source. If the context contains relevant information, use it.
Always cite the page number(s) from the context that support your answer, like (Page 15).
Only say "I don't know" if the context has absolutely no relevant information.

Context:
{context}

Question: {question}
Answer:"""
)