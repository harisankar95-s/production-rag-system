_rag_components = {}

def init_rag_components(llm, retriever, rerank_model):
    _rag_components['llm'] = llm
    _rag_components['retriever'] = retriever
    _rag_components['rerank_model'] = rerank_model