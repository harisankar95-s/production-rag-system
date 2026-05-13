from src.utils.utils import extract_content
from src.multi_agent.prompts import VALIDATOR_PROMPT
from src.multi_agent.state import MultiAgentState
from src.utils.timer import timer

def validate_answer(llm, query, answer, chunks):
    prompt = VALIDATOR_PROMPT.format(query=query, chunks=chunks, answer=answer)
    print(f"VALIDATING — Query: {query}")
    print(f"VALIDATING — Answer: {answer[:200]}")
    response = llm.invoke(prompt)
    if hasattr(response, 'content'):
        response = extract_content(response)

    return response

def create_validator_node(llm):
    def validator_node(state: MultiAgentState):
        with timer("validator_llm_call"):
            verdict = validate_answer(
                llm,
                state["rag_query"],
                state["answer"],
                state["chunks"]
            )
        return {"verdict": verdict}
    return validator_node