import json
import os
import logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from src.utils.utils import get_embedding_model, get_rerank_model,get_llm,get_eval_llm
from src.rag.retriever import load_vectorstore, get_hybrid_retriever,get_multi_query_retriever
from src.rag.pipeline import ask
from src.config import config

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def load_test_set(path: str):
    """Loads the evaluation questions and ground truths."""
    with open(path, 'r') as f:
        return json.load(f)


def run_evaluation():
    embedding_model = get_embedding_model()
    rerank_model = get_rerank_model()

    vectorstore = load_vectorstore(embedding_model)
    generator_llm = get_llm()
    retriever = get_multi_query_retriever(vectorstore,generator_llm)

    test_set = load_test_set("data/test_set.json")

    user_inputs = []
    responses = []
    retrieved_contexts = []
    references = []

    print(f"--- Generating RAG responses for {len(test_set)} items ---")
    

    for item in test_set:
        question = item["question"]
        ground_truth = item["ground_truth"]

        answer, chunks = ask(generator_llm, retriever, rerank_model, question)
        
        user_inputs.append(question)
        responses.append(answer)
        retrieved_contexts.append([chunk.page_content for chunk in chunks])
        references.append(ground_truth)

        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print(f"Chunks used: {len(chunks)}\n")


    data = {
        "user_input": user_inputs,
        "response": responses,
        "retrieved_contexts": retrieved_contexts,
        "reference": references
    }
    dataset = Dataset.from_dict(data)

    evaluator_llm = LangchainLLMWrapper(get_eval_llm())

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=config.google_api_key
        )
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(), 
            AnswerRelevancy(), 
            ContextPrecision()
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(timeout=300, max_workers=1)
    )

    print("\n=== RAGAS Scores ===")
    print(result)
    


if __name__ == "__main__":
    run_evaluation()