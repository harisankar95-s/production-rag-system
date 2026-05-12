import json
import os
import sys
import time
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.retriever import load_vectorstore, get_multi_query_retriever
from src.multi_agent.supervisor import build_supervisor
from src.utils.rag_components import init_rag_components
from src.config import config
from src.utils.utils import get_embedding_model, get_rerank_model, get_llm, load_json
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage


THRESHOLDS = {
    "routing_accuracy":   0.80,
    "faithfulness":       0.70,
    "answer_relevance":   0.70,
    "context_relevance":  0.60,
    "end_to_end_quality": 3.5,
}


@dataclass
class SingleResult:
    question_id: str
    question: str
    expected_route: str
    actual_route: str
    routing_correct: bool
    final_answer: str
    latency_seconds: float
    error: Optional[str] = None
    retrieved_chunks: list = field(default_factory=list)
    rag_answer: str = ""
    faithfulness_score: float = 0.0
    answer_relevance_score: float = 0.0
    context_relevance_score: float = 0.0
    end_to_end_score: float = 0.0
    judge_reasoning: str = ""


@dataclass
class EvaluationReport:
    timestamp: str
    total_questions: int
    routing_accuracy: float
    avg_faithfulness: float
    avg_answer_relevance: float
    avg_context_relevance: float
    avg_end_to_end_score: float
    avg_latency_seconds: float
    results: list = field(default_factory=list)
    passed_thresholds: dict = field(default_factory=dict)
    failed_questions: list = field(default_factory=list)


class AgentRunner:
    def __init__(self):
        print("Initialising agent...")
        self.embedding_model = get_embedding_model()
        self.rerank_model    = get_rerank_model()
        self.vector_store    = load_vectorstore(self.embedding_model)
        self.llm             = get_llm()
        self.retriever       = get_multi_query_retriever(self.vector_store, self.llm)

        init_rag_components(self.llm, self.retriever, self.rerank_model)

        doc_details  = load_json(os.path.join(config.data_dir, "document_summaries.json"))
        self.graph   = build_supervisor(self.llm, doc_details)
        print("Agent ready.\n")

    def run(self, question: str) -> dict:
        import uuid
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        result = self.graph.invoke(
            {"query": question, "messages": [HumanMessage(content=question)]},
            cfg
        )
        return result


def evaluate_routing(results: list[SingleResult]) -> float:
    correct = sum(1 for r in results if r.routing_correct)
    return round(correct / len(results), 3) if results else 0.0


def evaluate_ragas(llm, question: str, answer: str, chunks: list) -> dict:
    if not answer or not chunks:
        return {"faithfulness": 0.0, "answer_relevance": 0.0, "context_relevance": 0.0}

    context = "\n\n".join([
        c.page_content if hasattr(c, "page_content") else str(c)
        for c in chunks[:5]
    ])

    faithfulness_prompt = f"""You are evaluating whether an AI answer is faithful to the provided context.

Context:
{context}

Answer:
{answer}

Score the faithfulness from 0.0 to 1.0 where:
1.0 = Every claim in the answer is directly supported by the context
0.5 = Some claims are supported, some are not
0.0 = The answer contradicts or ignores the context entirely

Return ONLY a JSON object like this: {{"score": 0.85, "reason": "brief explanation"}}"""

    relevance_prompt = f"""You are evaluating whether an AI answer is relevant to the question asked.

Question:
{question}

Answer:
{answer}

Score the relevance from 0.0 to 1.0 where:
1.0 = Answer directly and completely addresses the question
0.5 = Answer partially addresses the question
0.0 = Answer is completely unrelated to the question

Return ONLY a JSON object like this: {{"score": 0.85, "reason": "brief explanation"}}"""

    context_prompt = f"""You are evaluating whether the retrieved context is relevant to the question.

Question:
{question}

Retrieved Context:
{context}

Score the context relevance from 0.0 to 1.0 where:
1.0 = Context contains exactly what is needed to answer the question
0.5 = Context is partially relevant
0.0 = Context is completely unrelated to the question

Return ONLY a JSON object like this: {{"score": 0.85, "reason": "brief explanation"}}"""

    def get_score(prompt: str) -> float:
        try:
            response = llm.invoke([
                SystemMessage(content="You are an evaluation assistant. Return only valid JSON."),
                HumanMessage(content=prompt)
            ])
            content = response.content
            if isinstance(content, list):
                content = content[0].get("text", "")
            import re
            match = re.search(r'\{.*?\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return float(data.get("score", 0.0))
        except Exception as e:
            print(f"    RAGAS scoring error: {e}")
        return 0.0

    return {
        "faithfulness":      get_score(faithfulness_prompt),
        "answer_relevance":  get_score(relevance_prompt),
        "context_relevance": get_score(context_prompt),
    }


def evaluate_end_to_end(llm, question: str, answer: str) -> dict:
    if not answer:
        return {"score": 0.0, "reasoning": "No answer produced"}

    prompt = f"""You are evaluating the quality of an AI assistant's response.

Question asked by user:
{question}

AI assistant's response:
{answer}

Score the response from 1 to 5 where:
5 = Excellent — complete, accurate, clear, directly addresses the question
4 = Good — mostly complete and accurate, minor gaps
3 = Acceptable — partially answers but missing important details
2 = Poor — significant gaps or inaccuracies
1 = Unacceptable — wrong, irrelevant, or empty

Return ONLY a JSON object like this:
{{"score": 4, "reasoning": "brief explanation of the score"}}"""

    try:
        response = llm.invoke([
            SystemMessage(content="You are an evaluation assistant. Return only valid JSON."),
            HumanMessage(content=prompt)
        ])
        content = response.content
        if isinstance(content, list):
            content = content[0].get("text", "")
        import re
        match = re.search(r'\{.*?\}', content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "score":     float(data.get("score", 0.0)),
                "reasoning": data.get("reasoning", "")
            }
    except Exception as e:
        print(f"    End-to-end scoring error: {e}")
    return {"score": 0.0, "reasoning": "Scoring failed"}


def run_evaluation(
    test_set_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_set.json"),
    output_path:   str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    max_questions: Optional[int] = None,
    routes_to_test: Optional[list] = None,
):
    os.makedirs(output_path, exist_ok=True)

    with open(test_set_path) as f:
        test_set = json.load(f)

    if routes_to_test:
        test_set = [q for q in test_set if q["expected_route"] in routes_to_test]

    if max_questions:
        test_set = test_set[:max_questions]

    print(f"Running evaluation on {len(test_set)} questions...")
    print("=" * 60)

    runner = AgentRunner()
    all_results = []

    for i, test_case in enumerate(test_set):
        qid      = test_case["id"]
        question = test_case["question"]
        expected = test_case["expected_route"]

        print(f"\n[{i+1}/{len(test_set)}] {qid} — {question[:60]}...")

        start_time = time.time()
        error = None
        result_state = {}

        try:
            result_state = runner.run(question)
        except Exception as e:
            error = str(e)
            print(f"    ERROR: {e}")

        latency = round(time.time() - start_time, 2)

        actual_route = result_state.get("route", "unknown")
        final_answer = result_state.get("final_output", "") or result_state.get("answer", "")
        chunks       = result_state.get("chunks", [])
        rag_answer   = result_state.get("answer", "")

        routing_correct = actual_route == expected
        print(f"    Route: expected={expected} | actual={actual_route} | {'PASS' if routing_correct else 'FAIL'}")
        print(f"    Latency: {latency}s")

        ragas_scores = {"faithfulness": 0.0, "answer_relevance": 0.0, "context_relevance": 0.0}
        if actual_route in ("rag", "both") and rag_answer and chunks:
            print("    Running RAGAS evaluation...")
            rag_question = test_case.get("rag_query", question)
            ragas_scores = evaluate_ragas(runner.llm, rag_question, rag_answer, chunks)
            print(f"    Faithfulness: {ragas_scores['faithfulness']} | "
                  f"Relevance: {ragas_scores['answer_relevance']} | "
                  f"Context: {ragas_scores['context_relevance']}")

        print("    Running end-to-end evaluation...")
        e2e = evaluate_end_to_end(runner.llm, question, final_answer)
        print(f"    E2E Score: {e2e['score']}/5 — {e2e['reasoning'][:80]}")

        result = SingleResult(
            question_id             = qid,
            question                = question,
            expected_route          = expected,
            actual_route            = actual_route,
            routing_correct         = routing_correct,
            final_answer            = final_answer,
            latency_seconds         = latency,
            error                   = error,
            retrieved_chunks        = [c.page_content if hasattr(c, "page_content") else str(c) for c in chunks[:3]],
            rag_answer              = rag_answer,
            faithfulness_score      = ragas_scores["faithfulness"],
            answer_relevance_score  = ragas_scores["answer_relevance"],
            context_relevance_score = ragas_scores["context_relevance"],
            end_to_end_score        = e2e["score"],
            judge_reasoning         = e2e["reasoning"],
        )
        all_results.append(result)

        time.sleep(2)

    routing_accuracy      = evaluate_routing(all_results)
    rag_results           = [r for r in all_results if r.expected_route in ("rag", "both") and not r.error]
    avg_faithfulness      = round(sum(r.faithfulness_score for r in rag_results) / len(rag_results), 3) if rag_results else 0.0
    avg_answer_relevance  = round(sum(r.answer_relevance_score for r in rag_results) / len(rag_results), 3) if rag_results else 0.0
    avg_context_relevance = round(sum(r.context_relevance_score for r in rag_results) / len(rag_results), 3) if rag_results else 0.0
    avg_e2e               = round(sum(r.end_to_end_score for r in all_results) / len(all_results), 3) if all_results else 0.0
    avg_latency           = round(sum(r.latency_seconds for r in all_results) / len(all_results), 2) if all_results else 0.0

    passed = {
        "routing_accuracy":   routing_accuracy    >= THRESHOLDS["routing_accuracy"],
        "faithfulness":       avg_faithfulness     >= THRESHOLDS["faithfulness"],
        "answer_relevance":   avg_answer_relevance >= THRESHOLDS["answer_relevance"],
        "context_relevance":  avg_context_relevance>= THRESHOLDS["context_relevance"],
        "end_to_end_quality": avg_e2e              >= THRESHOLDS["end_to_end_quality"],
    }

    failed_questions = [r.question_id for r in all_results if not r.routing_correct or r.end_to_end_score < 3.0]

    report = EvaluationReport(
        timestamp             = datetime.now().isoformat(),
        total_questions       = len(all_results),
        routing_accuracy      = routing_accuracy,
        avg_faithfulness      = avg_faithfulness,
        avg_answer_relevance  = avg_answer_relevance,
        avg_context_relevance = avg_context_relevance,
        avg_end_to_end_score  = avg_e2e,
        avg_latency_seconds   = avg_latency,
        results               = [asdict(r) for r in all_results],
        passed_thresholds     = passed,
        failed_questions      = failed_questions,
    )

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path   = os.path.join(output_path, f"eval_report_{timestamp_str}.json")
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions evaluated : {report.total_questions}")
    print(f"Average latency           : {report.avg_latency_seconds}s per question")
    print()
    print("METRIC                  SCORE     THRESHOLD   STATUS")
    print("-" * 60)

    key_map = {
        "Routing Accuracy":   "routing_accuracy",
        "Faithfulness":       "faithfulness",
        "Answer Relevance":   "answer_relevance",
        "Context Relevance":  "context_relevance",
        "End-to-End Quality": "end_to_end_quality",
    }
    metrics = [
        ("Routing Accuracy",   routing_accuracy,       THRESHOLDS["routing_accuracy"],    "%"),
        ("Faithfulness",       avg_faithfulness,        THRESHOLDS["faithfulness"],        ""),
        ("Answer Relevance",   avg_answer_relevance,    THRESHOLDS["answer_relevance"],    ""),
        ("Context Relevance",  avg_context_relevance,   THRESHOLDS["context_relevance"],   ""),
        ("End-to-End Quality", avg_e2e,                 THRESHOLDS["end_to_end_quality"],  "/5"),
    ]
    for name, score, threshold, unit in metrics:
        status = "PASS" if passed[key_map[name]] else "FAIL"
        print(f"{name:<24} {score:<8}{threshold}{unit:<6}   {status}")

    print()
    if failed_questions:
        print(f"Failed questions: {', '.join(failed_questions)}")
    print(f"\nFull report saved to: {report_path}")

    return report


if __name__ == "__main__":
    import argparse

    _dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Evaluate multi-agent system")
    parser.add_argument("--test-set", default=os.path.join(_dir, "test_set.json"), help="Path to test set JSON")
    parser.add_argument("--output",   default=os.path.join(_dir, "results"),       help="Output directory for results")
    parser.add_argument("--max",      type=int, default=None,                       help="Max questions to run")
    parser.add_argument("--route",    choices=["rag", "web", "both"],               help="Test only specific route")
    args = parser.parse_args()

    routes = [args.route] if args.route else None

    run_evaluation(
        test_set_path  = args.test_set,
        output_path    = args.output,
        max_questions  = args.max,
        routes_to_test = routes,
    )