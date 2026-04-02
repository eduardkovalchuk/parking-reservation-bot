"""
RAG evaluation script.

Runs the full evaluation suite against the live system (requires running Docker
services) and outputs a performance report to the console and optionally a JSON file.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --output report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.database.vector_store import get_weaviate_client, similarity_search
from src.evaluation.metrics import EvalReport, EvalSample, RAGEvaluator
from src.rag.chain import generate_answer
from src.rag.retriever import retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground-truth evaluation dataset
# ---------------------------------------------------------------------------

EVAL_DATASET: List[EvalSample] = [
    EvalSample(
        query="Where is CityPark located?",
        expected_answer="Orlyplein 10, 1043 DP Amsterdam (Sloterdijk)",
        relevant_doc_ids=["location", "general"],
    ),
    EvalSample(
        query="What is the hourly parking rate?",
        expected_answer="The hourly rate is EUR 3.00",
        relevant_doc_ids=["hours", "general"],
    ),
    EvalSample(
        query="Does the parking offer EV charging?",
        expected_answer="Yes, 20 EV charging bays are available on Level B3 with CHAdeMO and CCS connectors.",
        relevant_doc_ids=["amenities", "spaces"],
    ),
    EvalSample(
        query="What are the parking opening hours?",
        expected_answer="The facility is open 24 hours a day, 7 days a week.",
        relevant_doc_ids=["hours"],
    ),
    EvalSample(
        query="How do I cancel my reservation?",
        expected_answer="Contact info@citypark.com or call +31 20 555 0123 with your Reservation ID.",
        relevant_doc_ids=["booking", "policies"],
    ),
    EvalSample(
        query="How many parking spaces are available?",
        expected_answer="There are 150 total spaces: 100 standard, 20 compact, 10 handicapped, 20 EV.",
        relevant_doc_ids=["spaces", "general"],
    ),
    EvalSample(
        query="What is the maximum vehicle height allowed?",
        expected_answer="The maximum vehicle height is 2.1 metres.",
        relevant_doc_ids=["policies"],
    ),
    EvalSample(
        query="What payment methods are accepted?",
        expected_answer="Visa, Mastercard, Amex, contactless, Apple Pay, Google Pay, and cash at kiosk.",
        relevant_doc_ids=["payment"],
    ),
]


def build_retrieve_fn(weaviate_client):
    """Return a retrieve function compatible with RAGEvaluator."""
    settings = get_settings()

    def retrieve_fn(query: str) -> Tuple[List[str], str]:
        result = retrieve(query, weaviate_client, k=settings.retrieval_k)
        # Use category metadata as doc IDs for retrieval evaluation
        doc_ids = [
            doc.metadata.get("category", "general") for doc in result.static_docs
        ]
        return doc_ids, result.combined_context

    return retrieve_fn


def build_generate_fn():
    """Return a generate function compatible with RAGEvaluator."""
    def generate_fn(query: str, context: str) -> str:
        return generate_answer(question=query, context=context)

    return generate_fn


def print_report(report: EvalReport) -> None:
    """Pretty-print the evaluation report to stdout."""
    summary = report.summary()
    print("\n" + "=" * 60)
    print("  RAG EVALUATION REPORT — CityPark Parking Chatbot")
    print("=" * 60)
    print(f"  Samples evaluated  : {summary['num_samples']}")
    print()
    print("  Retrieval Metrics:")
    print(f"    Precision@K       : {summary['mean_precision_at_k']:.4f}")
    print(f"    Recall@K          : {summary['mean_recall_at_k']:.4f}")
    print(f"    Hit Rate@K        : {summary['mean_hit_rate']:.4f}")
    print()
    print("  Generation Quality (LLM-as-judge):")
    print(f"    Faithfulness      : {summary['mean_faithfulness']:.4f}")
    print(f"    Answer Relevance  : {summary['mean_answer_relevance']:.4f}")
    print()
    print("  Latency:")
    print(f"    p50               : {summary['latency_p50_ms']:.1f} ms")
    print(f"    p95               : {summary['latency_p95_ms']:.1f} ms")
    print(f"    p99               : {summary['latency_p99_ms']:.1f} ms")
    print("=" * 60)

    print("\n  Per-sample details:")
    print("-" * 60)
    for r in report.results:
        print(f"\n  Q: {r.query}")
        print(f"     Precision@K={r.precision_at_k:.2f}  Recall@K={r.recall_at_k:.2f}"
              f"  Hit={int(r.hit_rate)}  Latency={r.latency_ms:.0f}ms")
        if r.faithfulness is not None:
            print(f"     Faithfulness={r.faithfulness:.2f}  Relevance={r.answer_relevance:.2f}")
        print(f"     Answer: {r.actual_answer[:120]}…")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG chatbot system.")
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Optional path to write the JSON report."
    )
    parser.add_argument(
        "--no-llm-judge", action="store_true",
        help="Skip LLM-as-judge metrics (faster, cheaper)."
    )
    args = parser.parse_args()

    settings = get_settings()
    client = get_weaviate_client()
    evaluator = RAGEvaluator(k=settings.retrieval_k, use_llm_judge=not args.no_llm_judge)

    try:
        retrieve_fn = build_retrieve_fn(client)
        generate_fn = build_generate_fn()

        logger.info("Starting evaluation on %d samples…", len(EVAL_DATASET))
        report = evaluator.evaluate(EVAL_DATASET, retrieve_fn, generate_fn)

        print_report(report)

        if args.output:
            output_path = Path(args.output)
            data = {
                "summary": report.summary(),
                "results": [
                    {
                        "query": r.query,
                        "expected_answer": r.expected_answer,
                        "actual_answer": r.actual_answer,
                        "retrieved_doc_ids": r.retrieved_doc_ids,
                        "latency_ms": round(r.latency_ms, 2),
                        "precision_at_k": r.precision_at_k,
                        "recall_at_k": r.recall_at_k,
                        "hit_rate": r.hit_rate,
                        "faithfulness": r.faithfulness,
                        "answer_relevance": r.answer_relevance,
                    }
                    for r in report.results
                ],
            }
            output_path.write_text(json.dumps(data, indent=2))
            logger.info("Report written to %s", output_path)
    finally:
        evaluator.close()
        client.close()


if __name__ == "__main__":
    main()
