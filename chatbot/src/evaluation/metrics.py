"""
RAG evaluation metrics for the parking chatbot.

Implemented metrics:
1. Retrieval quality
   - Precision@K  : fraction of retrieved docs that are relevant
   - Recall@K     : fraction of relevant docs that were retrieved
   - Hit Rate      : whether at least one relevant doc appears in top-K

2. Generation quality (LLM-as-judge)
   - Faithfulness  : does the answer stay within the retrieved context?
   - Answer Relevance: is the answer on-point for the question?

3. System performance
   - Latency (ms)  : end-to-end response time (p50, p95, p99)
   - Token usage  : prompt + completion tokens per call
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from langchain_openai import ChatOpenAI

from src.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    """A single evaluation sample with ground-truth labels."""

    query: str
    expected_answer: str                   # reference answer (ground-truth)
    relevant_doc_ids: List[str]            # IDs / fragments of truly relevant docs


@dataclass
class EvalResult:
    """Result for one evaluation run on a single sample."""

    query: str
    expected_answer: str
    actual_answer: str
    retrieved_doc_ids: List[str]
    latency_ms: float
    precision_at_k: float
    recall_at_k: float
    hit_rate: float
    faithfulness: Optional[float] = None   # 0.0 – 1.0
    answer_relevance: Optional[float] = None  # 0.0 – 1.0


@dataclass
class EvalReport:
    """Aggregate report across all evaluation samples."""

    results: List[EvalResult] = field(default_factory=list)

    # Aggregate stats (computed lazily)
    _computed: bool = False

    @property
    def mean_precision(self) -> float:
        return float(np.mean([r.precision_at_k for r in self.results])) if self.results else 0.0

    @property
    def mean_recall(self) -> float:
        return float(np.mean([r.recall_at_k for r in self.results])) if self.results else 0.0

    @property
    def mean_hit_rate(self) -> float:
        return float(np.mean([r.hit_rate for r in self.results])) if self.results else 0.0

    @property
    def mean_faithfulness(self) -> float:
        scores = [r.faithfulness for r in self.results if r.faithfulness is not None]
        return float(np.mean(scores)) if scores else 0.0

    @property
    def mean_answer_relevance(self) -> float:
        scores = [r.answer_relevance for r in self.results if r.answer_relevance is not None]
        return float(np.mean(scores)) if scores else 0.0

    @property
    def latency_p50(self) -> float:
        lats = [r.latency_ms for r in self.results]
        return float(np.percentile(lats, 50)) if lats else 0.0

    @property
    def latency_p95(self) -> float:
        lats = [r.latency_ms for r in self.results]
        return float(np.percentile(lats, 95)) if lats else 0.0

    @property
    def latency_p99(self) -> float:
        lats = [r.latency_ms for r in self.results]
        return float(np.percentile(lats, 99)) if lats else 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "num_samples": len(self.results),
            "mean_precision_at_k": round(self.mean_precision, 4),
            "mean_recall_at_k": round(self.mean_recall, 4),
            "mean_hit_rate": round(self.mean_hit_rate, 4),
            "mean_faithfulness": round(self.mean_faithfulness, 4),
            "mean_answer_relevance": round(self.mean_answer_relevance, 4),
            "latency_p50_ms": round(self.latency_p50, 2),
            "latency_p95_ms": round(self.latency_p95, 2),
            "latency_p99_ms": round(self.latency_p99, 2),
        }


# ---------------------------------------------------------------------------
# Retrieval metrics (pure Python, no LLM needed)
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Compute Precision@K.

    Precision@K = |retrieved_top_k ∩ relevant| / K

    Args:
        retrieved_ids: Ordered list of retrieved document identifiers.
        relevant_ids: Set of truly relevant document identifiers.
        k: Cut-off rank.

    Returns:
        Precision score in [0, 1].
    """
    if k <= 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(top_k & relevant_set) / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Compute Recall@K.

    Recall@K = |retrieved_top_k ∩ relevant| / |relevant|

    Args:
        retrieved_ids: Ordered list of retrieved document identifiers.
        relevant_ids: Set of truly relevant document identifiers.
        k: Cut-off rank.

    Returns:
        Recall score in [0, 1].  Returns 0 if relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(top_k & relevant_set) / len(relevant_set)


def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Compute Hit Rate@K (binary: 1 if any relevant doc in top-K, else 0).

    Args:
        retrieved_ids: Ordered list of retrieved document identifiers.
        relevant_ids: Set of truly relevant document identifiers.
        k: Cut-off rank.

    Returns:
        1.0 if at least one relevant doc appears in the top-K, else 0.0.
    """
    top_k = set(retrieved_ids[:k])
    return 1.0 if set(relevant_ids) & top_k else 0.0


# ---------------------------------------------------------------------------
# Generation quality metrics (LLM-as-judge)
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """You are an evaluator assessing whether an AI answer is faithful to the provided context.

Context:
{context}

Question: {question}
AI Answer: {answer}

Rate the faithfulness of the answer on a scale from 0 to 1:
- 1.0 = All claims in the answer are supported by the context
- 0.5 = Partially supported; some claims are outside the context
- 0.0 = The answer contradicts or ignores the context

Respond with ONLY a JSON object: {{"score": <float>}}"""


_RELEVANCE_PROMPT = """You are an evaluator assessing whether an AI answer is relevant to the question.

Question: {question}
AI Answer: {answer}

Rate the relevance of the answer on a scale from 0 to 1:
- 1.0 = The answer directly and fully addresses the question
- 0.5 = The answer partially addresses the question
- 0.0 = The answer does not address the question at all

Respond with ONLY a JSON object: {{"score": <float>}}"""


def _llm_score(prompt: str, llm: ChatOpenAI) -> Optional[float]:
    """Call the LLM judge and parse the score from its JSON response."""
    try:
        response = llm.invoke(prompt)
        content = response.content
        if not isinstance(content, str):
            logger.warning("LLM-as-judge returned unexpected content type: %s", type(content))
            return None
        content = content.strip()
        # Strip markdown code fences (e.g. ```json ... ``` or ``` ... ```)
        if content.startswith("```"):
            lines = content.splitlines()
            inner = [line for line in lines[1:] if line.strip() != "```"]
            content = "\n".join(inner).strip()
        if not content:
            logger.warning("LLM-as-judge returned empty content")
            return None
        parsed = json.loads(content)
        return float(parsed.get("score", 0.0))
    except Exception as exc:
        logger.warning("LLM-as-judge failed: %s", exc)
        return None


def faithfulness_score(question: str, context: str, answer: str, llm: ChatOpenAI) -> Optional[float]:
    """Evaluate how faithful the answer is to the retrieved context."""
    prompt = _FAITHFULNESS_PROMPT.format(
        context=context, question=question, answer=answer
    )
    return _llm_score(prompt, llm)


def answer_relevance_score(question: str, answer: str, llm: ChatOpenAI) -> Optional[float]:
    """Evaluate how relevant the answer is to the question."""
    prompt = _RELEVANCE_PROMPT.format(question=question, answer=answer)
    return _llm_score(prompt, llm)


# ---------------------------------------------------------------------------
# Latency measurement utility
# ---------------------------------------------------------------------------

def measure_latency(fn: Callable, *args, **kwargs) -> tuple[Any, float]:
    """
    Measure the wall-clock latency of a callable.

    Returns:
        (result, latency_ms): The function's return value and elapsed time in ms.
    """
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


# ---------------------------------------------------------------------------
# High-level evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Orchestrates retrieval and generation evaluation for a set of samples.

    Example:
        evaluator = RAGEvaluator(k=5, use_llm_judge=True)
        report = evaluator.evaluate(samples, retrieve_fn, generate_fn)
        print(report.summary())
    """

    def __init__(self, k: int = 5, use_llm_judge: bool = True) -> None:
        self.k = k
        self.use_llm_judge = use_llm_judge
        self._llm: Optional[ChatOpenAI] = None

    def _get_llm(self) -> ChatOpenAI:
        if self._llm is None:
            s = get_settings()
            self._llm = ChatOpenAI(
                model=s.openai_chat_model,
                openai_api_key=s.openai_api_key,
                temperature=0.0,
            )
        return self._llm

    def close(self) -> None:
        """Release the underlying HTTP client to avoid ResourceWarning on exit."""
        if self._llm is not None:
            try:
                self._llm.root_client.close()
            except Exception:
                pass
            self._llm = None

    def evaluate(
        self,
        samples: List[EvalSample],
        retrieve_fn: Callable[[str], tuple[List[str], str]],
        generate_fn: Callable[[str, str], str],
    ) -> EvalReport:
        """
        Run evaluation over all samples.

        Args:
            samples: List of EvalSample objects with ground-truth.
            retrieve_fn: Callable that takes a query and returns
                         (list_of_doc_ids, combined_context_string).
            generate_fn: Callable that takes (query, context) and returns answer string.

        Returns:
            EvalReport with per-sample results and aggregate stats.
        """
        report = EvalReport()

        for i, sample in enumerate(samples):
            logger.info("Evaluating sample %d/%d: '%s'", i + 1, len(samples), sample.query[:60])

            # ── Retrieval + latency ───────────────────────────────────────
            (retrieved_ids, context), retrieve_ms = measure_latency(
                retrieve_fn, sample.query
            )

            prec = precision_at_k(retrieved_ids, sample.relevant_doc_ids, self.k)
            rec = recall_at_k(retrieved_ids, sample.relevant_doc_ids, self.k)
            hit = hit_rate_at_k(retrieved_ids, sample.relevant_doc_ids, self.k)

            # ── Generation + latency ──────────────────────────────────────
            actual_answer, gen_ms = measure_latency(generate_fn, sample.query, context)
            total_ms = retrieve_ms + gen_ms

            # ── LLM-as-judge ──────────────────────────────────────────────
            faith: Optional[float] = None
            relevance: Optional[float] = None
            if self.use_llm_judge:
                llm = self._get_llm()
                faith = faithfulness_score(sample.query, context, actual_answer, llm)
                relevance = answer_relevance_score(sample.query, actual_answer, llm)

            result = EvalResult(
                query=sample.query,
                expected_answer=sample.expected_answer,
                actual_answer=actual_answer,
                retrieved_doc_ids=retrieved_ids,
                latency_ms=total_ms,
                precision_at_k=prec,
                recall_at_k=rec,
                hit_rate=hit,
                faithfulness=faith,
                answer_relevance=relevance,
            )
            report.results.append(result)

        return report
