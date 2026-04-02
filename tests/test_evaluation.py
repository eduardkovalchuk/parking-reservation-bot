"""
Tests for src/evaluation/metrics.py

Tests cover:
- precision_at_k, recall_at_k, hit_rate_at_k (pure functions)
- measure_latency utility
- EvalReport aggregate properties
- RAGEvaluator.evaluate orchestration (with mocked LLM judge)
"""
import time
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.metrics import (
    EvalReport,
    EvalResult,
    EvalSample,
    RAGEvaluator,
    answer_relevance_score,
    faithfulness_score,
    hit_rate_at_k,
    measure_latency,
    precision_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# Tests: precision_at_k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_perfect_precision(self):
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b", "c", "d"]
        assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)

    def test_zero_precision(self):
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b"]
        assert precision_at_k(retrieved, relevant, k=3) == 0.0

    def test_partial_precision(self):
        retrieved = ["a", "x", "b"]
        relevant = ["a", "b"]
        # 2 relevant in top 3 → 2/3
        assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)

    def test_k_larger_than_retrieved(self):
        retrieved = ["a"]
        relevant = ["a", "b"]
        # Only 1 retrieved; k=5 → 1/5
        assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(1 / 5)

    def test_zero_k_returns_zero(self):
        assert precision_at_k(["a"], ["a"], k=0) == 0.0


# ---------------------------------------------------------------------------
# Tests: recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["a", "b"]
        relevant = ["a", "b"]
        assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(1.0)

    def test_zero_recall(self):
        retrieved = ["x", "y"]
        relevant = ["a", "b"]
        assert recall_at_k(retrieved, relevant, k=2) == 0.0

    def test_partial_recall(self):
        retrieved = ["a", "x", "y", "b"]
        relevant = ["a", "b", "c"]
        # Top-4 retrieved: a, x, y, b → 2 out of 3 relevant found
        assert recall_at_k(retrieved, relevant, k=4) == pytest.approx(2 / 3)

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k(["a", "b"], [], k=2) == 0.0

    def test_k_truncates_retrieved(self):
        retrieved = ["a", "b", "c"]    # c is relevant but beyond k=2
        relevant = ["a", "c"]
        # Only top-2 considered: a, b → only 'a' is relevant → 1/2
        assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(1 / 2)


# ---------------------------------------------------------------------------
# Tests: hit_rate_at_k
# ---------------------------------------------------------------------------

class TestHitRateAtK:
    def test_hit_when_relevant_in_top_k(self):
        assert hit_rate_at_k(["x", "a", "y"], ["a", "b"], k=3) == 1.0

    def test_miss_when_no_relevant_in_top_k(self):
        assert hit_rate_at_k(["x", "y", "z"], ["a", "b"], k=3) == 0.0

    def test_hit_at_first_position(self):
        assert hit_rate_at_k(["a", "x", "y"], ["a"], k=3) == 1.0

    def test_miss_when_relevant_beyond_k(self):
        assert hit_rate_at_k(["x", "y", "z", "a"], ["a"], k=2) == 0.0


# ---------------------------------------------------------------------------
# Tests: measure_latency
# ---------------------------------------------------------------------------

class TestMeasureLatency:
    def test_returns_result_and_positive_latency(self):
        def slow_fn():
            time.sleep(0.01)
            return 42

        result, ms = measure_latency(slow_fn)
        assert result == 42
        assert ms > 0.0

    def test_latency_in_milliseconds(self):
        def instant_fn():
            return "done"

        result, ms = measure_latency(instant_fn)
        assert result == "done"
        assert isinstance(ms, float)

    def test_passes_args_correctly(self):
        def add(a, b):
            return a + b

        result, _ = measure_latency(add, 3, 4)
        assert result == 7


# ---------------------------------------------------------------------------
# Tests: EvalReport aggregate properties
# ---------------------------------------------------------------------------

class TestEvalReport:
    def _make_result(self, prec, rec, hit, lat, faith=None, rel=None):
        return EvalResult(
            query="q", expected_answer="e", actual_answer="a",
            retrieved_doc_ids=["x"],
            latency_ms=lat,
            precision_at_k=prec,
            recall_at_k=rec,
            hit_rate=hit,
            faithfulness=faith,
            answer_relevance=rel,
        )

    def test_mean_precision_calculated_correctly(self):
        report = EvalReport()
        report.results = [
            self._make_result(0.5, 0.5, 1.0, 100),
            self._make_result(1.0, 1.0, 1.0, 200),
        ]
        assert report.mean_precision == pytest.approx(0.75)

    def test_mean_recall_calculated_correctly(self):
        report = EvalReport()
        report.results = [
            self._make_result(1.0, 0.0, 0.0, 100),
            self._make_result(1.0, 1.0, 1.0, 100),
        ]
        assert report.mean_recall == pytest.approx(0.5)

    def test_latency_percentiles_single_sample(self):
        report = EvalReport()
        report.results = [self._make_result(1.0, 1.0, 1.0, 250.0)]
        assert report.latency_p50 == pytest.approx(250.0)

    def test_mean_faithfulness_excludes_none(self):
        report = EvalReport()
        report.results = [
            self._make_result(1.0, 1.0, 1.0, 100, faith=0.8),
            self._make_result(1.0, 1.0, 1.0, 100, faith=None),
        ]
        assert report.mean_faithfulness == pytest.approx(0.8)

    def test_summary_dict_has_expected_keys(self):
        report = EvalReport()
        report.results = [self._make_result(1.0, 1.0, 1.0, 100)]
        summary = report.summary()
        for key in [
            "num_samples", "mean_precision_at_k", "mean_recall_at_k",
            "mean_hit_rate", "latency_p50_ms", "latency_p95_ms"
        ]:
            assert key in summary


# ---------------------------------------------------------------------------
# Tests: RAGEvaluator.evaluate
# ---------------------------------------------------------------------------

class TestRAGEvaluator:
    def test_evaluate_returns_report_with_correct_sample_count(self):
        samples = [
            EvalSample("Q1", "A1", ["doc1"]),
            EvalSample("Q2", "A2", ["doc2"]),
        ]

        def mock_retrieve(query):
            return (["doc1"], "Context for " + query)

        def mock_generate(query, context):
            return "Generated answer"

        evaluator = RAGEvaluator(k=5, use_llm_judge=False)
        report = evaluator.evaluate(samples, mock_retrieve, mock_generate)

        assert len(report.results) == 2

    def test_evaluate_computes_precision_correctly(self):
        samples = [EvalSample("Q", "A", ["doc1", "doc2"])]

        def mock_retrieve(query):
            return (["doc1", "doc3", "doc4"], "Context")

        def mock_generate(query, context):
            return "Answer"

        evaluator = RAGEvaluator(k=3, use_llm_judge=False)
        report = evaluator.evaluate(samples, mock_retrieve, mock_generate)

        # 1 out of 3 retrieved is relevant → precision = 1/3
        assert report.results[0].precision_at_k == pytest.approx(1 / 3)

    def test_evaluate_calls_llm_judge_when_enabled(self):
        samples = [EvalSample("Q", "A", ["doc1"])]

        def mock_retrieve(query):
            return (["doc1"], "Context")

        def mock_generate(query, context):
            return "Answer"

        evaluator = RAGEvaluator(k=5, use_llm_judge=True)

        with patch("src.evaluation.metrics.faithfulness_score", return_value=0.9) as mock_faith, \
             patch("src.evaluation.metrics.answer_relevance_score", return_value=0.8) as mock_rel, \
             patch.object(evaluator, "_get_llm", return_value=MagicMock()):
            report = evaluator.evaluate(samples, mock_retrieve, mock_generate)

        assert report.results[0].faithfulness == pytest.approx(0.9)
        assert report.results[0].answer_relevance == pytest.approx(0.8)

    def test_evaluate_measures_latency(self):
        samples = [EvalSample("Q", "A", ["doc1"])]

        def slow_retrieve(query):
            time.sleep(0.01)
            return (["doc1"], "ctx")

        def slow_generate(query, context):
            time.sleep(0.01)
            return "ans"

        evaluator = RAGEvaluator(k=5, use_llm_judge=False)
        report = evaluator.evaluate(samples, slow_retrieve, slow_generate)
        assert report.results[0].latency_ms > 0.0
