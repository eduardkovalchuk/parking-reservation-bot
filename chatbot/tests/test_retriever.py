"""
Tests for src/rag/retriever.py and src/rag/chain.py
"""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.rag.retriever import (
    RetrievalResult,
    _classify_query,
    _fetch_dynamic_context,
    retrieve,
)


# ---------------------------------------------------------------------------
# Tests: _classify_query
# ---------------------------------------------------------------------------

class TestClassifyQuery:
    """Tests for the private query classifier."""

    def test_price_query_needs_dynamic(self):
        needs_static, needs_dynamic = _classify_query("What is the hourly price?")
        assert needs_dynamic is True

    def test_location_query_needs_static_only(self):
        needs_static, needs_dynamic = _classify_query("Where is the parking located?")
        # Location is static; dynamic keywords may not be present
        assert needs_static is True

    def test_availability_query_needs_dynamic(self):
        needs_static, needs_dynamic = _classify_query("Are there any parking spaces available?")
        assert needs_dynamic is True

    def test_ev_query_needs_dynamic(self):
        needs_static, needs_dynamic = _classify_query("Do you have EV charging?")
        assert needs_dynamic is True

    def test_general_question_still_needs_static(self):
        needs_static, needs_dynamic = _classify_query("Tell me about the parking amenities")
        assert needs_static is True


# ---------------------------------------------------------------------------
# Tests: _fetch_dynamic_context
# ---------------------------------------------------------------------------

class TestFetchDynamicContext:
    def test_price_keyword_fetches_prices(self):
        mock_prices = [
            {"price_type": "hourly", "amount": 3.00, "currency": "EUR", "description": "Hourly"},
        ]
        with patch("src.rag.retriever.sql_store.get_all_prices", return_value=mock_prices):
            context = _fetch_dynamic_context("What is the price?")
        assert "Pricing" in context or "hourly" in context.lower()

    def test_availability_keyword_fetches_availability(self):
        mock_summary = {
            "total_available": 50,
            "by_type": [{"space_type": "standard", "total_available": 50, "total_spaces": 100}],
        }
        with patch("src.rag.retriever.sql_store.get_availability_summary", return_value=mock_summary):
            context = _fetch_dynamic_context("How many spaces are available?")
        assert "50" in context or "available" in context.lower()

    def test_hours_keyword_fetches_working_hours(self):
        mock_hours = [{"day_of_week": "Monday", "is_24h": True, "open_time": None, "close_time": None}]
        with patch("src.rag.retriever.sql_store.get_working_hours", return_value=mock_hours):
            context = _fetch_dynamic_context("What are the opening hours?")
        assert "24" in context or "Hours" in context

    def test_unrelated_query_returns_empty(self):
        context = _fetch_dynamic_context("Tell me about parking rules")
        # No dynamic keywords — should return empty string
        assert context == ""


# ---------------------------------------------------------------------------
# Tests: retrieve (integration-level with mocks)
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_retrieve_returns_retrieval_result(self, mock_weaviate_client):
        docs = [Document(page_content="Parking info", metadata={"source": "parking_info.md", "category": "general"})]
        with patch("src.rag.retriever.vector_store.similarity_search", return_value=docs):
            result = retrieve("Where are you?", mock_weaviate_client)
        assert isinstance(result, RetrievalResult)
        assert len(result.static_docs) == 1

    def test_retrieve_combined_context_not_empty(self, mock_weaviate_client):
        docs = [Document(page_content="Located at 123 Main St", metadata={"category": "location"})]
        with patch("src.rag.retriever.vector_store.similarity_search", return_value=docs):
            result = retrieve("Where are you located?", mock_weaviate_client)
        assert result.combined_context != ""

    def test_retrieve_dynamic_context_for_price_query(self, mock_weaviate_client):
        mock_prices = [{"price_type": "hourly", "amount": 3.00, "currency": "EUR", "description": ""}]
        with patch("src.rag.retriever.vector_store.similarity_search", return_value=[]), \
             patch("src.rag.retriever.sql_store.get_all_prices", return_value=mock_prices):
            result = retrieve("What is the hourly price?", mock_weaviate_client)
        # Dynamic context should be populated
        assert "hourly" in result.dynamic_context.lower() or "3" in result.dynamic_context

    def test_retrieve_handles_weaviate_error_gracefully(self, mock_weaviate_client):
        """If Weaviate fails, retrieve should return a result with empty static_docs."""
        with patch(
            "src.rag.retriever.vector_store.similarity_search",
            side_effect=Exception("Weaviate connection error"),
        ):
            result = retrieve("Where are you?", mock_weaviate_client)
        assert result.static_docs == []


# ---------------------------------------------------------------------------
# Tests: RetrievalResult.to_context_string
# ---------------------------------------------------------------------------

class TestRetrievalResultContextString:
    def test_empty_result_returns_empty_string(self):
        r = RetrievalResult()
        assert r.to_context_string() == ""

    def test_static_docs_appear_in_context(self):
        doc = Document(page_content="Hello world", metadata={"source": "test.md"})
        r = RetrievalResult(static_docs=[doc])
        ctx = r.to_context_string()
        assert "Hello world" in ctx

    def test_dynamic_context_appears_in_context(self):
        r = RetrievalResult(dynamic_context="Hourly price: $3.00")
        ctx = r.to_context_string()
        assert "Hourly price: $3.00" in ctx

    def test_both_sources_merged_with_separator(self):
        doc = Document(page_content="Static info", metadata={"source": "test.md"})
        r = RetrievalResult(static_docs=[doc], dynamic_context="Dynamic info")
        ctx = r.to_context_string()
        assert "Static info" in ctx
        assert "Dynamic info" in ctx


# ---------------------------------------------------------------------------
# Tests: generate_answer (chain)
# ---------------------------------------------------------------------------

class TestGenerateAnswer:
    def test_returns_string_answer(self):
        from src.rag.chain import generate_answer
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Located at 123 Main Street."

        with patch("src.rag.chain.build_rag_chain", return_value=mock_chain):
            answer = generate_answer("Where are you?", "Parking at 123 Main Street.")

        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_chain_receives_correct_inputs(self):
        from src.rag.chain import generate_answer
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Answer."

        with patch("src.rag.chain.build_rag_chain", return_value=mock_chain):
            generate_answer("My question", "My context")

        call_args = mock_chain.invoke.call_args[0][0]
        assert call_args["question"] == "My question"
        assert call_args["context"] == "My context"
