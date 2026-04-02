"""
Tests for src/chatbot/nodes.py and src/chatbot/graph.py

Tests verify node behaviour under various state conditions using mocked
LLM responses and mocked database calls.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.chatbot.nodes import (
    node_blocked_response,
    node_classify_intent,
    node_generate_response,
    node_handle_reservation,
    node_input_guardrail,
    node_off_topic_response,
    node_output_guardrail,
)
from src.chatbot.state import ChatState


def _base_state(**kwargs) -> dict:
    """Build a minimal valid state dict with sensible defaults."""
    state = {
        "messages": [],
        "user_input": "",
        "intent": "general",
        "retrieved_context": "",
        "reservation_stage": "idle",
        "reservation_data": {},
        "input_blocked": False,
        "output_blocked": False,
        "block_reason": "",
    }
    state.update(kwargs)
    return state


# ---------------------------------------------------------------------------
# Tests: node_input_guardrail
# ---------------------------------------------------------------------------

class TestInputGuardrailNode:
    def test_clean_input_not_blocked(self):
        state = _base_state(user_input="What is the hourly price?")
        result = node_input_guardrail(state)
        assert result.get("input_blocked") is False

    def test_credit_card_input_is_blocked(self):
        state = _base_state(user_input="My card number is 4111111111111111")
        result = node_input_guardrail(state)
        assert result.get("input_blocked") is True


# ---------------------------------------------------------------------------
# Tests: node_blocked_response
# ---------------------------------------------------------------------------

class TestBlockedResponseNode:
    def test_returns_ai_message(self):
        state = _base_state(block_reason="Contains credit card data.")
        result = node_blocked_response(state)
        messages = result.get("messages", [])
        assert len(messages) == 1
        assert isinstance(messages[0], AIMessage)

    def test_reply_mentions_reason(self):
        state = _base_state(block_reason="Contains credit card data.")
        result = node_blocked_response(state)
        content = result["messages"][0].content
        assert "credit card" in content.lower() or "cannot process" in content.lower()


# ---------------------------------------------------------------------------
# Tests: node_classify_intent
# ---------------------------------------------------------------------------

class TestClassifyIntentNode:
    def test_reservation_intent_detected(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"intent": "reservation"}')

        state = _base_state(user_input="I want to book a space", reservation_stage="idle")
        with patch("src.chatbot.nodes._get_llm", return_value=mock_llm):
            result = node_classify_intent(state)

        assert result["intent"] == "reservation"

    def test_information_intent_detected(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"intent": "information"}')

        state = _base_state(user_input="What is the hourly price?")
        with patch("src.chatbot.nodes._get_llm", return_value=mock_llm):
            result = node_classify_intent(state)

        assert result["intent"] == "information"

    def test_mid_reservation_stays_reservation(self):
        """When reservation_stage is not 'idle', intent should be forced to reservation."""
        state = _base_state(
            user_input="John",
            reservation_stage="need_name",
        )
        # No LLM call expected when in mid-reservation
        result = node_classify_intent(state)
        assert result["intent"] == "reservation"

    def test_llm_failure_defaults_to_general(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM error")

        state = _base_state(user_input="Something random")
        with patch("src.chatbot.nodes._get_llm", return_value=mock_llm):
            result = node_classify_intent(state)

        assert result["intent"] == "general"


# ---------------------------------------------------------------------------
# Tests: node_generate_response
# ---------------------------------------------------------------------------

class TestGenerateResponseNode:
    def test_returns_ai_message_with_content(self):
        mock_answer = "The parking is located at 123 Main Street."
        state = _base_state(
            user_input="Where are you?",
            retrieved_context="Located at 123 Main Street.",
        )
        with patch("src.chatbot.nodes.rag_chain.generate_answer", return_value=mock_answer):
            result = node_generate_response(state)

        messages = result.get("messages", [])
        assert len(messages) == 1
        assert messages[0].content == mock_answer

    def test_no_context_returns_fallback(self):
        state = _base_state(user_input="Some question", retrieved_context="")
        result = node_generate_response(state)
        content = result["messages"][0].content.lower()
        assert "contact" in content or "sorry" in content or "don't have" in content


# ---------------------------------------------------------------------------
# Tests: node_handle_reservation
# ---------------------------------------------------------------------------

class TestHandleReservationNode:
    def test_idle_stage_starts_reservation(self):
        state = _base_state(reservation_stage="idle")
        result = node_handle_reservation(state)
        assert result["reservation_stage"] == "need_name"
        assert len(result["messages"]) == 1

    def test_need_name_stage_extracts_name(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"name": "Alice"}')

        state = _base_state(user_input="My name is Alice", reservation_stage="need_name")
        with patch("src.chatbot.nodes._get_llm", return_value=mock_llm):
            result = node_handle_reservation(state)

        assert result["reservation_data"].get("name") == "Alice"
        assert result["reservation_stage"] == "need_surname"

    def test_extraction_failure_stays_on_same_stage(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"name": null}')

        state = _base_state(user_input="asdfghjkl", reservation_stage="need_name")
        with patch("src.chatbot.nodes._get_llm", return_value=mock_llm):
            result = node_handle_reservation(state)

        assert result["reservation_stage"] == "need_name"

    def test_cancellation_at_confirm_stage(self):
        state = _base_state(
            user_input="no",
            reservation_stage="confirming",
            reservation_data={"name": "John", "surname": "Doe"},
        )
        result = node_handle_reservation(state)
        assert result["reservation_stage"] == "cancelled"
        assert result.get("reservation_data") == {}

    def test_confirmation_calls_create_reservation(self, sample_reservation_data):
        mock_space = {"id": 1, "floor": "B1", "space_number": "S01", "space_type": "standard"}
        mock_reservation = {
            "id": 99, "space_id": 1, "customer_name": "John", "customer_surname": "Doe",
            "car_number": "ABC-1234",
            "start_datetime": "2026-04-01T09:00:00",
            "end_datetime": "2026-04-01T17:00:00",
            "total_cost": 24.00, "status": "pending",
        }
        state = _base_state(
            user_input="yes",
            reservation_stage="confirming",
            reservation_data=sample_reservation_data,
        )
        with patch("src.chatbot.nodes.sql_store.find_available_space", return_value=mock_space), \
             patch("src.chatbot.nodes.sql_store.create_reservation", return_value=mock_reservation):
            result = node_handle_reservation(state)

        assert result["reservation_stage"] == "completed"
        assert "99" in result["messages"][0].content  # Reservation ID in reply


# ---------------------------------------------------------------------------
# Tests: node_off_topic_response
# ---------------------------------------------------------------------------

class TestOffTopicResponseNode:
    def test_returns_ai_message(self):
        state = _base_state(user_input="Tell me a joke.")
        result = node_off_topic_response(state)
        messages = result.get("messages", [])
        assert len(messages) == 1
        assert isinstance(messages[0], AIMessage)

    def test_reply_mentions_parking(self):
        state = _base_state(user_input="What is the weather like?")
        result = node_off_topic_response(state)
        content = result["messages"][0].content.lower()
        assert "parking" in content


# ---------------------------------------------------------------------------
# Tests: node_output_guardrail
# ---------------------------------------------------------------------------

class TestOutputGuardrailNode:
    def test_clean_response_not_blocked(self):
        state = _base_state(
            messages=[AIMessage(content="The parking is at Orlyplein 10, Amsterdam. Contact info@citypark.com.")]
        )
        result = node_output_guardrail(state)
        assert result.get("output_blocked") is not True

    def test_sensitive_response_is_blocked(self):
        state = _base_state(
            messages=[AIMessage(content="The customer's credit card is 4111 1111 1111 1111.")]
        )
        result = node_output_guardrail(state)
        # Should replace with a safe reply
        assert result.get("output_blocked") is True or (
            result.get("messages") and "privacy" in result["messages"][-1].content.lower()
        )
