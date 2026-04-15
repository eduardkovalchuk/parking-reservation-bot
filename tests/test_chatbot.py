"""
Tests for src/chatbot/nodes.py and src/chatbot/tools.py

Covers:
  - Guardrail nodes (input, output, blocked response)
  - ReAct agent tools (retrieve_parking_info, get_reservation_draft, update_reservation_draft)
"""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from src.chatbot.nodes import (
    node_blocked_response,
    node_input_guardrail,
    node_output_guardrail,
)
from src.chatbot.tools import create_tools


def _base_state(**kwargs) -> dict:
    """Build a minimal valid AgentState dict with sensible defaults."""
    state = {
        "messages": [],
        "reservation_data": {},
        "input_blocked": False,
        "block_reason": "",
    }
    state.update(kwargs)
    return state


def _get_tool(tools: list, name: str):
    """Find a tool by name from the list returned by create_tools()."""
    return next(t for t in tools if t.name == name)


# ---------------------------------------------------------------------------
# Tests: node_input_guardrail
# ---------------------------------------------------------------------------

class TestInputGuardrailNode:
    def test_clean_input_not_blocked(self):
        state = _base_state(messages=[HumanMessage(content="What is the hourly price?")])
        result = node_input_guardrail(state)
        assert result.get("input_blocked") is False

    def test_credit_card_input_is_blocked(self):
        state = _base_state(messages=[HumanMessage(content="My card number is 4111111111111111")])
        result = node_input_guardrail(state)
        assert result.get("input_blocked") is True

    def test_no_messages_returns_not_blocked(self):
        state = _base_state(messages=[])
        result = node_input_guardrail(state)
        assert result.get("input_blocked") is False

    def test_name_and_plate_are_allowed(self):
        state = _base_state(messages=[HumanMessage(content="My name is John Doe, plate ABC-1234")])
        result = node_input_guardrail(state)
        assert result.get("input_blocked") is False


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
# Tests: node_output_guardrail
# ---------------------------------------------------------------------------

class TestOutputGuardrailNode:
    def test_clean_response_passes_through(self):
        state = _base_state(
            messages=[AIMessage(content="The parking is at Orlyplein 10. Contact info@citypark.com.")]
        )
        result = node_output_guardrail(state)
        # No replacement — result is empty or has no blocked indicator
        if result.get("messages"):
            assert "privacy" not in result["messages"][-1].content.lower()

    def test_sensitive_response_is_replaced(self):
        state = _base_state(
            messages=[AIMessage(content="The customer's card is 4111 1111 1111 1111.")]
        )
        result = node_output_guardrail(state)
        assert result.get("messages") and "privacy" in result["messages"][-1].content.lower()

    def test_no_messages_returns_empty(self):
        state = _base_state(messages=[])
        result = node_output_guardrail(state)
        assert result == {}

    def test_non_ai_last_message_returns_empty(self):
        state = _base_state(messages=[HumanMessage(content="Hello")])
        result = node_output_guardrail(state)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: retrieve_parking_info tool
# ---------------------------------------------------------------------------

class TestRetrieveParkingInfoTool:
    def setup_method(self):
        self.mock_client = MagicMock()
        self.tools = create_tools(self.mock_client)
        self.tool = _get_tool(self.tools, "retrieve_parking_info")

    def test_returns_combined_context(self):
        mock_result = MagicMock()
        mock_result.combined_context = "Hourly rate: EUR 3.00"
        with patch("src.chatbot.tools.rag_retriever.retrieve", return_value=mock_result):
            result = self.tool.func(query="What is the hourly rate?")
        assert result == "Hourly rate: EUR 3.00"

    def test_empty_context_returns_fallback(self):
        mock_result = MagicMock()
        mock_result.combined_context = ""
        with patch("src.chatbot.tools.rag_retriever.retrieve", return_value=mock_result):
            result = self.tool.func(query="anything")
        assert "no relevant" in result.lower()

    def test_retrieval_failure_returns_fallback(self):
        with patch("src.chatbot.tools.rag_retriever.retrieve", side_effect=Exception("connection error")):
            result = self.tool.func(query="anything")
        assert "unavailable" in result.lower() or "citypark" in result.lower()

    def test_passes_query_to_retriever(self):
        mock_result = MagicMock()
        mock_result.combined_context = "some context"
        with patch("src.chatbot.tools.rag_retriever.retrieve", return_value=mock_result) as mock_retrieve:
            self.tool.func(query="EV charging?")
        call_args = mock_retrieve.call_args
        assert call_args[0][0] == "EV charging?" or call_args[1].get("query") == "EV charging?"


# ---------------------------------------------------------------------------
# Tests: get_reservation_draft tool
# ---------------------------------------------------------------------------

class TestGetReservationDraftTool:
    def setup_method(self):
        self.tools = create_tools(MagicMock())
        self.tool = _get_tool(self.tools, "get_reservation_draft")

    def test_empty_state_reports_no_data(self):
        result = self.tool.func(state={"reservation_data": {}})
        assert "no reservation data" in result.lower()

    def test_empty_reservation_data_none(self):
        result = self.tool.func(state={"reservation_data": None})
        assert "no reservation data" in result.lower()

    def test_partial_data_shows_missing_fields(self):
        state = {"reservation_data": {"name": "John", "surname": "Doe"}}
        result = self.tool.func(state=state)
        assert "MISSING" in result
        assert "car_number" in result or "start_datetime" in result

    def test_all_required_fields_shows_ready(self):
        state = {
            "reservation_data": {
                "name": "John",
                "surname": "Doe",
                "car_number": "ABC-1234",
                "start_datetime": "2026-04-01T09:00:00",
                "end_datetime": "2026-04-01T17:00:00",
            }
        }
        result = self.tool.func(state=state)
        assert "ready" in result.lower() or "collected" in result.lower()

    def test_collected_fields_appear_in_output(self):
        state = {"reservation_data": {"name": "Alice"}}
        result = self.tool.func(state=state)
        assert "Alice" in result


# ---------------------------------------------------------------------------
# Tests: update_reservation_draft tool
# ---------------------------------------------------------------------------

class TestUpdateReservationDraftTool:
    def setup_method(self):
        self.tools = create_tools(MagicMock())
        self.tool = _get_tool(self.tools, "update_reservation_draft")

    def _invoke(self, state: dict, **fields):
        return self.tool.func(
            state=state,
            tool_call_id="test-call-id",
            **fields,
        )

    def test_returns_command(self):
        result = self._invoke({"reservation_data": {}}, name="John")
        assert isinstance(result, Command)

    def test_new_field_is_saved(self):
        result = self._invoke({"reservation_data": {}}, name="John")
        assert result.update["reservation_data"]["name"] == "John"

    def test_existing_fields_are_preserved(self):
        state = {"reservation_data": {"name": "John"}}
        result = self._invoke(state, surname="Doe")
        assert result.update["reservation_data"]["name"] == "John"
        assert result.update["reservation_data"]["surname"] == "Doe"

    def test_multiple_fields_saved_at_once(self):
        result = self._invoke(
            {"reservation_data": {}},
            name="Ed",
            surname="Lee",
            car_number="XYZ-999",
        )
        data = result.update["reservation_data"]
        assert data["name"] == "Ed"
        assert data["surname"] == "Lee"
        assert data["car_number"] == "XYZ-999"

    def test_none_fields_are_not_saved(self):
        result = self._invoke({"reservation_data": {"name": "John"}}, surname=None)
        assert "surname" not in result.update["reservation_data"]

    def test_tool_message_included_in_update(self):
        from langchain_core.messages import ToolMessage
        result = self._invoke({"reservation_data": {}}, name="John")
        messages = result.update.get("messages", [])
        assert any(isinstance(m, ToolMessage) for m in messages)

    def test_tool_message_has_correct_tool_call_id(self):
        from langchain_core.messages import ToolMessage
        result = self._invoke({"reservation_data": {}}, name="John")
        tool_msgs = [m for m in result.update.get("messages", []) if isinstance(m, ToolMessage)]
        assert tool_msgs[0].tool_call_id == "test-call-id"
