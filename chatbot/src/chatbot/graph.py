"""
LangGraph graph assembly for the parking chatbot.

Graph topology:
  START
    └─► input_guardrail
          ├─► (blocked) blocked_response ─► output_guardrail ─► END
          └─► (ok)      chatbot_agent
                ├─► (booking_requested=False) output_guardrail ─► END
                └─► (booking_requested=True)  booking_agent
                      └─► clear_booking_flag ─► output_guardrail ─► END

booking_agent contains an interrupt() inside request_admin_approval tool.
The graph pauses there until Streamlit resumes it with the admin's decision.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import weaviate
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.chatbot.chat_agent import create_chat_agent
from src.chatbot.booking_agent import create_booking_agent
from src.chatbot.nodes import (
    node_blocked_response,
    node_input_guardrail,
    node_output_guardrail,
)
from src.chatbot.state import AgentState

logger = logging.getLogger(__name__)

_CONFIG = {"configurable": {"thread_id": "default"}}


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class ChatResult:
    reply: str
    interrupted: bool = False
    interrupt_reservation_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def node_clear_booking_flag(_state: AgentState) -> dict:
    """Reset booking_requested so the chatbot agent routes normally next turn."""
    return {"booking_requested": False}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_after_guardrail(state: AgentState) -> Literal["blocked_response", "chatbot_agent"]:
    return "blocked_response" if state.get("input_blocked") else "chatbot_agent"


def _route_after_chatbot(state: AgentState) -> Literal["booking_agent", "output_guardrail"]:
    return "booking_agent" if state.get("booking_requested") else "output_guardrail"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(weaviate_client: weaviate.WeaviateClient, checkpointer=None) -> Any:
    """
    Compile and return the LangGraph app.

    Args:
        weaviate_client: Active Weaviate client — injected into the retrieval tool.
        checkpointer:    LangGraph checkpointer. Defaults to MemorySaver (non-persistent).
                         Pass a PostgresSaver in production for HITL interrupt/resume.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    chatbot_agent = create_chat_agent(weaviate_client)
    booking_agent = create_booking_agent()

    builder = StateGraph(AgentState)

    builder.add_node("input_guardrail", node_input_guardrail)
    builder.add_node("blocked_response", node_blocked_response)
    builder.add_node("chatbot_agent", chatbot_agent)
    builder.add_node("booking_agent", booking_agent)
    builder.add_node("clear_booking_flag", node_clear_booking_flag)
    builder.add_node("output_guardrail", node_output_guardrail)

    builder.add_edge(START, "input_guardrail")
    builder.add_conditional_edges(
        "input_guardrail",
        _route_after_guardrail,
        {"blocked_response": "blocked_response", "chatbot_agent": "chatbot_agent"},
    )
    builder.add_edge("blocked_response", "output_guardrail")
    builder.add_conditional_edges(
        "chatbot_agent",
        _route_after_chatbot,
        {"booking_agent": "booking_agent", "output_guardrail": "output_guardrail"},
    )
    builder.add_edge("booking_agent", "clear_booking_flag")
    builder.add_edge("clear_booking_flag", "output_guardrail")
    builder.add_edge("output_guardrail", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------------------------

def _last_ai_message(app: Any) -> str:
    snapshot = app.get_state(_CONFIG)
    messages = snapshot.values.get("messages", [])
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    return last_ai.content if last_ai else ""


def chat(app: Any, user_message: str) -> ChatResult:
    """
    Send a user message, run the graph, and return the result.

    If the booking agent hits an interrupt (awaiting admin approval) the result
    will have interrupted=True and interrupt_reservation_id set.
    """
    app.invoke({"messages": [HumanMessage(content=user_message)]}, _CONFIG)

    snapshot = app.get_state(_CONFIG)

    # Check for interrupt inside booking_agent
    for task in snapshot.tasks:
        for intr in (task.interrupts or []):
            value = intr.value if hasattr(intr, "value") else intr
            reservation_id = value.get("reservation_id") if isinstance(value, dict) else None
            reply = _last_ai_message(app) or "Your reservation is pending admin approval."
            return ChatResult(
                reply=reply,
                interrupted=True,
                interrupt_reservation_id=reservation_id,
            )

    return ChatResult(reply=_last_ai_message(app) or "I'm sorry, I couldn't generate a response.")


def resume_after_admin_decision(app: Any, status: str, reservation_id: int) -> ChatResult:
    """
    Resume the paused graph after the admin approves or rejects a reservation.

    Args:
        status:         "confirmed" or "cancelled"
        reservation_id: The reservation that was decided.
    """
    app.invoke(
        Command(resume={"status": status, "reservation_id": reservation_id}),
        _CONFIG,
    )
    return ChatResult(reply=_last_ai_message(app) or "Your booking has been processed.")
