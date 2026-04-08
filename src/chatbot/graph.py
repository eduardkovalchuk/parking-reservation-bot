"""
LangGraph graph assembly for the parking chatbot.

Graph topology:
  START
    └─► input_guardrail
          ├─► (blocked)   blocked_response ─► output_guardrail ─► END
          └─► (ok)        classify_intent
                              ├─► "information"  retrieve_context ─► generate_response ─► output_guardrail ─► END
                              ├─► "reservation"  handle_reservation ─► output_guardrail ─► END
                              ├─► "general"      general_response   ─► output_guardrail ─► END
                              └─► "off_topic"    off_topic_response ─► END
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Literal

import weaviate
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.chatbot.nodes import (
    node_blocked_response,
    node_classify_intent,
    node_general_response,
    node_generate_response,
    node_handle_reservation,
    node_input_guardrail,
    node_off_topic_response,
    node_output_guardrail,
    node_retrieve_context,
)
from src.chatbot.state import ChatState

logger = logging.getLogger(__name__)


def _route_after_guardrail(state: ChatState) -> Literal["blocked_response", "classify_intent"]:
    if state.get("input_blocked"):
        return "blocked_response"
    return "classify_intent"


def _route_after_classification(
    state: ChatState,
) -> Literal["retrieve_context", "handle_reservation", "general_response", "off_topic_response"]:
    intent = state.get("intent", "general")
    if intent == "information":
        return "retrieve_context"
    if intent == "reservation":
        return "handle_reservation"
    if intent == "off_topic":
        return "off_topic_response"
    return "general_response"


def build_graph(weaviate_client: weaviate.WeaviateClient) -> Any:
    """
    Compile and return the LangGraph StateGraph.

    Args:
        weaviate_client: An active Weaviate client injected into retrieval nodes.

    Returns:
        A compiled LangGraph app with MemorySaver checkpointing.
    """
    # Bind weaviate_client to the retrieval node (partial application)
    retrieve_node: Callable = functools.partial(
        node_retrieve_context, weaviate_client=weaviate_client
    )

    builder = StateGraph(ChatState)

    # ── Register nodes ─────────────────────────────────────────────────────
    builder.add_node("input_guardrail", node_input_guardrail)
    builder.add_node("classify_intent", node_classify_intent)
    builder.add_node("retrieve_context", retrieve_node)
    builder.add_node("generate_response", node_generate_response)
    builder.add_node("handle_reservation", node_handle_reservation)
    builder.add_node("general_response", node_general_response)
    builder.add_node("off_topic_response", node_off_topic_response)
    builder.add_node("blocked_response", node_blocked_response)
    builder.add_node("output_guardrail", node_output_guardrail)

    # ── Wire edges ─────────────────────────────────────────────────────────
    builder.add_edge(START, "input_guardrail")

    builder.add_conditional_edges(
        "input_guardrail",
        _route_after_guardrail,
        {
            "blocked_response": "blocked_response",
            "classify_intent": "classify_intent",
        },
    )

    builder.add_conditional_edges(
        "classify_intent",
        _route_after_classification,
        {
            "retrieve_context": "retrieve_context",
            "handle_reservation": "handle_reservation",
            "general_response": "general_response",
            "off_topic_response": "off_topic_response",
        },
    )

    builder.add_edge("retrieve_context", "generate_response")
    builder.add_edge("generate_response", "output_guardrail")
    builder.add_edge("handle_reservation", "output_guardrail")
    builder.add_edge("general_response", "output_guardrail")
    builder.add_edge("blocked_response", "output_guardrail")
    builder.add_edge("off_topic_response", END)
    builder.add_edge("output_guardrail", END)

    # ── Compile with in-memory checkpointer ───────────────────────────────
    checkpointer = MemorySaver()
    app = builder.compile(checkpointer=checkpointer)
    return app


def chat(
    app: Any,
    user_message: str,
) -> str:
    """
    Send a user message to the compiled graph and return the assistant's reply.

    Args:
        app: Compiled LangGraph app from build_graph().
        user_message: The raw user input string.

    Returns:
        The assistant's reply as a plain string.
    """
    config = {"configurable": {"thread_id": "default"}}
    input_state: dict = {
        "messages": [HumanMessage(content=user_message)],
        "user_input": user_message,
    }

    result = app.invoke(input_state, config=config)

    messages = result.get("messages", [])
    if messages:
        last_ai = next(
            (m for m in reversed(messages) if hasattr(m, "content") and not isinstance(m, HumanMessage)),
            None,
        )
        if last_ai:
            return last_ai.content

    return "I'm sorry, I couldn't generate a response. Please try again."
