"""
LangGraph graph assembly for the parking chatbot.

Graph topology:
  START
    └─► input_guardrail
          ├─► (blocked) blocked_response ─► output_guardrail ─► END
          └─► (ok)      agent ────────────► output_guardrail ─► END

The 'agent' node is a compiled create_react_agent subgraph that loops
internally between its LLM node and its tools node until it produces
a final AI reply.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

import weaviate
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.chatbot.agent import create_parking_agent
from src.chatbot.nodes import (
    node_blocked_response,
    node_input_guardrail,
    node_output_guardrail,
)
from src.chatbot.state import AgentState

logger = logging.getLogger(__name__)


def _route_after_guardrail(
    state: AgentState,
) -> Literal["blocked_response", "agent"]:
    if state.get("input_blocked"):
        return "blocked_response"
    return "agent"


def build_graph(weaviate_client: weaviate.WeaviateClient) -> Any:
    """
    Compile and return the LangGraph app.

    Args:
        weaviate_client: Active Weaviate client — injected into the retrieval tool.

    Returns:
        Compiled LangGraph app with MemorySaver checkpointing.
    """
    agent = create_parking_agent(weaviate_client)

    builder = StateGraph(AgentState)

    builder.add_node("input_guardrail", node_input_guardrail)
    builder.add_node("blocked_response", node_blocked_response)
    builder.add_node("agent", agent)
    builder.add_node("output_guardrail", node_output_guardrail)

    builder.add_edge(START, "input_guardrail")
    builder.add_conditional_edges(
        "input_guardrail",
        _route_after_guardrail,
        {"blocked_response": "blocked_response", "agent": "agent"},
    )
    builder.add_edge("blocked_response", "output_guardrail")
    builder.add_edge("agent", "output_guardrail")
    builder.add_edge("output_guardrail", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def chat(app: Any, user_message: str) -> str:
    """
    Send a user message to the graph and return the assistant's reply.

    Args:
        app:          Compiled LangGraph app from build_graph().
        user_message: Raw user input string.

    Returns:
        The assistant's reply as a plain string.
    """
    config = {"configurable": {"thread_id": "default"}}
    input_state = {"messages": [HumanMessage(content=user_message)]}

    result = app.invoke(input_state, config=config)

    messages = result.get("messages", [])
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)),
        None,
    )
    if last_ai:
        return last_ai.content

    return "I'm sorry, I couldn't generate a response. Please try again."
