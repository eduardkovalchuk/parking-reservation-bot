"""
Guardrail node functions for the parking chatbot graph.

These are the only nodes in the outer graph — the agent itself lives in agent.py.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage

from src.chatbot.state import AgentState
from src.guardrails.filters import GuardrailFilter

logger = logging.getLogger(__name__)


def node_input_guardrail(state: AgentState) -> Dict[str, Any]:
    """Block user input that contains sensitive financial / identity PII."""
    guardrail = GuardrailFilter()

    messages = state.get("messages", [])
    last_human = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )
    if not last_human:
        return {"input_blocked": False, "block_reason": ""}

    result = guardrail.check_input(last_human.content)
    if result.blocked:
        logger.info("Input blocked by guardrail: %s", result.reason)
        return {"input_blocked": True, "block_reason": result.reason}

    return {"input_blocked": False, "block_reason": ""}


def node_blocked_response(state: AgentState) -> Dict[str, Any]:
    """Return a polite refusal when input was blocked by the guardrail."""
    reason = state.get("block_reason", "Your message could not be processed.")
    reply = (
        f"I'm sorry, I cannot process that request. {reason} "
        "If you have questions about our parking facility, I'm happy to help!"
    )
    return {"messages": [AIMessage(content=reply)]}


def node_output_guardrail(state: AgentState) -> Dict[str, Any]:
    """Inspect the last AI message and redact / block any leaked sensitive data."""
    guardrail = GuardrailFilter()

    messages = state.get("messages", [])
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)), None
    )
    if not last_ai:
        return {}

    result = guardrail.check_output(last_ai.content)
    if result.blocked:
        logger.warning("Output blocked by guardrail: %s", result.reason)
        safe_reply = (
            "I'm sorry, I can't provide that information for privacy reasons. "
            "Please contact us directly at info@citypark.com or +31 20 555 0123."
        )
        return {"messages": [AIMessage(content=safe_reply)]}

    anonymised = guardrail.anonymize(last_ai.content)
    if anonymised != last_ai.content:
        return {"messages": [AIMessage(content=anonymised)]}

    return {}
