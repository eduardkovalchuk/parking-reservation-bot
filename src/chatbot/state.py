"""
LangGraph state definition for the parking chatbot.

The graph is compiled with a MemorySaver checkpointer so each user session
(identified by a thread_id) has its own persistent conversation state.
"""
from __future__ import annotations

import operator
from typing import Annotated, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ReservationData(TypedDict, total=False):
    """Fields collected during the multi-turn reservation flow."""

    name: str
    surname: str
    car_number: str
    start_datetime: str      # ISO-format string; parsed when saving
    end_datetime: str        # ISO-format string
    space_type: str          # default "standard"
    space_id: Optional[int]
    reservation_id: Optional[int]
    total_cost: Optional[float]


class ChatState(TypedDict):
    """Full state carried through every graph node for a single turn."""

    # Conversation history (LangGraph reducer: append-only)
    messages: Annotated[List[BaseMessage], add_messages]

    # Raw text of the latest user input
    user_input: str

    # Classified intent for the current turn
    # Values: "information" | "reservation" | "general" | "off_topic"
    intent: str

    # Retrieved context string (static + dynamic)
    retrieved_context: str

    # Multi-turn reservation state machine
    # Stages: "idle" | "need_name" | "need_surname" | "need_car_number"
    #         | "need_start" | "need_end" | "confirming" | "completed" | "cancelled"
    reservation_stage: str
    reservation_data: ReservationData

    # Guardrail flags
    input_blocked: bool
    output_blocked: bool
    block_reason: str
