"""
LangGraph state definition for the parking chatbot.
"""
from __future__ import annotations

from typing import Optional

from langgraph.graph import MessagesState
from typing_extensions import TypedDict


class ReservationData(TypedDict, total=False):
    """Fields collected during the reservation flow."""

    name: str
    surname: str
    car_number: str
    start_datetime: str     # ISO-format string
    end_datetime: str       # ISO-format string
    space_type: str         # default "standard"
    space_id: Optional[int]
    floor: Optional[str]
    space_number: Optional[str]
    reservation_id: Optional[int]
    total_cost: Optional[float]


class AgentState(MessagesState):
    """State carried through the graph for a single conversation thread."""

    reservation_data: ReservationData
    input_blocked: bool
    block_reason: str
    booking_requested: bool  # set True by Agent 1 when user confirms booking
