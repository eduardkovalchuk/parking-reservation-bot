"""
Booking agent — Agent 2 in the HITL reservation flow.

Responsibilities:
  1. Create a pending reservation in PostgreSQL (no admin approval needed).
  2. Call request_admin_approval, which interrupts the graph until an admin
     decision arrives via the Admin API + Streamlit polling.
  3. After resuming, communicate the outcome (confirmed / cancelled) to the user.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated

import weaviate
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt

from src.chatbot.state import AgentState
from src.config import get_settings
from src.database import sql_store

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are the CityPark Booking Agent. You handle the final step of the reservation process.

The customer's details are already in state (collected by the assistant before you).

## Your workflow — follow it in order, no skipping steps

### Step 1 — Create the pending reservation
Call `create_pending_reservation`.
- If it fails (no available space), apologise to the user, explain what happened, \
and suggest they try different dates or a different space type.
- If it succeeds, proceed to Step 2.

### Step 2 — Request admin approval
Call `request_admin_approval` with the reservation_id you just received.
The graph will pause here until an admin reviews the request.

### Step 3 — Communicate the outcome
After the admin's decision arrives:
- **Confirmed**: congratulate the customer warmly and give the full details:
  Reservation ID, space number, floor, total cost.
- **Cancelled / rejected**: apologise professionally and invite the customer \
  to contact info@citypark.com or +31 20 555 0123 for assistance.
"""


def create_booking_tools() -> list:
    """Return tools for the booking agent."""

    @tool
    def create_pending_reservation(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Find an available space and create a pending reservation in the database.

        Reads all required fields from state — no parameters needed.
        Returns the reservation_id, space details, and total cost on success.
        """
        data: dict = state.get("reservation_data") or {}

        try:
            start = datetime.fromisoformat(data["start_datetime"])
            end = datetime.fromisoformat(data["end_datetime"])
        except (KeyError, ValueError) as exc:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Invalid datetime in reservation data: {exc}",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        space_type = data.get("space_type", "standard")
        space = sql_store.find_available_space(space_type, start, end)
        if not space:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=(
                                f"No available {space_type} spaces for "
                                f"{start.strftime('%Y-%m-%d %H:%M')} – "
                                f"{end.strftime('%Y-%m-%d %H:%M')}."
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        reservation = sql_store.create_reservation(
            space_id=space["id"],
            customer_name=data["name"],
            customer_surname=data["surname"],
            car_number=data["car_number"],
            start_datetime=start,
            end_datetime=end,
        )

        updated_data = {
            **data,
            "reservation_id": reservation["id"],
            "space_id": space["id"],
            "floor": space["floor"],
            "space_number": space["space_number"],
            "total_cost": float(reservation["total_cost"]),
        }

        logger.info(
            "Created pending reservation #%d — space %s floor %s",
            reservation["id"],
            space["space_number"],
            space["floor"],
        )

        return Command(
            update={
                "reservation_data": updated_data,
                "messages": [
                    ToolMessage(
                        content=(
                            f"Pending reservation created — "
                            f"ID #{reservation['id']}, "
                            f"Space {space['space_number']} (Floor {space['floor']}), "
                            f"Cost €{reservation['total_cost']:.2f}. "
                            "Now requesting admin approval."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    @tool
    def request_admin_approval(reservation_id: int) -> str:
        """Interrupt the graph and wait for an admin to approve or reject the reservation.

        The graph resumes automatically once the admin makes a decision via the Admin UI.
        Returns the admin's decision so you can inform the customer.
        """
        decision = interrupt({"reservation_id": reservation_id})
        status = decision.get("status", "unknown") if isinstance(decision, dict) else str(decision)
        logger.info("Admin decision for reservation #%d: %s", reservation_id, status)
        return f"Admin decision: {status}"

    return [create_pending_reservation, request_admin_approval]


def create_booking_agent() -> object:
    """Build and return the compiled booking agent subgraph."""
    settings = get_settings()

    llm = ChatOpenAI(
        model=settings.openai_chat_model,
        openai_api_key=settings.openai_api_key,
        temperature=0,
    )

    return create_agent(
        model=llm,
        tools=create_booking_tools(),
        state_schema=AgentState,
        system_prompt=_SYSTEM_PROMPT,
    )
