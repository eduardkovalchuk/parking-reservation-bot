"""
Booking agent — Agent 2 in the HITL reservation flow.

Responsibilities:
  1. Create a pending reservation in PostgreSQL.
  2. Interrupt the graph until an admin approves or rejects via the Admin UI.
  3. Return a structured BookingResult consumed by Agent 1 (chat agent),
     which composes the final user-facing response.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from datetime import datetime, timezone
from typing import Annotated, Literal, Optional

from fastmcp import Client as MCPClient
from fastmcp.client.transports import SSETransport
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from src.chatbot.state import AgentState
from src.config import get_settings
from src.database import sql_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class BookingResult(BaseModel):
    """Structured outcome from the booking process, handed off to the chat agent."""

    status: Literal["confirmed", "cancelled", "no_space_available"] = Field(
        description="Final status of the booking attempt."
    )
    reservation_id: Optional[int] = Field(
        None, description="Reservation ID if a record was created."
    )
    space_number: Optional[str] = Field(None, description="Assigned parking space number.")
    floor: Optional[str] = Field(None, description="Floor of the assigned space.")
    total_cost: Optional[float] = Field(None, description="Total cost in EUR.")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the CityPark Booking Agent — a backend worker.
The customer's details are already in state.

## Workflow — follow in order

### Step 1 — Create the pending reservation
Call `create_pending_reservation`. It reads all required fields from state automatically.

### Step 2 — Request admin approval
If Step 1 succeeded, call `request_admin_approval` with the reservation_id.
The graph pauses here until an admin decides.

### Step 3 — Log the confirmed reservation (approved path only)
If the admin approved, call `write_reservation_to_log`.
It reads all required data from state automatically.
If logging fails, continue anyway — the reservation is already confirmed in the database.
Skip this step if the admin rejected or no space was available.

### Step 4 — Return a structured BookingResult
After every outcome (success, rejection, or no space), output a BookingResult:
- Admin approved  → status="confirmed", fill reservation_id, space_number, floor, total_cost
- Admin rejected  → status="cancelled"
- No space found  → status="no_space_available"

Do NOT write any conversational message to the user — your only output is the BookingResult struct.
The chat agent will read it and compose the user-facing reply.
"""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def create_booking_tools() -> list:

    @tool
    def create_pending_reservation(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Find an available space and create a pending reservation in the database.

        Reads all required fields from state — no parameters needed.
        Returns reservation_id, space details, and total cost on success.
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
                            f"Cost €{reservation['total_cost']:.2f}."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    @tool
    def request_admin_approval(reservation_id: int) -> str:
        """Interrupt the graph and wait for an admin to approve or reject.

        Resumes automatically once the admin decides via the Admin UI.
        Returns the admin's decision.
        """
        decision = interrupt({"reservation_id": reservation_id})
        status = decision.get("status", "unknown") if isinstance(decision, dict) else str(decision)
        logger.info("Admin decision for reservation #%d: %s", reservation_id, status)
        return f"Admin decision: {status}"

    @tool
    def write_reservation_to_log(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Persist the confirmed reservation to the MCP reservation log server.

        Reads all required fields from state automatically.
        Call this only after the admin has approved the reservation.
        """
        data: dict = state.get("reservation_data") or {}
        settings = get_settings()
        approval_time = datetime.now(tz=timezone.utc).replace(tzinfo=None).isoformat()

        async def _call_mcp() -> list:
            transport = SSETransport(f"http://{settings.mcp_server_host}:{settings.mcp_server_port}/sse")
            async with MCPClient(transport) as client:
                return await client.call_tool(
                    "write_confirmed_reservation",
                    {
                        "api_key": settings.mcp_api_key,
                        "reservation_id": data["reservation_id"],
                        "customer_name": data["name"],
                        "customer_surname": data["surname"],
                        "car_number": data["car_number"],
                        "start_datetime": data["start_datetime"],
                        "end_datetime": data["end_datetime"],
                        "approval_time": approval_time,
                    },
                )

        try:
            # Run in a dedicated thread so asyncio.run() always gets a fresh
            # event loop — safe regardless of whether the caller (Streamlit)
            # has a running loop in the current thread.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(asyncio.run, _call_mcp()).result(timeout=30)
            content = str(result[0].text) if result else "Logged."
            logger.info(
                "Reservation #%d logged via MCP server.", data.get("reservation_id")
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("MCP log write failed for reservation #%s: %s", data.get("reservation_id"), exc)
            content = f"Warning: reservation confirmed but MCP logging failed: {exc}"

        return Command(
            update={
                "messages": [
                    ToolMessage(content=content, tool_call_id=tool_call_id)
                ]
            }
        )

    return [create_pending_reservation, request_admin_approval, write_reservation_to_log]


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

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
        response_format=ToolStrategy(BookingResult),
    )
