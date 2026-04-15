"""
LangChain tools for the parking ReAct agent.

Tools:
  - retrieve_parking_info  : hybrid RAG retrieval (Weaviate + PostgreSQL)
  - get_reservation_draft  : read current reservation fields from state
  - update_reservation_draft: persist collected fields into state
"""
from __future__ import annotations

import logging
from typing import Annotated, Optional

import weaviate
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from src.config import get_settings
from src.database import sql_store
from src.rag import retriever as rag_retriever

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = ("name", "surname", "car_number", "start_datetime", "end_datetime", "space_type")


def create_tools(weaviate_client: weaviate.WeaviateClient) -> list:
    """
    Return the list of tools for the parking agent.
    The weaviate_client is closed over so retrieval tools have access to it.
    """
    settings = get_settings()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @tool
    def retrieve_parking_info(query: str) -> str:
        """Retrieve information about CityPark parking facility.

        Use for any question about: prices, availability, location, opening hours,
        policies, amenities, directions, EV charging, booking process, etc.
        Returns raw context — use it to compose your answer.
        """
        try:
            result = rag_retriever.retrieve(query, weaviate_client, k=settings.retrieval_k)
            return result.combined_context or "No relevant information found for that query."
        except Exception as exc:
            logger.warning("retrieve_parking_info failed: %s", exc)
            return "Retrieval unavailable right now. Suggest the user contact info@citypark.com."

    # ------------------------------------------------------------------
    # Reservation state tools
    # ------------------------------------------------------------------

    @tool
    def get_reservation_draft(state: Annotated[dict, InjectedState]) -> str:
        """Return the current reservation draft — which fields are filled and which are still missing."""
        data: dict = state.get("reservation_data") or {}

        if not data:
            return (
                "No reservation data collected yet.\n"
                f"Required fields: {', '.join(_REQUIRED_FIELDS)}\n"
                "Optional: space_type (standard | compact | handicapped | ev) — defaults to standard."
            )

        lines = []
        missing = []
        for field in _REQUIRED_FIELDS:
            value = data.get(field)
            if value:
                lines.append(f"  {field}: {value}")
            else:
                lines.append(f"  {field}: MISSING")
                missing.append(field)

        space_type = data.get("space_type", "standard (default)")
        lines.append(f"  space_type: {space_type}")

        summary = "Reservation draft:\n" + "\n".join(lines)
        if missing:
            summary += f"\n\nStill needed: {', '.join(missing)}"
        else:
            summary += "\n\nAll required fields collected — ready to show confirmation summary."
        return summary

    @tool
    def update_reservation_draft(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        name: Optional[str] = None,
        surname: Optional[str] = None,
        car_number: Optional[str] = None,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None,
        space_type: Optional[str] = None,
    ) -> Command:
        """Persist one or more reservation fields that the user just provided.

        Only pass fields the user has explicitly provided in this message.
        Convert any natural-language date/time to ISO-8601 before passing (e.g. '2026-04-20T09:00:00').
        Returns a state update — no need to echo the values back; the draft will
        be visible via get_reservation_draft.
        """
        current: dict = dict(state.get("reservation_data") or {})

        updates = {
            k: v
            for k, v in {
                "name": name,
                "surname": surname,
                "car_number": car_number,
                "start_datetime": start_datetime,
                "end_datetime": end_datetime,
                "space_type": space_type,
            }.items()
            if v is not None
        }
        current.update(updates)

        saved = ", ".join(f"{k}={v}" for k, v in updates.items())
        return Command(
            update={
                "reservation_data": current,
                "messages": [ToolMessage(content=f"Saved: {saved}", tool_call_id=tool_call_id)],
            }
        )

    @tool
    def calculate_reservation_cost(start_datetime: str, end_datetime: str) -> str:
        """Calculate the exact cost for a reservation given start and end datetimes.

        Use this when the user asks how much their reservation will cost before confirming.
        Accepts ISO-8601 strings (e.g. '2026-04-20T09:00:00').
        Returns the total cost in EUR with a breakdown (hourly rate vs daily cap).
        """
        from datetime import datetime as dt
        try:
            start = dt.fromisoformat(start_datetime)
            end = dt.fromisoformat(end_datetime)
        except ValueError as exc:
            return f"Could not parse datetimes: {exc}"

        if end <= start:
            return "End datetime must be after start datetime."

        total = sql_store.calculate_cost(start, end)
        duration_hours = (end - start).total_seconds() / 3600
        return (
            f"Estimated cost: €{total:.2f} "
            f"({duration_hours:.1f} hours at the current rates)."
        )

    @tool
    def submit_for_booking(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Hand off to the booking agent to create and confirm the reservation.

        Only call this after:
        1. ALL fields are collected (name, surname, car_number, start_datetime, end_datetime, space_type).
        2. The user has explicitly said "yes" / "confirm" / "proceed".
        Never call this proactively — always wait for explicit user confirmation.
        """
        data: dict = state.get("reservation_data") or {}
        missing = [f for f in _REQUIRED_FIELDS if not data.get(f)]
        if missing:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Cannot submit — still missing: {', '.join(missing)}. Collect them first.",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        return Command(
            update={
                "booking_requested": True,
                "messages": [
                    ToolMessage(
                        content="All fields collected. Handing off to booking agent.",
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return [
        retrieve_parking_info,
        get_reservation_draft,
        update_reservation_draft,
        calculate_reservation_cost,
        submit_for_booking,
    ]
