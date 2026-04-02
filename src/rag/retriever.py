"""
Hybrid retriever combining Weaviate (static) and PostgreSQL (dynamic) sources.

Query routing:
- Dynamic queries (prices, availability, hours) → SQL database
- Static queries (location, policies, amenities, FAQ) → Weaviate vector search
- Mixed queries → both sources, results merged
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import weaviate
from langchain_core.documents import Document

from src.database import sql_store, vector_store

logger = logging.getLogger(__name__)

# Keywords that signal the query is about dynamic data
_DYNAMIC_KEYWORDS = re.compile(
    r"\b(price|prices|cost|rate|rates|fee|fees|charge|charges|how much|"
    r"available|availability|open|closed|hours|schedule|space|spaces|spot|spots|"
    r"slots?|empty|full|capacity|ev|electric|charging)\b",
    re.IGNORECASE,
)


@dataclass
class RetrievalResult:
    """Container for retrieval results from both sources."""

    static_docs: List[Document] = field(default_factory=list)
    dynamic_context: str = ""
    combined_context: str = ""

    def to_context_string(self) -> str:
        parts: List[str] = []
        if self.static_docs:
            static_text = "\n\n".join(
                f"[Source: {doc.metadata.get('source', 'parking_info')}]\n{doc.page_content}"
                for doc in self.static_docs
            )
            parts.append(f"### Static Parking Information\n{static_text}")
        if self.dynamic_context:
            parts.append(f"### Live Parking Data\n{self.dynamic_context}")
        return "\n\n---\n\n".join(parts)


def _classify_query(query: str) -> tuple[bool, bool]:
    """
    Classify the query to determine which data sources to query.

    Returns:
        (needs_static, needs_dynamic) – booleans indicating which sources to use.
    """
    needs_dynamic = bool(_DYNAMIC_KEYWORDS.search(query))
    # Static info is almost always useful; only skip if clearly dynamic-only
    needs_static = True
    return needs_static, needs_dynamic


def _fetch_dynamic_context(query: str) -> str:
    """Fetch relevant dynamic data from PostgreSQL and format it as context."""
    parts: List[str] = []

    q = query.lower()

    # Prices
    if any(kw in q for kw in ("price", "cost", "rate", "fee", "how much", "charge")):
        try:
            prices = sql_store.get_all_prices()
            if prices:
                lines = [
                    f"  - {p['price_type']}: {p['currency']} {float(p['amount']):.2f}"
                    f"{' (' + p['description'] + ')' if p['description'] else ''}"
                    for p in prices
                ]
                parts.append("Current Pricing:\n" + "\n".join(lines))
        except Exception as exc:
            logger.warning("Failed to fetch prices: %s", exc)

    # Availability
    if any(kw in q for kw in ("available", "availability", "space", "spot", "slot", "empty", "full", "capacity")):
        try:
            summary = sql_store.get_availability_summary()
            total = summary["total_available"]
            by_type = summary["by_type"]
            lines = [
                f"  - {r['space_type'].capitalize()}: {r['total_available']} available / {r['total_spaces']} total"
                for r in by_type
            ]
            parts.append(
                f"Parking Space Availability (Total available: {total}):\n" + "\n".join(lines)
            )
        except Exception as exc:
            logger.warning("Failed to fetch availability: %s", exc)

    # Working hours
    if any(kw in q for kw in ("open", "closed", "hours", "schedule", "time", "when")):
        try:
            hours = sql_store.get_working_hours()
            if hours and hours[0].get("is_24h"):
                parts.append("Working Hours: Open 24 hours a day, 7 days a week (automated facility).")
            elif hours:
                lines = [
                    f"  - {h['day_of_week']}: {h['open_time']} – {h['close_time']}"
                    for h in hours
                ]
                parts.append("Working Hours:\n" + "\n".join(lines))
        except Exception as exc:
            logger.warning("Failed to fetch working hours: %s", exc)

    return "\n\n".join(parts)


def retrieve(
    query: str,
    weaviate_client: weaviate.WeaviateClient,
    k: int = 5,
) -> RetrievalResult:
    """
    Main retrieval function.  Fetches from Weaviate and/or PostgreSQL
    depending on the detected query intent.

    Args:
        query: The user's natural-language question.
        weaviate_client: Active Weaviate client instance.
        k: Number of vector search results to retrieve.

    Returns:
        A RetrievalResult with populated sources and a combined context string.
    """
    needs_static, needs_dynamic = _classify_query(query)
    result = RetrievalResult()

    if needs_static:
        try:
            result.static_docs = vector_store.similarity_search(
                query, client=weaviate_client, k=k
            )
        except Exception as exc:
            logger.warning("Weaviate retrieval failed: %s", exc)

    if needs_dynamic:
        result.dynamic_context = _fetch_dynamic_context(query)

    result.combined_context = result.to_context_string()
    return result
