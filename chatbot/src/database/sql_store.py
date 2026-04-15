"""
PostgreSQL store for dynamic parking data.

Dynamic data stored here:
- Parking space inventory and availability (real-time)
- Pricing tiers (hourly, daily, monthly, EV charging)
- Working hours
- Reservations (created during chatbot interactions)
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

import psycopg2
import psycopg2.extras

from src.config import get_settings

logger = logging.getLogger(__name__)


@contextmanager
def get_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """Context manager that yields a PostgreSQL connection and closes it on exit."""
    settings = get_settings()
    conn = psycopg2.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        cursor_factory=psycopg2.extras.RealDictCursor,
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def get_all_prices() -> List[Dict[str, Any]]:
    """Return all pricing tiers from the database."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT price_type, amount, currency, description FROM prices ORDER BY id")
            return [dict(row) for row in cur.fetchall()]


def get_price(price_type: str) -> Optional[Dict[str, Any]]:
    """Return a single price record by type (e.g. 'hourly', 'daily_max')."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT price_type, amount, currency, description FROM prices WHERE price_type = %s",
                (price_type,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


# ---------------------------------------------------------------------------
# Working hours
# ---------------------------------------------------------------------------

def get_working_hours() -> List[Dict[str, Any]]:
    """Return working hours for all days of the week."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT day_of_week, open_time, close_time, is_24h FROM working_hours ORDER BY id"
            )
            return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Parking space availability
# ---------------------------------------------------------------------------

def get_availability_summary() -> Dict[str, Any]:
    """Return a summary of available and occupied parking spaces by type.

    Availability is derived from the reservations table: a space is considered
    occupied if it has an active (pending or confirmed) reservation that covers
    the current moment.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    ps.space_type,
                    COUNT(*)                                           AS total_spaces,
                    COUNT(*) FILTER (WHERE r.id IS NULL)               AS total_available,
                    COUNT(*) FILTER (WHERE r.id IS NOT NULL)           AS total_occupied
                FROM parking_spaces ps
                LEFT JOIN reservations r
                    ON r.space_id = ps.id
                   AND r.status IN ('pending', 'confirmed')
                   AND r.start_datetime <= NOW()
                   AND r.end_datetime   >  NOW()
                WHERE ps.status = 'operating'
                GROUP BY ps.space_type
                ORDER BY ps.space_type
                """
            )
            rows = [dict(r) for r in cur.fetchall()]
            total_available = sum(r["total_available"] for r in rows)
            return {"by_type": rows, "total_available": total_available}


def find_available_space(
    space_type: str = "standard",
    start_datetime: Optional[datetime] = None,
    end_datetime: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """
    Find the first available parking space of the requested type for the given
    time window.  A space is considered unavailable if it has an active
    (pending or confirmed) reservation that overlaps the requested window.

    Args:
        space_type: One of 'standard', 'compact', 'handicapped', 'ev'.
        start_datetime: Start of the requested window (inclusive).
        end_datetime: End of the requested window (exclusive).

    Returns:
        A dict with space details or None if no spaces available.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ps.id, ps.floor, ps.space_number, ps.space_type
                FROM parking_spaces ps
                WHERE ps.status = 'operating'
                  AND ps.space_type = %s
                  AND NOT EXISTS (
                      SELECT 1 FROM reservations r
                      WHERE r.space_id = ps.id
                        AND r.status IN ('pending', 'confirmed')
                        AND r.start_datetime < %s
                        AND r.end_datetime   > %s
                  )
                ORDER BY ps.floor, ps.space_number
                LIMIT 1
                """,
                (space_type, end_datetime, start_datetime),
            )
            row = cur.fetchone()
            return dict(row) if row else None


# ---------------------------------------------------------------------------
# Reservations
# ---------------------------------------------------------------------------

def calculate_cost(start_datetime: datetime, end_datetime: datetime) -> float:
    """
    Calculate the total cost for a reservation window.

    Uses hourly rate capped at a daily maximum (ceil days).
    Fetches current rates from the prices table.
    """
    hourly_price_row = get_price("hourly")
    hourly_rate = float(hourly_price_row["amount"]) if hourly_price_row else 3.0
    daily_max_row = get_price("daily_max")
    daily_max = float(daily_max_row["amount"]) if daily_max_row else 25.0

    duration_hours = (end_datetime - start_datetime).total_seconds() / 3600
    duration_days = duration_hours / 24
    cost_hourly = duration_hours * hourly_rate
    cost_daily_cap = int(duration_days + 0.999) * daily_max  # ceil days
    return round(min(cost_hourly, cost_daily_cap), 2)


def create_reservation(
    space_id: int,
    customer_name: str,
    customer_surname: str,
    car_number: str,
    start_datetime: datetime,
    end_datetime: datetime,
) -> Dict[str, Any]:
    """
    Insert a new reservation and mark the space as reserved.

    Returns the newly created reservation record.
    """
    total_cost = calculate_cost(start_datetime, end_datetime)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reservations
                    (space_id, customer_name, customer_surname, car_number,
                     start_datetime, end_datetime, total_cost, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending')
                RETURNING id, space_id, customer_name, customer_surname,
                          car_number, start_datetime, end_datetime, total_cost, status
                """,
                (
                    space_id,
                    customer_name,
                    customer_surname,
                    car_number,
                    start_datetime,
                    end_datetime,
                    total_cost,
                ),
            )
            reservation = dict(cur.fetchone())

    logger.info(
        "Created reservation #%d for %s %s (car: %s)",
        reservation["id"],
        customer_name,
        customer_surname,
        car_number,
    )
    return reservation


def get_reservation_by_id(reservation_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a reservation by its ID."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM reservations WHERE id = %s",
                (reservation_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def confirm_reservation(reservation_id: int) -> bool:
    """Confirm a pending reservation (human-in-the-loop stage placeholder)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE reservations SET status = 'confirmed' WHERE id = %s AND status = 'pending'",
                (reservation_id,),
            )
            updated = cur.rowcount
    return updated > 0


def cancel_reservation(reservation_id: int) -> bool:
    """Cancel a reservation and free the parking space."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Get space_id first
            cur.execute("SELECT space_id FROM reservations WHERE id = %s", (reservation_id,))
            row = cur.fetchone()
            if not row:
                return False
            space_id = row["space_id"]

            cur.execute(
                "UPDATE reservations SET status = 'cancelled' WHERE id = %s",
                (reservation_id,),
            )
    return True
