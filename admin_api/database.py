"""Database access layer for the admin API."""
from __future__ import annotations

import math
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator

import psycopg2
import psycopg2.extras

from config import get_settings


@contextmanager
def get_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    s = get_settings()
    conn = psycopg2.connect(
        host=s.postgres_host,
        port=s.postgres_port,
        dbname=s.postgres_db,
        user=s.postgres_user,
        password=s.postgres_password,
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


def _get_price(price_type: str) -> float:
    """Fetch a single price amount by type."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT amount FROM prices WHERE price_type = %s", (price_type,)
            )
            row = cur.fetchone()
            return float(row["amount"]) if row else 0.0


def _calculate_cost(start: datetime, end: datetime) -> float:
    hourly_rate = _get_price("hourly") or 3.0
    daily_max = _get_price("daily_max") or 25.0

    hours = (end - start).total_seconds() / 3600
    days = hours / 24
    cost_hourly = hours * hourly_rate
    cost_daily = math.ceil(days) * daily_max
    return round(min(cost_hourly, cost_daily), 2)


def find_available_space(
    space_type: str, start: datetime, end: datetime
) -> dict[str, Any] | None:
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
                (space_type, end, start),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def get_space(space_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, floor, space_number, space_type, status FROM parking_spaces WHERE id = %s",
                (space_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def is_space_available(space_id: int, start: datetime, end: datetime) -> bool:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1 FROM reservations
                WHERE space_id = %s
                  AND status IN ('pending', 'confirmed')
                  AND start_datetime < %s
                  AND end_datetime   > %s
                LIMIT 1
                """,
                (space_id, end, start),
            )
            return cur.fetchone() is None


def create_reservation(
    space_id: int,
    customer_name: str,
    customer_surname: str,
    car_number: str,
    start: datetime,
    end: datetime,
    total_cost: float,
) -> dict[str, Any]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reservations
                    (space_id, customer_name, customer_surname, car_number,
                     start_datetime, end_datetime, total_cost, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending')
                RETURNING id, space_id, customer_name, customer_surname,
                          car_number, start_datetime, end_datetime,
                          total_cost, status, created_at
                """,
                (space_id, customer_name, customer_surname, car_number,
                 start, end, total_cost),
            )
            row = dict(cur.fetchone())
    # Attach space details
    space = get_space(space_id)
    if space:
        row["floor"] = space["floor"]
        row["space_number"] = space["space_number"]
    return row


def list_reservations(status: str | None = None) -> list[dict[str, Any]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            base = """
                SELECT r.*, ps.floor, ps.space_number
                FROM reservations r
                JOIN parking_spaces ps ON ps.id = r.space_id
            """
            if status:
                cur.execute(base + " WHERE r.status = %s ORDER BY r.created_at DESC", (status,))
            else:
                cur.execute(base + " ORDER BY r.created_at DESC")
            return [dict(row) for row in cur.fetchall()]


def get_reservation(reservation_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.*, ps.floor, ps.space_number
                FROM reservations r
                JOIN parking_spaces ps ON ps.id = r.space_id
                WHERE r.id = %s
                """,
                (reservation_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def update_reservation_status(reservation_id: int, new_status: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE reservations
                SET status = %s
                WHERE id = %s AND status = 'pending'
                RETURNING id
                """,
                (new_status, reservation_id),
            )
            row = cur.fetchone()
            if not row:
                return None
    return get_reservation(reservation_id)
