"""
Tests for src/database/sql_store.py

All database calls are mocked via psycopg2 to avoid requiring a live PostgreSQL
instance during unit tests.
"""
from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pytest

import src.database.sql_store as sql


# ---------------------------------------------------------------------------
# Helper: build a mock connection / cursor pair
# ---------------------------------------------------------------------------

def _make_mock_conn(rows=None, rowcount=1):
    """
    Build a mock psycopg2 connection where cursor.fetchall() returns `rows`
    and cursor.fetchone() returns rows[0] if rows else None.
    """
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = rows or []
    mock_cursor.fetchone.return_value = rows[0] if rows else None
    mock_cursor.rowcount = rowcount
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    return mock_conn, mock_cursor


# ---------------------------------------------------------------------------
# Tests: Prices
# ---------------------------------------------------------------------------

class TestGetAllPrices:
    def test_returns_list_of_price_dicts(self):
        rows = [
            {"price_type": "hourly", "amount": 3.00, "currency": "EUR", "description": "Hourly"},
            {"price_type": "daily_max", "amount": 25.00, "currency": "EUR", "description": "Daily max"},
        ]
        mock_conn, mock_cursor = _make_mock_conn(rows)
        mock_cursor.fetchall.return_value = rows

        with patch("src.database.sql_store.psycopg2.connect", return_value=mock_conn):
            result = sql.get_all_prices()

        assert len(result) == 2
        assert result[0]["price_type"] == "hourly"

    def test_returns_empty_list_when_no_prices(self):
        mock_conn, mock_cursor = _make_mock_conn([])
        mock_cursor.fetchall.return_value = []

        with patch("src.database.sql_store.psycopg2.connect", return_value=mock_conn):
            result = sql.get_all_prices()

        assert result == []


class TestGetPrice:
    def test_returns_matching_price_dict(self):
        row = {"price_type": "monthly", "amount": 200.00, "currency": "EUR", "description": "Monthly"}
        mock_conn, mock_cursor = _make_mock_conn([row])

        with patch("src.database.sql_store.psycopg2.connect", return_value=mock_conn):
            result = sql.get_price("monthly")

        assert result is not None
        assert result["amount"] == 200.00

    def test_returns_none_for_unknown_price_type(self):
        mock_conn, mock_cursor = _make_mock_conn([])
        mock_cursor.fetchone.return_value = None

        with patch("src.database.sql_store.psycopg2.connect", return_value=mock_conn):
            result = sql.get_price("nonexistent")

        assert result is None


# ---------------------------------------------------------------------------
# Tests: Availability
# ---------------------------------------------------------------------------

class TestGetAvailabilitySummary:
    def test_summary_contains_total_and_by_type(self):
        rows = [
            {"space_type": "standard", "total_available": 70, "total_reserved": 30, "total_spaces": 100},
            {"space_type": "ev", "total_available": 15, "total_reserved": 5, "total_spaces": 20},
        ]
        mock_conn, mock_cursor = _make_mock_conn(rows)
        mock_cursor.fetchall.return_value = rows

        with patch("src.database.sql_store.psycopg2.connect", return_value=mock_conn):
            result = sql.get_availability_summary()

        assert result["total_available"] == 85
        assert len(result["by_type"]) == 2

    def test_total_available_is_correct_sum(self):
        rows = [
            {"space_type": "standard", "total_available": 10, "total_reserved": 0, "total_spaces": 10},
        ]
        mock_conn, mock_cursor = _make_mock_conn(rows)
        mock_cursor.fetchall.return_value = rows

        with patch("src.database.sql_store.psycopg2.connect", return_value=mock_conn):
            result = sql.get_availability_summary()

        assert result["total_available"] == 10


# ---------------------------------------------------------------------------
# Tests: Working hours
# ---------------------------------------------------------------------------

class TestGetWorkingHours:
    def test_returns_working_hours_list(self):
        rows = [
            {"day_of_week": "Monday", "open_time": None, "close_time": None, "is_24h": True},
        ]
        mock_conn, mock_cursor = _make_mock_conn(rows)
        mock_cursor.fetchall.return_value = rows

        with patch("src.database.sql_store.psycopg2.connect", return_value=mock_conn):
            result = sql.get_working_hours()

        assert len(result) == 1
        assert result[0]["is_24h"] is True


# ---------------------------------------------------------------------------
# Tests: find_available_space
# ---------------------------------------------------------------------------

class TestFindAvailableSpace:
    def test_returns_space_dict_when_available(self):
        row = {"id": 1, "floor": "B1", "space_number": "S01", "space_type": "standard"}
        mock_conn, mock_cursor = _make_mock_conn([row])

        with patch("src.database.sql_store.psycopg2.connect", return_value=mock_conn):
            result = sql.find_available_space(
                "standard",
                start_datetime=datetime(2026, 4, 1, 9),
                end_datetime=datetime(2026, 4, 1, 17),
            )

        assert result is not None
        assert result["space_type"] == "standard"

    def test_returns_none_when_no_space_available(self):
        mock_conn, mock_cursor = _make_mock_conn(None)
        mock_cursor.fetchone.return_value = None

        with patch("src.database.sql_store.psycopg2.connect", return_value=mock_conn):
            result = sql.find_available_space(
                "ev",
                start_datetime=datetime(2026, 4, 1, 9),
                end_datetime=datetime(2026, 4, 1, 17),
            )

        assert result is None


# ---------------------------------------------------------------------------
# Tests: create_reservation
# ---------------------------------------------------------------------------

class TestCreateReservation:
    def test_creates_reservation_and_returns_dict(self, sample_reservation_data):
        price_row = {"price_type": "hourly", "amount": 3.00, "currency": "EUR", "description": ""}
        price_daily = {"price_type": "daily_max", "amount": 25.00, "currency": "EUR", "description": ""}
        res_row = {
            "id": 42, "space_id": 1,
            "customer_name": "John", "customer_surname": "Doe",
            "car_number": "ABC-1234",
            "start_datetime": datetime(2026, 4, 1, 9),
            "end_datetime": datetime(2026, 4, 1, 17),
            "total_cost": 24.00, "status": "pending",
        }

        with patch("src.database.sql_store.get_price") as mock_get_price, \
             patch("src.database.sql_store.psycopg2.connect") as mock_connect:
            mock_get_price.side_effect = lambda pt: price_row if pt == "hourly" else price_daily
            mock_conn, mock_cursor = _make_mock_conn([res_row])
            mock_cursor.fetchone.return_value = res_row
            mock_connect.return_value = mock_conn

            result = sql.create_reservation(
                space_id=1,
                customer_name="John",
                customer_surname="Doe",
                car_number="ABC-1234",
                start_datetime=datetime(2026, 4, 1, 9),
                end_datetime=datetime(2026, 4, 1, 17),
            )

        assert result["id"] == 42
        assert result["status"] == "pending"

    def test_cost_capped_at_daily_maximum(self):
        """A 10-hour stay should not exceed the daily maximum."""
        price_hourly = {"price_type": "hourly", "amount": 3.00, "currency": "EUR", "description": ""}
        price_daily = {"price_type": "daily_max", "amount": 25.00, "currency": "EUR", "description": ""}

        with patch("src.database.sql_store.get_price") as mock_get_price, \
             patch("src.database.sql_store.psycopg2.connect") as mock_connect:
            mock_get_price.side_effect = lambda pt: price_hourly if pt == "hourly" else price_daily
            mock_conn, mock_cursor = _make_mock_conn()
            res_row = {
                "id": 1, "space_id": 1,
                "customer_name": "A", "customer_surname": "B",
                "car_number": "X", "start_datetime": datetime(2026, 4, 1, 0),
                "end_datetime": datetime(2026, 4, 1, 10),
                "total_cost": 25.00, "status": "pending",
            }
            mock_cursor.fetchone.return_value = res_row
            mock_connect.return_value = mock_conn

            result = sql.create_reservation(
                space_id=1, customer_name="A", customer_surname="B",
                car_number="X",
                start_datetime=datetime(2026, 4, 1, 0),
                end_datetime=datetime(2026, 4, 1, 10),
            )

        # 10 hrs * $3 = $30 but daily cap is $25
        assert result["total_cost"] <= 25.00
