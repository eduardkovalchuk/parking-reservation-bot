"""
Shared pytest fixtures for the parking chatbot test suite.

Fixtures are grouped by scope:
- session: expensive setup done once (e.g. patching env vars)
- function: fresh instance per test (e.g. mock clients)
"""
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Environment / settings fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Ensure tests never require a real .env file."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("WEAVIATE_HOST", "localhost")
    monkeypatch.setenv("WEAVIATE_PORT", "8080")
    monkeypatch.setenv("WEAVIATE_GRPC_PORT", "50051")
    monkeypatch.setenv("WEAVIATE_COLLECTION_NAME", "ParkingInfo")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "parking_db")
    monkeypatch.setenv("POSTGRES_USER", "parking_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "parking_password")
    monkeypatch.setenv("RETRIEVAL_K", "5")


@pytest.fixture()
def mock_weaviate_client():
    """Return a fully mocked Weaviate client."""
    client = MagicMock()
    client.collections.exists.return_value = True
    return client


@pytest.fixture()
def mock_db_connection(mocker):
    """Patch psycopg2.connect to avoid real database connections."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mocker.patch("psycopg2.connect", return_value=mock_conn)
    return mock_conn, mock_cursor


@pytest.fixture()
def sample_reservation_data():
    """Minimal valid reservation data dict (all user-collected fields)."""
    return {
        "name": "John",
        "surname": "Doe",
        "car_number": "ABC-1234",
        "start_datetime": "2026-04-01T09:00:00",
        "end_datetime": "2026-04-01T17:00:00",
        "space_type": "standard",
    }


@pytest.fixture()
def sample_reservation_data_full():
    """Reservation data dict including DB-assigned fields (space, cost)."""
    return {
        "name": "John",
        "surname": "Doe",
        "car_number": "ABC-1234",
        "start_datetime": "2026-04-01T09:00:00",
        "end_datetime": "2026-04-01T17:00:00",
        "space_type": "standard",
        "space_id": 1,
        "floor": "B1",
        "space_number": "S01",
        "reservation_id": 99,
        "total_cost": 24.0,
    }
