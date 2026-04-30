"""
Tests for the CityPark Admin API.

All database calls are mocked — no live PostgreSQL required.
"""
from __future__ import annotations

import base64
from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Patch settings before importing app so config doesn't fail
with patch.dict(
    "os.environ",
    {
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DB": "parking_db",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "postgres",
        "ADMIN_USERNAME": "admin",
        "ADMIN_PASSWORD": "admin",
    },
):
    from config import get_settings

    get_settings.cache_clear()
    from main import app

client = TestClient(app)

ADMIN_AUTH = ("admin", "admin")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reservation_row(
    id: int = 1,
    status: str = "pending",
    **overrides,
) -> dict:
    base = {
        "id": id,
        "space_id": 10,
        "floor": "B1",
        "space_number": "S01",
        "customer_name": "John",
        "customer_surname": "Doe",
        "car_number": "ABC-1234",
        "start_datetime": datetime(2026, 4, 15, 9, 0),
        "end_datetime": datetime(2026, 4, 15, 17, 0),
        "total_cost": 24.0,
        "status": status,
        "created_at": datetime(2026, 4, 14, 12, 0),
    }
    base.update(overrides)
    return base


def _create_body(**overrides) -> dict:
    base = {
        "customer_name": "John",
        "customer_surname": "Doe",
        "car_number": "ABC-1234",
        "start_datetime": "2026-04-15T09:00:00",
        "end_datetime": "2026-04-15T17:00:00",
        "space_type": "standard",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# POST /api/reservations
# ---------------------------------------------------------------------------

class TestCreateReservation:
    def test_creates_reservation_successfully(self):
        space = {"id": 10, "floor": "B1", "space_number": "S01", "space_type": "standard"}
        row = _reservation_row()

        with patch("main.db.find_available_space", return_value=space), \
             patch("main.db._calculate_cost", return_value=24.0), \
             patch("main.db.create_reservation", return_value=row):
            resp = client.post("/api/reservations", json=_create_body())

        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == 1
        assert data["status"] == "pending"
        assert data["floor"] == "B1"
        assert data["space_number"] == "S01"

    def test_rejects_invalid_period(self):
        body = _create_body(
            start_datetime="2026-04-15T17:00:00",
            end_datetime="2026-04-15T09:00:00",
        )
        resp = client.post("/api/reservations", json=body)
        assert resp.status_code == 400
        assert "end_datetime" in resp.json()["detail"]

    def test_returns_409_when_no_space_available(self):
        with patch("main.db.find_available_space", return_value=None):
            resp = client.post("/api/reservations", json=_create_body())

        assert resp.status_code == 409

    def test_specific_space_id_success(self):
        space = {"id": 5, "floor": "B2", "space_number": "C01", "space_type": "compact", "status": "operating"}
        row = _reservation_row(space_id=5, floor="B2", space_number="C01")

        with patch("main.db.get_space", return_value=space), \
             patch("main.db.is_space_available", return_value=True), \
             patch("main.db._calculate_cost", return_value=24.0), \
             patch("main.db.create_reservation", return_value=row):
            resp = client.post("/api/reservations", json=_create_body(space_id=5))

        assert resp.status_code == 201
        assert resp.json()["space_id"] == 5

    def test_specific_space_id_not_found(self):
        with patch("main.db.get_space", return_value=None):
            resp = client.post("/api/reservations", json=_create_body(space_id=999))

        assert resp.status_code == 404

    def test_specific_space_id_not_operating(self):
        space = {"id": 5, "floor": "B2", "space_number": "C01", "space_type": "compact", "status": "maintenance"}
        with patch("main.db.get_space", return_value=space):
            resp = client.post("/api/reservations", json=_create_body(space_id=5))

        assert resp.status_code == 409
        assert "maintenance" in resp.json()["detail"]

    def test_specific_space_id_already_reserved(self):
        space = {"id": 5, "floor": "B2", "space_number": "C01", "space_type": "compact", "status": "operating"}
        with patch("main.db.get_space", return_value=space), \
             patch("main.db.is_space_available", return_value=False):
            resp = client.post("/api/reservations", json=_create_body(space_id=5))

        assert resp.status_code == 409
        assert "already reserved" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# GET /api/reservations
# ---------------------------------------------------------------------------

class TestListReservations:
    def test_returns_all_reservations(self):
        rows = [_reservation_row(id=1), _reservation_row(id=2, status="confirmed")]
        with patch("main.db.list_reservations", return_value=rows):
            resp = client.get("/api/reservations")

        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_filters_by_status(self):
        rows = [_reservation_row(id=1)]
        with patch("main.db.list_reservations", return_value=rows) as mock:
            resp = client.get("/api/reservations?status=pending")

        assert resp.status_code == 200
        mock.assert_called_once_with("pending")

    def test_returns_empty_list(self):
        with patch("main.db.list_reservations", return_value=[]):
            resp = client.get("/api/reservations")

        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# GET /api/reservations/{id}
# ---------------------------------------------------------------------------

class TestGetReservation:
    def test_returns_reservation(self):
        row = _reservation_row(id=42)
        with patch("main.db.get_reservation", return_value=row):
            resp = client.get("/api/reservations/42")

        assert resp.status_code == 200
        assert resp.json()["id"] == 42

    def test_returns_404_for_missing(self):
        with patch("main.db.get_reservation", return_value=None):
            resp = client.get("/api/reservations/999")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/reservations/{id}/approve
# ---------------------------------------------------------------------------

class TestApproveReservation:
    def test_approves_pending_reservation(self):
        row = _reservation_row(id=1, status="pending")
        updated = _reservation_row(id=1, status="confirmed")

        with patch("main.db.get_reservation", return_value=row), \
             patch("main.db.update_reservation_status", return_value=updated):
            resp = client.post("/api/reservations/1/approve", auth=ADMIN_AUTH)

        assert resp.status_code == 200
        data = resp.json()
        assert data["reservation"]["status"] == "confirmed"
        assert "approved" in data["message"]

    def test_returns_401_without_auth(self):
        resp = client.post("/api/reservations/1/approve")
        assert resp.status_code == 401

    def test_returns_401_with_wrong_credentials(self):
        resp = client.post("/api/reservations/1/approve", auth=("wrong", "wrong"))
        assert resp.status_code == 401

    def test_returns_404_for_missing_reservation(self):
        with patch("main.db.get_reservation", return_value=None):
            resp = client.post("/api/reservations/999/approve", auth=ADMIN_AUTH)

        assert resp.status_code == 404

    def test_returns_409_for_non_pending(self):
        row = _reservation_row(id=1, status="confirmed")
        with patch("main.db.get_reservation", return_value=row):
            resp = client.post("/api/reservations/1/approve", auth=ADMIN_AUTH)

        assert resp.status_code == 409
        assert "confirmed" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# POST /api/reservations/{id}/reject
# ---------------------------------------------------------------------------

class TestRejectReservation:
    def test_rejects_pending_reservation(self):
        row = _reservation_row(id=1, status="pending")
        updated = _reservation_row(id=1, status="cancelled")

        with patch("main.db.get_reservation", return_value=row), \
             patch("main.db.update_reservation_status", return_value=updated):
            resp = client.post("/api/reservations/1/reject", auth=ADMIN_AUTH)

        assert resp.status_code == 200
        data = resp.json()
        assert data["reservation"]["status"] == "cancelled"
        assert "rejected" in data["message"]

    def test_returns_401_without_auth(self):
        resp = client.post("/api/reservations/1/reject")
        assert resp.status_code == 401

    def test_returns_404_for_missing_reservation(self):
        with patch("main.db.get_reservation", return_value=None):
            resp = client.post("/api/reservations/999/reject", auth=ADMIN_AUTH)

        assert resp.status_code == 404

    def test_returns_409_for_non_pending(self):
        row = _reservation_row(id=1, status="cancelled")
        with patch("main.db.get_reservation", return_value=row):
            resp = client.post("/api/reservations/1/reject", auth=ADMIN_AUTH)

        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
